"""Imposto de renda sobre investimentos: tabela regressiva, isenções, ações e come-cotas.

As alíquotas vêm de `premissas.json`. As funções que recebem retornos
aceitam tanto um número quanto um array do numpy (um valor por cenário).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt

from talos.modelos import ErroDeDados, FaixaIR, Premissas, PremissasImpostos

MESES_POR_ANO = 12
TRIBUTACOES_COM_COME_COTAS = ("fundo_longo_prazo", "fundo_curto_prazo")

Retorno = Union[float, npt.NDArray[np.float64]]


def aliquota_por_faixa(dias: int, faixas: tuple[FaixaIR, ...]) -> float:
    """Devolve a alíquota da faixa em que cabe um investimento de `dias` dias corridos."""
    if dias < 0:
        raise ErroDeDados("impostos: o prazo em dias não pode ser negativo.")
    for faixa in faixas:
        if faixa.ate_dias is None or dias <= faixa.ate_dias:
            return faixa.aliquota
    raise ErroDeDados("impostos: a tabela de IR precisa terminar com uma faixa sem limite.")


def aliquota_ir(tributacao: str, dias: int, impostos: PremissasImpostos) -> float:
    """Alíquota de IR cobrada no resgate, para cada tipo de tributação.

    - `regressivo` e `fundo_longo_prazo`: tabela regressiva (22,5% a 15%);
    - `fundo_curto_prazo`: tabela de curto prazo (22,5% ou 20%);
    - `acoes`: alíquota fixa de ações e ETFs;
    - `isento`: zero (LCI, LCA, debêntures incentivadas).
    """
    if tributacao == "isento":
        return 0.0
    if tributacao == "acoes":
        return impostos.aliquota_acoes
    if tributacao == "fundo_curto_prazo":
        return aliquota_por_faixa(dias, impostos.tabela_fundo_curto_prazo)
    if tributacao in ("regressivo", "fundo_longo_prazo"):
        return aliquota_por_faixa(dias, impostos.tabela_regressiva)
    raise ErroDeDados(f"impostos: tributação '{tributacao}' não reconhecida.")


def aliquota_come_cotas(tributacao: str, impostos: PremissasImpostos) -> float:
    """Alíquota antecipada em maio e novembro (zero para quem não tem come-cotas)."""
    if tributacao == "fundo_longo_prazo":
        return impostos.come_cotas_longo_prazo
    if tributacao == "fundo_curto_prazo":
        return impostos.come_cotas_curto_prazo
    return 0.0


def meses_para_dias(prazo_meses: int, premissas: Premissas) -> int:
    """Converte um prazo em meses para dias corridos (12 meses = 365 dias)."""
    return round(prazo_meses * premissas.dias_corridos_por_ano / MESES_POR_ANO)


def retorno_liquido_anualizado(
    retorno_bruto_aa: Retorno,
    tributacao: str,
    prazo_meses: int,
    premissas: Premissas,
    mes_inicial: int | None = None,
) -> Retorno:
    """Retorno anual depois do IR, para quem aplica hoje e resgata em `prazo_meses`.

    `retorno_bruto_aa` já deve estar descontado da taxa de administração.
    `mes_inicial` (1 a 12) é o mês da aplicação, que define quando caem os
    come-cotas; se não for informado, usa o mês da data de referência das
    premissas.
    """
    _conferir_entradas(retorno_bruto_aa, prazo_meses)
    retorno = np.asarray(retorno_bruto_aa, dtype=float)
    anos = prazo_meses / MESES_POR_ANO
    aliquota_final = aliquota_ir(tributacao, meses_para_dias(prazo_meses, premissas), premissas.impostos)

    if tributacao in TRIBUTACOES_COM_COME_COTAS:
        mes = premissas.data_referencia.month if mes_inicial is None else mes_inicial
        fator = _fator_liquido_com_come_cotas(
            retorno, prazo_meses, mes, aliquota_final,
            aliquota_come_cotas(tributacao, premissas.impostos),
            premissas.impostos.meses_come_cotas,
        )
    else:
        fator_bruto = (1 + retorno) ** anos
        fator = fator_bruto - aliquota_final * np.maximum(fator_bruto - 1, 0)

    liquido = fator ** (1 / anos) - 1
    return float(liquido) if liquido.ndim == 0 else liquido


def percentual_cdi_equivalente(
    percentual_cdi_isento: float, prazo_meses: int, premissas: Premissas
) -> float:
    """Quanto um CDB tributado precisaria pagar para empatar com um produto isento.

    Usa a conta de mercado: percentual isento / (1 - alíquota do CDB no mesmo
    prazo). Exemplo: LCA de 90% do CDI em 2 anos (IR de 15%) equivale a um
    CDB de 90% / 0,85 = 105,9% do CDI.
    """
    dias = meses_para_dias(prazo_meses, premissas)
    aliquota = aliquota_ir("regressivo", dias, premissas.impostos)
    return percentual_cdi_isento / (1 - aliquota)


def _fator_liquido_com_come_cotas(
    retorno_aa: npt.NDArray[np.float64],
    prazo_meses: int,
    mes_inicial: int,
    aliquota_final: float,
    aliquota_antecipada: float,
    meses_come_cotas: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Quanto cada R$ 1 aplicado vira, líquido de IR, num fundo com come-cotas.

    Simula mês a mês com a cota começando em 1. No fim de cada mês de
    come-cotas, cobra a alíquota antecipada sobre a valorização desde o
    último come-cotas, reduzindo a quantidade de cotas. No resgate, cobra a
    diferença entre a alíquota final e o que já foi antecipado sobre as
    cotas que sobraram.
    """
    crescimento_mensal = (1 + retorno_aa) ** (1 / MESES_POR_ANO)
    cotas = np.ones_like(retorno_aa)
    preco = np.ones_like(retorno_aa)
    base = np.ones_like(retorno_aa)
    for passo in range(prazo_meses):
        preco = preco * crescimento_mensal
        mes_do_calendario = (mes_inicial - 1 + passo) % MESES_POR_ANO + 1
        if mes_do_calendario in meses_come_cotas:
            ganho_por_cota = np.maximum(preco - base, 0)
            imposto = aliquota_antecipada * cotas * ganho_por_cota
            cotas = cotas - imposto / preco
            base = np.maximum(base, preco)
    ja_antecipado = aliquota_antecipada * (base - 1)
    devido = aliquota_final * np.maximum(preco - 1, 0)
    complemento = cotas * np.maximum(devido - ja_antecipado, 0)
    return cotas * preco - complemento


def _conferir_entradas(retorno_bruto_aa: Retorno, prazo_meses: int) -> None:
    """Recusa prazo menor que 1 mês e perdas de 100% ou mais ao ano."""
    if prazo_meses < 1:
        raise ErroDeDados("impostos: o prazo precisa ser de pelo menos 1 mês.")
    if np.any(np.asarray(retorno_bruto_aa) <= -1):
        raise ErroDeDados("impostos: retorno bruto de -100% ou pior não é possível.")

