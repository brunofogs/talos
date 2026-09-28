"""Testes de `impostos.py`, com contas feitas à mão usando as premissas de exemplo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from talos.impostos import (
    aliquota_ir,
    meses_para_dias,
    percentual_cdi_equivalente,
    retorno_liquido_anualizado,
)
from talos.modelos import ErroDeDados, Premissas, carregar_premissas


@pytest.fixture
def premissas(exemplos: Path) -> Premissas:
    """Premissas de exemplo (IR regressivo 22,5% / 20% / 17,5% / 15%)."""
    return carregar_premissas(exemplos / "premissas.json")


# --- Alíquotas ----------------------------------------------------------------

@pytest.mark.parametrize(
    ("dias", "aliquota"),
    [
        (0, 0.225), (180, 0.225),
        (181, 0.200), (360, 0.200),
        (361, 0.175), (720, 0.175),
        (721, 0.150), (3650, 0.150),
    ],
)
def test_tabela_regressiva_nas_bordas(premissas: Premissas, dias: int, aliquota: float) -> None:
    """Cada faixa da tabela regressiva vale até o último dia dela, inclusive."""
    assert aliquota_ir("regressivo", dias, premissas.impostos) == aliquota
    assert aliquota_ir("fundo_longo_prazo", dias, premissas.impostos) == aliquota


@pytest.mark.parametrize(("dias", "aliquota"), [(180, 0.225), (181, 0.20), (2000, 0.20)])
def test_fundo_curto_prazo(premissas: Premissas, dias: int, aliquota: float) -> None:
    """Fundo de curto prazo: 22,5% até 180 dias e 20% depois."""
    assert aliquota_ir("fundo_curto_prazo", dias, premissas.impostos) == aliquota


@pytest.mark.parametrize("dias", [30, 400, 3000])
def test_isentos_e_acoes_nao_dependem_do_prazo(premissas: Premissas, dias: int) -> None:
    """LCI/LCA/incentivadas pagam 0%; ações e ETFs pagam 15% em qualquer prazo."""
    assert aliquota_ir("isento", dias, premissas.impostos) == 0.0
    assert aliquota_ir("acoes", dias, premissas.impostos) == 0.15


def test_tributacao_desconhecida_gera_erro(premissas: Premissas) -> None:
    """Uma tributação que não existe é recusada."""
    with pytest.raises(ErroDeDados, match="não reconhecida"):
        aliquota_ir("poupanca", 100, premissas.impostos)


@pytest.mark.parametrize(("meses", "dias"), [(1, 30), (12, 365), (24, 730), (60, 1825)])
def test_meses_para_dias(premissas: Premissas, meses: int, dias: int) -> None:
    """Meses viram dias corridos na proporção 12 meses = 365 dias."""
    assert meses_para_dias(meses, premissas) == dias


# --- Retorno líquido sem come-cotas --------------------------------------------

def test_cdb_dois_anos(premissas: Premissas) -> None:
    """CDB a 10% a.a. por 2 anos (730 dias, IR de 15%).

    R$ 1 vira 1,1^2 = 1,21. Ganho de 0,21; IR de 15% = 0,0315.
    Líquido: 1,1785, ou seja, raiz(1,1785) - 1 = 8,558% a.a.
    """
    liquido = retorno_liquido_anualizado(0.10, "regressivo", 24, premissas)
    assert liquido == pytest.approx(1.1785 ** 0.5 - 1)
    assert liquido == pytest.approx(0.08558, abs=1e-5)


def test_cdb_seis_meses(premissas: Premissas) -> None:
    """CDB a 21% a.a. por 6 meses (182 dias, IR de 20%).

    R$ 1 vira raiz(1,21) = 1,10. IR de 20% sobre 0,10 = 0,02. Líquido 1,08
    em meio ano, que anualizado dá 1,08^2 - 1 = 16,64%.
    """
    assert retorno_liquido_anualizado(0.21, "regressivo", 6, premissas) == pytest.approx(0.1664)


def test_lca_isenta_nao_perde_nada(premissas: Premissas) -> None:
    """Na LCA o retorno líquido é igual ao bruto."""
    assert retorno_liquido_anualizado(0.12, "isento", 18, premissas) == pytest.approx(0.12)


def test_acoes_pagam_15_por_cento_do_ganho(premissas: Premissas) -> None:
    """ETF a 10% a.a. por 3 anos: 1,1^3 = 1,331; IR = 15% de 0,331 = 0,04965.

    Líquido: 1,28135 em 3 anos.
    """
    liquido = retorno_liquido_anualizado(0.10, "acoes", 36, premissas)
    assert liquido == pytest.approx(1.28135 ** (1 / 3) - 1)


def test_prejuizo_nao_paga_imposto(premissas: Premissas) -> None:
    """Com perda de 10% em 1 ano não há IR: o líquido é a própria perda."""
    assert retorno_liquido_anualizado(-0.10, "acoes", 12, premissas) == pytest.approx(-0.10)


# --- Come-cotas -----------------------------------------------------------------

def test_fundo_longo_prazo_com_dois_come_cotas(premissas: Premissas) -> None:
    """Fundo a 21% a.a., aplicado em dezembro e resgatado 12 meses depois.

    A cota sobe 10% por semestre (1,1 x 1,1 = 1,21).
    - Fim de maio: cota 1,10. Come-cotas de 15% sobre 0,10 = 0,015,
      pagos com 0,015 / 1,10 cotas.
    - Fim de novembro: cota 1,21. Come-cotas de 15% sobre 0,11 por cota.
    - Resgate com 365 dias: alíquota final de 17,5%. Falta pagar
      17,5% - 15% = 2,5% sobre o ganho de 0,21 por cota que sobrou.
    """
    cotas = 1 - 0.015 / 1.10
    cotas -= 0.15 * cotas * 0.11 / 1.21
    final = cotas * 1.21 - cotas * (0.175 - 0.15) * 0.21

    liquido = retorno_liquido_anualizado(0.21, "fundo_longo_prazo", 12, premissas, mes_inicial=12)
    assert liquido == pytest.approx(final - 1)
    assert liquido == pytest.approx(0.172117, abs=1e-6)


def test_come_cotas_custa_mais_que_cdb_na_mesma_taxa(premissas: Premissas) -> None:
    """A antecipação do IR tira rendimento: o fundo rende menos que o CDB com a mesma taxa bruta."""
    fundo = retorno_liquido_anualizado(0.21, "fundo_longo_prazo", 12, premissas, mes_inicial=12)
    cdb = retorno_liquido_anualizado(0.21, "regressivo", 12, premissas)
    assert cdb == pytest.approx(0.21 * (1 - 0.175))
    assert fundo < cdb


def test_fundo_sem_come_cotas_no_periodo_igual_ao_cdb(premissas: Premissas) -> None:
    """Aplicado em junho e resgatado em 4 meses, o fundo não passa por maio nem novembro."""
    fundo = retorno_liquido_anualizado(0.15, "fundo_longo_prazo", 4, premissas, mes_inicial=6)
    cdb = retorno_liquido_anualizado(0.15, "regressivo", 4, premissas)
    assert fundo == pytest.approx(cdb)


def test_fundo_curto_prazo_come_cotas_igual_a_aliquota_final(premissas: Premissas) -> None:
    """Fundo curto prazo por 12 meses: come-cotas e alíquota final são 20%, sem complemento.

    Mesmo roteiro do teste de longo prazo, com 20% no lugar de 15%.
    """
    cotas = 1 - 0.20 * 0.10 / 1.10
    cotas -= 0.20 * cotas * 0.11 / 1.21
    liquido = retorno_liquido_anualizado(0.21, "fundo_curto_prazo", 12, premissas, mes_inicial=12)
    assert liquido == pytest.approx(cotas * 1.21 - 1)


def test_mes_inicial_padrao_vem_da_data_de_referencia(premissas: Premissas) -> None:
    """Sem mês inicial, usa o mês da data de referência (setembro, nas premissas de exemplo)."""
    assert premissas.data_referencia.month == 9
    padrao = retorno_liquido_anualizado(0.13, "fundo_longo_prazo", 30, premissas)
    setembro = retorno_liquido_anualizado(0.13, "fundo_longo_prazo", 30, premissas, mes_inicial=9)
    assert padrao == setembro


# --- Vários cenários de uma vez ----------------------------------------------------

@pytest.mark.parametrize("tributacao", ["regressivo", "isento", "acoes", "fundo_longo_prazo"])
def test_aceita_array_de_cenarios(premissas: Premissas, tributacao: str) -> None:
    """Com um array de retornos, cada posição dá o mesmo que a conta feita sozinha."""
    brutos = np.array([-0.30, 0.0, 0.08, 0.25])
    liquidos = retorno_liquido_anualizado(brutos, tributacao, 30, premissas)
    assert isinstance(liquidos, np.ndarray)
    esperados = [retorno_liquido_anualizado(float(r), tributacao, 30, premissas) for r in brutos]
    assert liquidos == pytest.approx(esperados)


@pytest.mark.parametrize(("bruto", "prazo"), [(-1.0, 12), (0.10, 0)])
def test_entradas_impossiveis_geram_erro(premissas: Premissas, bruto: float, prazo: int) -> None:
    """Perda de 100% ao ano ou prazo zero são recusados."""
    with pytest.raises(ErroDeDados):
        retorno_liquido_anualizado(bruto, "regressivo", prazo, premissas)


# --- Equivalência de isentos ---------------------------------------------------------

@pytest.mark.parametrize(
    ("meses", "esperado"),
    [(24, 0.90 / 0.85), (12, 0.90 / 0.825), (3, 0.90 / 0.775)],
)
def test_lca_equivalente_a_cdb(premissas: Premissas, meses: int, esperado: float) -> None:
    """LCA de 90% do CDI: em 2 anos equivale a CDB de 105,9% do CDI; em 1 ano, 109,1%."""
    assert percentual_cdi_equivalente(0.90, meses, premissas) == pytest.approx(esperado)
