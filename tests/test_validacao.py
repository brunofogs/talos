"""Testes de `validacao.py`: o Talos contra 100% do CDI e pesos iguais, nos mesmos cenários."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from talos.mercado import CurvaDI, carregar_curva_di, gerar_cenarios
from talos.modelos import Meta, Premissas, Produto, carregar_cliente, carregar_prateleira, carregar_premissas
from talos.otimizador import ATINGIVEL, Recomendacao, fatores_liquidos, filtrar_produtos, otimizar_cliente
from talos.validacao import (
    NOME_CDI,
    NOME_PESOS_IGUAIS,
    NOME_TALOS,
    ComparacaoMeta,
    Estatisticas,
    conclusao,
    estatisticas,
    resumo_validacao,
    validar,
)

EXEMPLOS = Path(__file__).resolve().parent.parent / "exemplos"
CENARIOS = 1000
META_1_ANO = Meta("Meta", 100.0, 115.0, 12, False)


@pytest.fixture(scope="module")
def entradas() -> tuple[Premissas, CurvaDI, list[Produto]]:
    """Premissas, curva e prateleira de exemplo."""
    premissas = carregar_premissas(EXEMPLOS / "premissas.json")
    return premissas, carregar_curva_di(EXEMPLOS / "curva_di.csv", premissas), carregar_prateleira(EXEMPLOS / "prateleira.csv")


@pytest.fixture(scope="module")
def recomendacao(entradas) -> Recomendacao:
    """Recomendação do cliente de exemplo."""
    premissas, curva, produtos = entradas
    return otimizar_cliente(carregar_cliente(EXEMPLOS / "cliente.json"), produtos, premissas, curva, quantidade=CENARIOS)


def so_a_reserva():
    """Cliente de exemplo só com a reserva de emergência (12 meses, de R$ 12 mil para R$ 13 mil)."""
    cliente = carregar_cliente(EXEMPLOS / "cliente.json")
    return replace(cliente, metas=(cliente.metas[0],))


def test_estatisticas_a_mao() -> None:
    """R$ 1 vira 1,2; 1,1; 0,9 ou 1,3. Alvo 1,15 em 1 ano, nível 75% (o pior cenário):
    média 1,125 (12,5% a.a.), pior cenário -10%, chance 2 em 4 = 50%.
    """
    resultado = estatisticas("X", [1.2, 1.1, 0.9, 1.3], META_1_ANO, 0.75)
    assert resultado.retorno_esperado_aa == pytest.approx(0.125)
    assert resultado.resultado_piores == pytest.approx(-0.10)
    assert resultado.probabilidade_sucesso == pytest.approx(0.5)


def test_talos_igual_ao_otimizador(recomendacao: Recomendacao, entradas) -> None:
    """Nos mesmos cenários, a linha do Talos repete os números do otimizador."""
    premissas, curva, produtos = entradas
    comparacoes = validar(recomendacao, produtos, premissas, curva, CENARIOS)
    for comparacao, resultado in zip(comparacoes, recomendacao.resultados):
        assert comparacao.talos.carteira == NOME_TALOS
        assert comparacao.talos.retorno_esperado_aa == pytest.approx(resultado.retorno_esperado_aa)
        assert comparacao.talos.resultado_piores == pytest.approx(-resultado.cvar)
        assert comparacao.talos.probabilidade_sucesso == pytest.approx(resultado.probabilidade_sucesso)


def test_cdi_sem_incerteza_a_mao(entradas) -> None:
    """Com volatilidade zero no CDI, 100% do CDI em 1 ano rende 14% bruto e
    14% x (1 - 17,5%) = 11,55% líquido, em todos os cenários.
    """
    premissas, curva, produtos = entradas
    classes = dict(premissas.cenarios.classes)
    classes["pos_fixado"] = replace(classes["pos_fixado"], volatilidade_aa=0.0)
    sem_incerteza = replace(premissas, cenarios=replace(premissas.cenarios, classes=classes))
    recomendacao = otimizar_cliente(so_a_reserva(), produtos, sem_incerteza, curva, quantidade=CENARIOS)
    cdi = validar(recomendacao, produtos, sem_incerteza, curva, CENARIOS)[0].cdi
    assert cdi.carteira == NOME_CDI
    assert cdi.retorno_esperado_aa == pytest.approx(0.14 * (1 - 0.175))
    assert cdi.resultado_piores == pytest.approx(0.14 * (1 - 0.175))
    assert cdi.probabilidade_sucesso == 1.0


def test_pesos_iguais_entre_os_elegiveis(recomendacao: Recomendacao, entradas) -> None:
    """Pesos iguais = média simples dos produtos que passam no filtro da meta (Entrada, 4 anos)."""
    premissas, curva, produtos = entradas
    resultado = recomendacao.resultados[1]
    elegiveis, _ = filtrar_produtos(produtos, resultado.meta, recomendacao.perfil, premissas)
    cenarios = gerar_cenarios(premissas, curva, resultado.meta.prazo_meses, resultado.semente, CENARIOS)
    patrimonio = fatores_liquidos(elegiveis, cenarios, premissas).mean(axis=1)
    pesos_iguais = validar(recomendacao, produtos, premissas, curva, CENARIOS)[1].pesos_iguais
    assert pesos_iguais.carteira == NOME_PESOS_IGUAIS
    assert pesos_iguais.retorno_esperado_aa == pytest.approx(np.mean(patrimonio) ** (1 / 4) - 1)


def test_meta_sem_produtos(entradas) -> None:
    """Sem nenhum produto elegível: não há Talos nem pesos iguais, só 100% do CDI."""
    premissas, curva, produtos = entradas
    acoes = [p for p in produtos if p.renda_variavel]
    recomendacao = otimizar_cliente(so_a_reserva(), acoes, premissas, curva, quantidade=CENARIOS)
    comparacao = validar(recomendacao, acoes, premissas, curva, CENARIOS)[0]
    assert comparacao.talos is None and comparacao.pesos_iguais is None
    assert comparacao.linhas == (comparacao.cdi,)
    assert conclusao(comparacao).startswith("O Talos não montou carteira para esta meta.")


# --- Frases por regras ------------------------------------------------------------------------

def comparacao_manual(talos: tuple, cdi: tuple, iguais: tuple) -> ComparacaoMeta:
    """Comparação montada à mão: cada tupla é (retorno, resultado nos piores, chance)."""
    return ComparacaoMeta(
        META_1_ANO, ATINGIVEL,
        Estatisticas(NOME_TALOS, *talos), Estatisticas(NOME_CDI, *cdi), Estatisticas(NOME_PESOS_IGUAIS, *iguais),
    )


def test_conclusao_melhor_e_pior() -> None:
    """Cada quesito compara com a melhor referência naquele quesito."""
    comparacao = comparacao_manual((0.11, 0.20, 0.99), (0.12, 0.10, 0.90), (0.10, 0.15, 0.95))
    assert conclusao(comparacao) == (
        "Frente à melhor referência em cada quesito, o Talos é melhor em chance de sucesso "
        "(99,0% contra 95,0% de pesos iguais); é melhor em resultado nos 5% piores cenários "
        "(+20,00% contra +15,00% de pesos iguais); é pior em retorno esperado "
        "(11,00% contra 12,00% de 100% do CDI)."
    )


def test_conclusao_tudo_empatado() -> None:
    """Diferenças dentro da margem de empate nos três quesitos geram uma frase só."""
    comparacao = comparacao_manual((0.1100, 0.0800, 0.987), (0.1098, 0.0797, 0.987), (0.10, 0.07, 0.95))
    assert conclusao(comparacao).startswith("O Talos empata com a melhor referência")


def test_resumo_conta_as_metas() -> None:
    """Uma meta em que o Talos é melhor em tudo e outra em que perde em chance."""
    boa = comparacao_manual((0.11, 0.20, 0.99), (0.12, 0.10, 0.90), (0.10, 0.15, 0.95))
    ruim = comparacao_manual((0.11, 0.20, 0.80), (0.12, 0.10, 0.90), (0.10, 0.15, 0.95))
    assert resumo_validacao([boa, ruim]) == (
        "Resumo: em 1 de 2 metas o Talos tem chance de sucesso igual ou maior que as duas referências, "
        "e em 2 de 2 tem resultado igual ou melhor nos 5% piores cenários."
    )
