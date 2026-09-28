"""Testes de `fgc.py`: limites por conglomerado e teto global, com contas à mão."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

from talos.fgc import (
    formatar_reais,
    restricoes_fgc,
    somar_usos,
    uso_por_conglomerado,
    violacoes_fgc,
)
from talos.modelos import ErroDeDados, Produto, PremissasFGC, carregar_premissas


@pytest.fixture
def fgc(exemplos: Path) -> PremissasFGC:
    """Limites de exemplo: R$ 250 mil por conglomerado e R$ 1 milhão por CPF."""
    return carregar_premissas(exemplos / "premissas.json").fgc


def produto(nome: str, conglomerado: str, coberto: bool = True) -> Produto:
    """Cria um produto simples de um conglomerado."""
    return Produto(
        nome=nome, emissor=nome, conglomerado=conglomerado, indexador="cdi",
        taxa=1.1, taxa_adm_aa=0.0, vencimento_meses=24, carencia_meses=0,
        liquidez_diaria=False, tributacao="regressivo", classe="pos_fixado",
        risco=2, coberto_fgc=coberto, aplicacao_minima=0.0, setor="financeiro",
    )


BOREAL_A = produto("CDB Banco Boreal", "Boreal")
BOREAL_B = produto("CDB Financeira Boreal", "Boreal")
TESOURO = produto("Tesouro Selic", "Tesouro Nacional", coberto=False)
AURORA = produto("CDB Aurora", "Aurora")
PRODUTOS = [BOREAL_A, BOREAL_B, TESOURO, AURORA]


def por_descricao(restricoes: list) -> dict:
    """Organiza as restrições pelo texto da descrição."""
    return {r.descricao: r for r in restricoes}


# --- Restrições para o otimizador ----------------------------------------------------

def test_restricoes_somam_o_conglomerado_e_ignoram_nao_cobertos(fgc: PremissasFGC) -> None:
    """Meta de R$ 400 mil: os dois Boreal dividem o mesmo limite; o Tesouro fica de fora."""
    restricoes = por_descricao(restricoes_fgc(PRODUTOS, 400_000, {}, fgc))
    assert set(restricoes) == {"FGC: limite do Aurora", "FGC: limite do Boreal", "FGC: teto global por CPF"}

    boreal = restricoes["FGC: limite do Boreal"]
    assert boreal.coeficientes.tolist() == [400_000, 400_000, 0, 0]
    assert boreal.limite == 250_000

    teto = restricoes["FGC: teto global por CPF"]
    assert teto.coeficientes.tolist() == [400_000, 400_000, 0, 400_000]
    assert teto.limite == 1_000_000


@pytest.mark.parametrize(("uso_boreal", "folga"), [(100_000, 150_000), (250_000, 0), (300_000, 0)])
def test_uso_de_metas_anteriores_reduz_a_folga(fgc: PremissasFGC, uso_boreal: float, folga: float) -> None:
    """Se metas anteriores já usaram parte do limite do Boreal, sobra só a diferença (nunca negativa)."""
    restricoes = por_descricao(restricoes_fgc(PRODUTOS, 400_000, {"Boreal": uso_boreal}, fgc))
    assert restricoes["FGC: limite do Boreal"].limite == folga
    assert restricoes["FGC: limite do Aurora"].limite == 250_000


def test_teto_global_com_uso_previo(fgc: PremissasFGC) -> None:
    """R$ 900 mil já aplicados em outros quatro grupos deixam só R$ 100 mil de teto global."""
    uso = {"G1": 225_000, "G2": 225_000, "G3": 225_000, "G4": 225_000}
    restricoes = por_descricao(restricoes_fgc(PRODUTOS, 400_000, uso, fgc))
    assert restricoes["FGC: teto global por CPF"].limite == pytest.approx(100_000)


def test_juros_contam_para_o_limite(fgc: PremissasFGC) -> None:
    """Se o CDB Boreal A cresce 25% até o vencimento, cada R$ 1 aplicado conta como R$ 1,25."""
    restricoes = por_descricao(
        restricoes_fgc(PRODUTOS, 400_000, {}, fgc, fatores_crescimento=[1.25, 1.0, 1.0, 1.0])
    )
    assert restricoes["FGC: limite do Boreal"].coeficientes.tolist() == [500_000, 400_000, 0, 0]


def test_sem_produtos_cobertos_nao_ha_restricao(fgc: PremissasFGC) -> None:
    """Uma carteira só com Tesouro não tem restrição de FGC."""
    assert restricoes_fgc([TESOURO], 400_000, {}, fgc) == []


@pytest.mark.parametrize(
    ("valor", "fatores", "mensagem"),
    [(0, None, "maior que zero"), (1000, [1.0], "cada produto")],
)
def test_entradas_invalidas(fgc: PremissasFGC, valor: float, fatores: list | None, mensagem: str) -> None:
    """Valor investido zero ou fatores de crescimento faltando geram erro."""
    with pytest.raises(ErroDeDados, match=mensagem):
        restricoes_fgc(PRODUTOS, valor, {}, fgc, fatores_crescimento=fatores)


def test_otimizador_respeita_limite_do_fgc(fgc: PremissasFGC) -> None:
    """Pedindo o máximo possível nos CDBs Boreal para uma meta de R$ 400 mil,
    o linprog para em 250 / 400 = 62,5% somando os dois.
    """
    restricoes = restricoes_fgc(PRODUTOS, 400_000, {}, fgc)
    resultado = linprog(
        c=[-1, -1, 0, 0],
        A_ub=np.array([r.coeficientes for r in restricoes]),
        b_ub=[r.limite for r in restricoes],
        A_eq=[[1, 1, 1, 1]], b_eq=[1],
        bounds=[(0, None)] * 4,
        method="highs",
    )
    assert resultado.success
    assert resultado.x[0] + resultado.x[1] == pytest.approx(0.625)
    assert violacoes_fgc(zip(PRODUTOS, resultado.x * 400_000), fgc) == []


# --- Conferência de uma carteira em reais ---------------------------------------------

def test_uso_por_conglomerado_soma_so_os_cobertos() -> None:
    """R$ 100 mil + R$ 50 mil no Boreal, R$ 80 mil no Tesouro (não conta)."""
    uso = uso_por_conglomerado([(BOREAL_A, 100_000), (BOREAL_B, 50_000), (TESOURO, 80_000)])
    assert uso == {"Boreal": 150_000}


def test_somar_usos_de_varias_metas() -> None:
    """O uso de duas metas é somado por conglomerado."""
    assert somar_usos({"Boreal": 100.0}, {"Boreal": 50.0, "Aurora": 10.0}) == {"Boreal": 150.0, "Aurora": 10.0}


def test_carteira_dentro_dos_limites(fgc: PremissasFGC) -> None:
    """R$ 250 mil exatos no Boreal ainda estão dentro do limite."""
    assert violacoes_fgc([(BOREAL_A, 200_000), (BOREAL_B, 50_000)], fgc) == []


def test_fracao_de_centavo_nao_e_violacao(fgc: PremissasFGC) -> None:
    """O otimizador pode passar do limite por frações de centavo (tolerância numérica do solver).
    Isso é arredondamento: R$ 250.000,004 não conta como violação, R$ 250.000,02 conta.
    """
    assert violacoes_fgc([(BOREAL_A, 250_000.004)], fgc) == []
    assert len(violacoes_fgc([(BOREAL_A, 250_000.02)], fgc)) == 1


def test_carteira_acima_do_limite_do_conglomerado(fgc: PremissasFGC) -> None:
    """R$ 200 mil + R$ 60 mil no Boreal estouram o limite de R$ 250 mil."""
    mensagens = violacoes_fgc([(BOREAL_A, 200_000), (BOREAL_B, 60_000), (TESOURO, 500_000)], fgc)
    assert mensagens == [
        "Boreal: R$ 260.000,00 em produtos cobertos, acima do limite de R$ 250.000,00 por conglomerado."
    ]


def test_uso_previo_entra_na_conferencia(fgc: PremissasFGC) -> None:
    """R$ 100 mil de uma meta anterior + R$ 200 mil agora estouram o limite do Aurora."""
    mensagens = violacoes_fgc([(AURORA, 200_000)], fgc, uso_previo={"Aurora": 100_000})
    assert len(mensagens) == 1
    assert mensagens[0].startswith("Aurora: R$ 300.000,00")


def test_carteira_acima_do_teto_global(fgc: PremissasFGC) -> None:
    """Cinco grupos com R$ 210 mil cada somam R$ 1,05 milhão, acima do teto global."""
    grupos = [produto(f"CDB {i}", f"Grupo {i}") for i in range(5)]
    mensagens = violacoes_fgc([(p, 210_000) for p in grupos], fgc)
    assert mensagens == [
        "Total de R$ 1.050.000,00 em produtos cobertos, acima do teto global de R$ 1.000.000,00 por CPF."
    ]


def test_formatar_reais() -> None:
    """Valores em reais usam ponto no milhar e vírgula nos centavos."""
    assert formatar_reais(1234567.891) == "R$ 1.234.567,89"
    assert formatar_reais(0) == "R$ 0,00"
