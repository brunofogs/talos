"""Testes das ações individuais: leitura da lista da corretora, retorno nos cenários,
limites no otimizador, estratégia de crescimento e explicações.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from talos.explicacao import explicar
from talos.mercado import (
    CurvaDI,
    carregar_curva_di,
    centrar_na_media,
    choques_mensais_somados,
    gerar_cenarios,
    retorno_antes_do_ir,
)
from talos.modelos import (
    ErroDeDados,
    Premissas,
    Produto,
    carregar_acoes,
    carregar_cliente,
    carregar_prateleira,
    carregar_premissas,
)
from talos.otimizador import (
    CRESCIMENTO,
    ProblemaCarteira,
    Recomendacao,
    limite_do_produto,
    maximizar_crescimento,
    otimizar_cliente,
    otimizar_meta,
)

EXEMPLOS = Path(__file__).resolve().parent.parent / "exemplos"
CENARIOS = 1000


@pytest.fixture(scope="module")
def premissas() -> Premissas:
    """Premissas de exemplo (10% por ação, 2% de peso mínimo, 50% por conglomerado)."""
    return carregar_premissas(EXEMPLOS / "premissas.json")


@pytest.fixture(scope="module")
def curva(premissas: Premissas) -> CurvaDI:
    """Curva de juros de exemplo."""
    return carregar_curva_di(EXEMPLOS / "curva_di.csv", premissas)


@pytest.fixture(scope="module")
def acoes() -> list[Produto]:
    """As 10 ações fictícias da lista de exemplo."""
    return carregar_acoes(EXEMPLOS / "acoes.csv")


@pytest.fixture(scope="module")
def produtos(acoes: list[Produto]) -> list[Produto]:
    """Prateleira de exemplo mais as ações."""
    return carregar_prateleira(EXEMPLOS / "prateleira.csv") + acoes


@pytest.fixture(scope="module")
def arrojado_crescimento(produtos, premissas, curva) -> Recomendacao:
    """Cliente de exemplo, perfil arrojado, estratégia de crescimento."""
    cliente = replace(carregar_cliente(EXEMPLOS / "cliente.json"), perfil="arrojado")
    return otimizar_cliente(cliente, produtos, premissas, curva, quantidade=CENARIOS, estrategia=CRESCIMENTO)


def escrever(caminho: Path, conteudo: str) -> Path:
    """Grava um arquivo de teste em UTF-8."""
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


# --- Leitura da lista de ações ------------------------------------------------------------------

def test_carregar_acoes_de_exemplo(acoes: list[Produto]) -> None:
    """Cada linha vira uma ação de renda variável, sem FGC, com liquidez diária e IR de ações."""
    assert len(acoes) == 10
    aura = acoes[0]
    assert (aura.nome, aura.ticker, aura.conglomerado, aura.setor) == (
        "AURA-F (Banco Aurora)", "AURA-F", "Conglomerado Aurora", "financeiro",
    )
    assert (aura.beta, aura.volatilidade_propria_aa, aura.alfa_aa, aura.aplicacao_minima) == (1.10, 0.22, 0.010, 32.50)
    assert aura.acao_individual and aura.renda_variavel
    assert (aura.tributacao, aura.coberto_fgc, aura.liquidez_diaria) == ("acoes", False, True)


def test_tickers_de_exemplo_sao_ficticios(acoes: list[Produto]) -> None:
    """Todos os tickers de exemplo terminam em -F, para não serem confundidos com ações reais."""
    assert all(acao.ticker.endswith("-F") for acao in acoes)
    primeira_linha = (EXEMPLOS / "acoes.csv").read_text(encoding="utf-8").splitlines()[0]
    assert primeira_linha.startswith("#") and "fictícios" in primeira_linha


def test_ticker_repetido(tmp_path: Path) -> None:
    """Dois tickers iguais na lista geram erro."""
    cabecalho = "ticker,empresa,conglomerado,setor,risco,beta,volatilidade_propria_aa,alfa_aa,aplicacao_minima"
    linha = "XYZ-F,Empresa X,,energia,4,1,0.2,0,10"
    with pytest.raises(ErroDeDados, match="tickers repetidos"):
        carregar_acoes(escrever(tmp_path / "acoes.csv", f"{cabecalho}\n{linha}\n{linha}\n"))


def test_conglomerado_vazio_usa_a_empresa(tmp_path: Path) -> None:
    """Sem conglomerado informado, a própria empresa é o conglomerado."""
    cabecalho = "ticker,empresa,conglomerado,setor,risco,beta,volatilidade_propria_aa,alfa_aa,aplicacao_minima"
    acao = carregar_acoes(escrever(tmp_path / "acoes.csv", f"{cabecalho}\nXYZ-F,Empresa X,,energia,4,1,0.2,0,10\n"))[0]
    assert acao.conglomerado == "Empresa X"


def test_setor_desconhecido_na_lista(tmp_path: Path) -> None:
    """Setor fora da lista de setores é recusado."""
    cabecalho = "ticker,empresa,conglomerado,setor,risco,beta,volatilidade_propria_aa,alfa_aa,aplicacao_minima"
    with pytest.raises(ErroDeDados, match="setor"):
        carregar_acoes(escrever(tmp_path / "acoes.csv", f"{cabecalho}\nXYZ-F,Empresa X,,cripto,4,1,0.2,0,10\n"))


def test_acao_precisa_ser_renda_variavel(acoes: list[Produto]) -> None:
    """Um produto com ticker precisa ter indexador variavel e classe acoes."""
    with pytest.raises(ErroDeDados, match="ação individual"):
        replace(acoes[0], indexador="cdi", classe="pos_fixado")


# --- Choques e centragem ------------------------------------------------------------------------

def test_centrar_na_media_a_mao() -> None:
    """R$ 1 vira 1,0 ou 1,21 em 1 ano (média 1,105). Centrando na média de 10%,
    os dois cenários são multiplicados por 1,10 / 1,105 e a média vira exatamente 1,10.
    """
    centrado = centrar_na_media(np.log([1.0, 1.21]), 0.10, 1.0)
    assert np.exp(centrado) == pytest.approx(np.array([1.0, 1.21]) * 1.10 / 1.105)
    assert np.mean(np.exp(centrado)) == pytest.approx(1.10)


def test_choques_mensais_somados() -> None:
    """A soma de choques mensais tem variância 1; em 1 mês tem caudas gordas, em 10 anos fica perto da normal."""
    gerador = np.random.default_rng(3)
    um_mes = choques_mensais_somados(100_000, 1, [[1.0]], 5, gerador)[:, 0]
    dez_anos = choques_mensais_somados(20_000, 120, [[1.0]], 5, gerador)[:, 0]
    for choques in (um_mes, dez_anos):
        assert choques.var() == pytest.approx(1.0, rel=0.05)

    def curtose_excedente(x: np.ndarray) -> float:
        return float(np.mean((x - x.mean()) ** 4) / x.var() ** 2 - 3)

    assert curtose_excedente(um_mes) > 2
    assert abs(curtose_excedente(dez_anos)) < 0.3


# --- Retorno das ações nos cenários -------------------------------------------------------------

def test_acao_sem_risco_proprio_rende_igual_a_bolsa(acoes, premissas, curva) -> None:
    """Beta 1, alfa 0 e volatilidade própria 0: a ação rende exatamente o que a bolsa rende."""
    neutra = replace(acoes[0], beta=1.0, alfa_aa=0.0, volatilidade_propria_aa=0.0)
    cenarios = gerar_cenarios(premissas, curva, 60, quantidade=CENARIOS)
    assert retorno_antes_do_ir(neutra, cenarios) == pytest.approx(cenarios.retornos_aa["acoes"])


def test_media_da_acao_segue_beta_e_alfa(acoes, premissas, curva) -> None:
    """Retorno esperado = CDI projetado + beta x (bolsa - CDI projetado) + alfa.
    AURA-F em 5 anos: 13,1% + 1,1 x 5% + 1% = 19,6% a.a. (patrimônio médio).
    """
    cenarios = gerar_cenarios(premissas, curva, 60, quantidade=CENARIOS)
    aura = acoes[0]
    esperado = cenarios.cdi_projetado_aa + 1.1 * 0.05 + 0.01
    patrimonio = (1 + retorno_antes_do_ir(aura, cenarios)) ** 5
    assert np.mean(patrimonio) == pytest.approx((1 + esperado) ** 5, rel=1e-9)
    assert esperado == pytest.approx(0.196)


def test_choques_proprios_reproduziveis_e_independentes(acoes, premissas, curva) -> None:
    """A mesma ação tem sempre os mesmos cenários; duas empresas têm choques próprios diferentes."""
    a = gerar_cenarios(premissas, curva, 60, quantidade=CENARIOS)
    b = gerar_cenarios(premissas, curva, 60, quantidade=CENARIOS)
    assert np.array_equal(retorno_antes_do_ir(acoes[0], a), retorno_antes_do_ir(acoes[0], b))
    bolsa = np.log1p(a.retornos_aa["acoes"])
    livre = np.log1p(a.retornos_aa["pos_fixado"])
    proprios = [np.log1p(retorno_antes_do_ir(acao, a)) - livre - acao.beta * (bolsa - livre) for acao in acoes[:2]]
    assert abs(np.corrcoef(*proprios)[0, 1]) < 0.1


# --- Otimizador ---------------------------------------------------------------------------------

def test_limite_de_cada_produto(acoes, produtos, premissas) -> None:
    """Tesouro Selic sem limite, produtos em 40% e cada ação em 10%."""
    por_nome = {p.nome: p for p in produtos}
    assert limite_do_produto(por_nome["Tesouro Selic 2031"], premissas) == 1.0
    assert limite_do_produto(por_nome["CDB Boreal 3 anos"], premissas) == 0.40
    assert limite_do_produto(acoes[0], premissas) == 0.10


def test_crescimento_a_mao() -> None:
    """Dois produtos, quatro cenários. Seguro: 1,10 sempre. Arriscado: 0,9; 1,4; 1,5; 1,6.
    Com w no arriscado, a metade pior é (1,1 - 0,2w) e (1,1 + 0,3w): média 1,1 + 0,05w, que
    cresce com w. O teto de risco (pior cenário, nível 75%) é perder no máximo 0%:
    1 - (1,1 - 0,2w) <= 0, ou seja, w <= 0,5. Ótimo: w = 0,5.
    """
    fatores = np.array([[1.10, 0.90], [1.10, 1.40], [1.10, 1.50], [1.10, 1.60]])
    problema = ProblemaCarteira(fatores, 0.75, np.array([1.0, 1.0]), ())
    assert maximizar_crescimento(problema, cvar_maximo=0.0, nivel_tipico=0.5) == pytest.approx([0.5, 0.5], abs=1e-6)


def test_crescimento_usa_acoes_com_limites(arrojado_crescimento: Recomendacao, premissas: Premissas) -> None:
    """Na aposentadoria (20 anos), a estratégia de crescimento escolhe ações, cada uma com até 10%,
    renda variável até 50% (arrojado) e risco dentro do teto do perfil.
    """
    aposentadoria = next(r for r in arrojado_crescimento.resultados if r.meta.nome == "Aposentadoria")
    acoes = [a for a in aposentadoria.alocacoes if a.produto.acao_individual]
    assert acoes
    assert all(a.peso <= 0.10 + 1e-6 for a in acoes)
    assert sum(a.peso for a in aposentadoria.alocacoes if a.produto.renda_variavel) <= 0.50 + 1e-6
    assert aposentadoria.cvar <= arrojado_crescimento.perfil.cvar_maximo + 1e-6


def test_peso_minimo_e_concentracao(arrojado_crescimento: Recomendacao) -> None:
    """Nenhum produto com menos de 2% da meta e nenhum grupo (fora o Tesouro) com mais de 50%."""
    for resultado in arrojado_crescimento.resultados:
        assert all(a.peso >= 0.02 - 1e-6 for a in resultado.alocacoes)
        grupos: dict[str, float] = {}
        for alocacao in resultado.alocacoes:
            grupos[alocacao.produto.conglomerado] = grupos.get(alocacao.produto.conglomerado, 0) + alocacao.peso
        assert all(peso <= 0.50 + 1e-6 for grupo, peso in grupos.items() if grupo != "Tesouro Nacional")


def test_quem_trabalha_na_empresa_nao_recebe_a_acao(produtos, premissas, curva) -> None:
    """Quem trabalha na Energia Sol (Grupo Sol) não recebe a ação SOLR-F nem a debênture do grupo."""
    cliente = replace(carregar_cliente(EXEMPLOS / "cliente.json"), perfil="arrojado", empregador="Grupo Sol")
    recomendacao = otimizar_cliente(cliente, produtos, premissas, curva, quantidade=CENARIOS, estrategia=CRESCIMENTO)
    for resultado in recomendacao.resultados:
        assert all(a.produto.conglomerado != "Grupo Sol" for a in resultado.alocacoes)


def test_estrategia_desconhecida(produtos, premissas, curva) -> None:
    """Estratégia que não existe gera erro com as opções válidas."""
    meta = carregar_cliente(EXEMPLOS / "cliente.json").metas[2]
    with pytest.raises(ErroDeDados, match="menor_risco, crescimento"):
        otimizar_meta(meta, premissas.perfil("arrojado"), produtos, premissas, curva, estrategia="agressiva")


# --- Explicações --------------------------------------------------------------------------------

def test_explicacoes_de_acoes_e_estrategia(arrojado_crescimento: Recomendacao, premissas: Premissas) -> None:
    """A estratégia é explicada no começo e cada ação diz de onde veio e o limite por ação."""
    explicacao = explicar(arrojado_crescimento, premissas)
    assert any(f.startswith("Estratégia de crescimento") for f in explicacao.antes_de_tudo)
    aposentadoria = next(m for m in explicacao.metas if m.meta == "Aposentadoria")
    frases_de_acoes = [f for nome, frases in aposentadoria.produtos.items() if "-F (" in nome for f in frases]
    assert any("da lista aprovada pela área de análise da corretora" in f for f in frases_de_acoes)
    resultado = next(r for r in arrojado_crescimento.resultados if r.meta.nome == "Aposentadoria")
    no_limite = [a.produto.nome for a in resultado.alocacoes if a.produto.acao_individual and a.peso >= 0.10 - 1e-6]
    for nome in no_limite:
        assert "Ficou no limite de 10% por ação, para não concentrar demais." in aposentadoria.produtos[nome]
