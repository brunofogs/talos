"""Testes de `mercado.py`: curva de juros, inflação implícita, cenários e retorno dos produtos."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from talos.mercado import (
    Cenarios,
    CurvaDI,
    carregar_curva_di,
    cdi_medio_projetado,
    choques_t_correlacionados,
    gerar_cenarios,
    inflacao_implicita,
    meses_para_dias_uteis,
    retorno_antes_do_ir,
    retorno_percentual_cdi,
    retornos_antes_do_ir,
    taxa_na_curva,
)
from talos.modelos import ErroDeDados, Premissas, Produto, carregar_premissas

TOLERANCIA_ESTATISTICA = 0.04  # 4% de erro relativo, para amostras de 20 mil cenários


@pytest.fixture
def premissas(exemplos: Path) -> Premissas:
    """Premissas de exemplo (juro real de 8%, 252 dias úteis)."""
    return carregar_premissas(exemplos / "premissas.json")


@pytest.fixture
def curva(exemplos: Path, premissas: Premissas) -> CurvaDI:
    """Curva de exemplo: 14% a.a. em 252 dias úteis, 13,2% em 504 etc."""
    return carregar_curva_di(exemplos / "curva_di.csv", premissas)


def escrever(caminho: Path, conteudo: str) -> Path:
    """Grava um arquivo de teste em UTF-8."""
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


# --- Curva de juros ---------------------------------------------------------------

def test_carregar_curva_de_exemplo(curva: CurvaDI) -> None:
    """A curva de exemplo tem 10 vértices, do 21º ao 5040º dia útil."""
    assert len(curva.prazos_du) == 10
    assert (curva.prazos_du[0], curva.taxas_aa[0]) == (21, 0.149)
    assert (curva.prazos_du[-1], curva.taxas_aa[-1]) == (5040, 0.134)


def test_curva_fora_de_ordem_gera_erro(tmp_path: Path, premissas: Premissas) -> None:
    """Vértices fora de ordem crescente são recusados."""
    arquivo = escrever(tmp_path / "curva.csv", "prazo_dias_uteis,taxa_aa\n252,0.14\n126,0.145\n")
    with pytest.raises(ErroDeDados, match="ordem crescente"):
        carregar_curva_di(arquivo, premissas)


def test_curva_sem_coluna_gera_erro(tmp_path: Path, premissas: Premissas) -> None:
    """CSV da curva sem a coluna de taxa é recusado."""
    arquivo = escrever(tmp_path / "curva.csv", "prazo_dias_uteis\n252\n")
    with pytest.raises(ErroDeDados, match="colunas ausentes: taxa_aa"):
        carregar_curva_di(arquivo, premissas)


@pytest.mark.parametrize(("meses", "dias_uteis"), [(1, 21), (12, 252), (24, 504), (240, 5040)])
def test_meses_para_dias_uteis(meses: int, dias_uteis: int) -> None:
    """Cada mês vale 21 dias úteis (252 / 12)."""
    assert meses_para_dias_uteis(meses, 252) == dias_uteis


@pytest.mark.parametrize(("dias_uteis", "taxa"), [(252, 0.14), (504, 0.132), (10, 0.149), (6000, 0.134)])
def test_taxa_nos_vertices_e_fora_da_curva(curva: CurvaDI, dias_uteis: int, taxa: float) -> None:
    """Num vértice, a taxa é a do vértice; antes do primeiro e depois do último, repete a da ponta."""
    assert taxa_na_curva(curva, dias_uteis) == pytest.approx(taxa)


def test_interpolacao_flat_forward_a_mao(curva: CurvaDI) -> None:
    """No meio do caminho entre 252 dias (14%) e 504 dias (13,2%):

    log do fator em 252 = ln(1,14) x 1 = 0,131028
    log do fator em 504 = ln(1,132) x 2 = 0,247978
    em 378 dias: média dos dois = 0,189503, em 1,5 ano
    taxa = exp(0,189503 / 1,5) - 1 = 13,467% a.a.
    """
    esperado = math.exp((math.log(1.14) + 2 * math.log(1.132)) / 2 / 1.5) - 1
    assert taxa_na_curva(curva, 378) == pytest.approx(esperado)
    assert taxa_na_curva(curva, 378) == pytest.approx(0.13467, abs=1e-5)


def test_flat_forward_tem_taxa_a_termo_constante(curva: CurvaDI) -> None:
    """Entre dois vértices, cada dia útil rende a mesma taxa a termo."""
    def fator(dias: int) -> float:
        return (1 + taxa_na_curva(curva, dias)) ** (dias / 252)
    primeiro_trecho = (fator(378) / fator(252)) ** (1 / 126)
    segundo_trecho = (fator(504) / fator(378)) ** (1 / 126)
    assert primeiro_trecho == pytest.approx(segundo_trecho)


def test_prazo_zero_gera_erro(curva: CurvaDI) -> None:
    """Pedir a taxa de um prazo de 0 dias úteis gera erro."""
    with pytest.raises(ErroDeDados):
        taxa_na_curva(curva, 0)


def test_cdi_medio_e_inflacao_implicita(curva: CurvaDI, premissas: Premissas) -> None:
    """Em 12 meses o CDI médio é 14%; com juro real de 8%, a inflação implícita é
    1,14 / 1,08 - 1 = 5,556% a.a.
    """
    assert cdi_medio_projetado(curva, 12) == pytest.approx(0.14)
    assert cdi_medio_projetado(curva, 24) == pytest.approx(0.132)
    assert inflacao_implicita(curva, 12, premissas) == pytest.approx(1.14 / 1.08 - 1)


# --- Choques aleatórios -------------------------------------------------------------

def test_choques_tem_variancia_um_e_caudas_gordas() -> None:
    """Com t de Student (5 graus de liberdade), choques além de 3 desvios são bem mais comuns
    que na distribuição normal (0,27%): na teoria, cerca de 1,2%.
    """
    gerador = np.random.default_rng(1)
    choques = choques_t_correlacionados(200_000, [[1.0]], 5, gerador)[:, 0]
    assert choques.var() == pytest.approx(1.0, rel=TOLERANCIA_ESTATISTICA)
    assert np.mean(np.abs(choques) > 3) > 0.009


def test_choques_seguem_as_correlacoes() -> None:
    """A correlação entre as colunas dos choques é a pedida."""
    gerador = np.random.default_rng(2)
    choques = choques_t_correlacionados(50_000, [[1.0, -0.5], [-0.5, 1.0]], 5, gerador)
    assert np.corrcoef(choques.T)[0, 1] == pytest.approx(-0.5, abs=0.02)


def test_correlacao_impossivel_gera_erro() -> None:
    """A e B muito parecidos, A e C muito parecidos, mas B e C opostos: não existe."""
    impossivel = [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]
    with pytest.raises(ErroDeDados, match="positiva definida"):
        choques_t_correlacionados(10, impossivel, 5, np.random.default_rng(0))


# --- Cenários ------------------------------------------------------------------------

def test_cenarios_usam_quantidade_e_semente_das_premissas(premissas: Premissas, curva: CurvaDI) -> None:
    """Sem parâmetros extras, gera 5.000 cenários com a semente 42."""
    cenarios = gerar_cenarios(premissas, curva, 24)
    assert cenarios.quantidade == 5000
    assert cenarios.semente == 42
    assert set(cenarios.retornos_aa) == {"pos_fixado", "prefixado", "inflacao", "acoes"}


def test_mesma_semente_mesmos_cenarios(premissas: Premissas, curva: CurvaDI) -> None:
    """A mesma semente sempre gera os mesmos números; outra semente gera outros."""
    a = gerar_cenarios(premissas, curva, 36, semente=7)
    b = gerar_cenarios(premissas, curva, 36, semente=7)
    c = gerar_cenarios(premissas, curva, 36, semente=8)
    assert np.array_equal(a.retornos_aa["acoes"], b.retornos_aa["acoes"])
    assert not np.array_equal(a.retornos_aa["acoes"], c.retornos_aa["acoes"])


def test_medias_centradas_na_curva(premissas: Premissas, curva: CurvaDI) -> None:
    """Em 24 meses: renda fixa com média no CDI projetado (13,2%), ações no CDI + 5% (18,2%).

    "Média" quer dizer patrimônio esperado no fim do prazo: R$ 1 vira, em
    média, 1,132^2 na renda fixa e 1,182^2 em ações.
    """
    cenarios = gerar_cenarios(premissas, curva, 24, quantidade=20_000)
    assert cenarios.medias_aa["pos_fixado"] == pytest.approx(0.132)
    assert cenarios.medias_aa["acoes"] == pytest.approx(0.182)
    patrimonio_rf = (1 + cenarios.retornos_aa["pos_fixado"]) ** 2
    patrimonio_acoes = (1 + cenarios.retornos_aa["acoes"]) ** 2
    assert patrimonio_rf.mean() == pytest.approx(1.132 ** 2, rel=0.001)
    assert patrimonio_acoes.mean() == pytest.approx(1.182 ** 2, rel=0.01)


def test_volatilidade_puxa_o_cenario_tipico_para_baixo(premissas: Premissas, curva: CurvaDI) -> None:
    """Em ações, o cenário do meio (mediana) rende menos que a média, porque perdas pesam mais
    que ganhos do mesmo tamanho: log(1 + mediana) = log(1,182) - 0,25² / 2.
    """
    cenarios = gerar_cenarios(premissas, curva, 24, quantidade=20_000)
    mediana = np.median(cenarios.retornos_aa["acoes"])
    assert mediana == pytest.approx(1.182 * math.exp(-0.25 ** 2 / 2) - 1, abs=0.005)


def test_classe_sem_volatilidade_nao_varia(premissas: Premissas, curva: CurvaDI) -> None:
    """A classe prefixada tem volatilidade zero: todos os cenários são iguais à média."""
    cenarios = gerar_cenarios(premissas, curva, 24)
    assert cenarios.retornos_aa["prefixado"] == pytest.approx(np.full(5000, 0.132))


def test_incerteza_de_acoes_cai_com_a_raiz_do_prazo(premissas: Premissas, curva: CurvaDI) -> None:
    """Ações (raiz_do_prazo): em 10 anos o desvio anual é 25% / raiz(10) = 7,9%.
    CDI (constante): 1,5% em 1 ano e também em 10 anos.
    """
    em_1_ano = gerar_cenarios(premissas, curva, 12, quantidade=20_000)
    em_10_anos = gerar_cenarios(premissas, curva, 120, quantidade=20_000)

    desvio_acoes = np.log1p(em_10_anos.retornos_aa["acoes"]).std()
    assert desvio_acoes == pytest.approx(0.25 / math.sqrt(10), rel=TOLERANCIA_ESTATISTICA)

    for cenarios in (em_1_ano, em_10_anos):
        desvio_cdi = np.log1p(cenarios.retornos_aa["pos_fixado"]).std()
        assert desvio_cdi == pytest.approx(0.015, rel=TOLERANCIA_ESTATISTICA)


def test_correlacao_entre_classes(premissas: Premissas, curva: CurvaDI) -> None:
    """Nas premissas, CDI e ações têm correlação de -0,2."""
    cenarios = gerar_cenarios(premissas, curva, 60, quantidade=20_000)
    correlacao = np.corrcoef(cenarios.retornos_aa["pos_fixado"], cenarios.retornos_aa["acoes"])[0, 1]
    assert correlacao == pytest.approx(-0.2, abs=0.04)


def _com_volatilidade_de_acoes(premissas: Premissas, volatilidade: float) -> Premissas:
    """Copia as premissas trocando só a volatilidade de ações."""
    classes = dict(premissas.cenarios.classes)
    classes["acoes"] = replace(classes["acoes"], volatilidade_aa=volatilidade)
    return replace(premissas, cenarios=replace(premissas.cenarios, classes=classes))


def test_nenhum_cenario_perde_mais_que_tudo(premissas: Premissas, curva: CurvaDI) -> None:
    """Com volatilidade alta (60% a.a.) em 1 ano, as perdas chegam perto de 100%, mas não passam."""
    cenarios = gerar_cenarios(_com_volatilidade_de_acoes(premissas, 0.60), curva, 12, quantidade=20_000)
    assert cenarios.retornos_aa["acoes"].min() > -1


def test_volatilidade_absurda_gera_erro(premissas: Premissas, curva: CurvaDI) -> None:
    """Volatilidade de 150% a.a. em 1 mês faz a conta perder o sentido: o programa avisa."""
    with pytest.raises(ErroDeDados, match="volatilidade alta demais"):
        gerar_cenarios(_com_volatilidade_de_acoes(premissas, 1.5), curva, 1, quantidade=20_000)


def test_ipca_dos_cenarios_em_media_e_a_inflacao_implicita(premissas: Premissas, curva: CurvaDI) -> None:
    """O IPCA dos cenários fica, em média, na inflação implícita da curva."""
    cenarios = gerar_cenarios(premissas, curva, 48, quantidade=20_000)
    assert cenarios.ipca_aa().mean() == pytest.approx(inflacao_implicita(curva, 48, premissas), abs=0.002)


def test_prazo_zero_nos_cenarios_gera_erro(premissas: Premissas, curva: CurvaDI) -> None:
    """Cenários precisam de prazo de pelo menos 1 mês."""
    with pytest.raises(ErroDeDados):
        gerar_cenarios(premissas, curva, 0)


# --- Retorno dos produtos --------------------------------------------------------------

def produto(indexador: str, taxa: float, classe: str, taxa_adm: float = 0.0) -> Produto:
    """Cria um produto simples com o indexador e a taxa pedidos."""
    return Produto(
        nome=f"{indexador} {taxa}", emissor="X", conglomerado="X", indexador=indexador,
        taxa=taxa, taxa_adm_aa=taxa_adm, vencimento_meses=None, carencia_meses=0,
        liquidez_diaria=True, tributacao="regressivo", classe=classe, risco=1,
        coberto_fgc=False, aplicacao_minima=0.0, setor="financeiro",
    )


@pytest.fixture
def dois_cenarios() -> Cenarios:
    """Dois cenários montados à mão.

    Cenário 1: CDI 10%, prefixado na média (11%), IPCA 4%, ações -20%.
    Cenário 2: CDI 12%, prefixado 12%, IPCA 6%, ações +30%.
    """
    return Cenarios(
        prazo_meses=24, semente=0, cdi_projetado_aa=0.11, juro_real_aa=0.08,
        dias_uteis_por_ano=252,
        medias_aa={"pos_fixado": 0.11, "prefixado": 0.11, "inflacao": 0.11, "acoes": 0.16},
        retornos_aa={
            "pos_fixado": np.array([0.10, 0.12]),
            "prefixado": np.array([0.11, 0.12]),
            "inflacao": np.array([1.04 * 1.08 - 1, 1.06 * 1.08 - 1]),
            "acoes": np.array([-0.20, 0.30]),
        },
    )


def test_percentual_do_cdi_a_mao() -> None:
    """110% do CDI com CDI de 10%: taxa diária (1,1)^(1/252) - 1 = 0,03783%;
    110% disso = 0,04161% ao dia, que composto em 252 dias dá cerca de 11,05% a.a.
    (um pouco mais que 1,1 x 10% = 11%, por causa dos juros compostos).
    """
    diaria = 1.1 ** (1 / 252) - 1
    esperado = (1 + 1.1 * diaria) ** 252 - 1
    assert retorno_percentual_cdi(0.10, 1.10, 252) == pytest.approx(esperado)
    assert 0.110 < esperado < 0.111
    assert retorno_percentual_cdi(0.10, 1.00, 252) == pytest.approx(0.10)


def test_retorno_de_cada_indexador(dois_cenarios: Cenarios) -> None:
    """Cada indexador lê a classe certa do cenário."""
    selic = retorno_antes_do_ir(produto("selic", 0.0005, "pos_fixado"), dois_cenarios)
    assert selic == pytest.approx([1.10 * 1.0005 - 1, 1.12 * 1.0005 - 1])

    prefixado = retorno_antes_do_ir(produto("prefixado", 0.13, "prefixado"), dois_cenarios)
    assert prefixado == pytest.approx([0.13, 1.13 * 1.12 / 1.11 - 1])

    ipca = retorno_antes_do_ir(produto("ipca", 0.075, "inflacao"), dois_cenarios)
    assert ipca == pytest.approx([1.04 * 1.075 - 1, 1.06 * 1.075 - 1])


def test_taxa_de_administracao_e_descontada(dois_cenarios: Cenarios) -> None:
    """ETF com taxa de 0,1% a.a.: -20% vira 0,8 x 0,999 - 1 = -20,08%."""
    etf = retorno_antes_do_ir(produto("variavel", 0.0, "acoes", taxa_adm=0.001), dois_cenarios)
    assert etf == pytest.approx([0.8 * 0.999 - 1, 1.3 * 0.999 - 1])


def test_tabela_de_retornos_por_produto(dois_cenarios: Cenarios) -> None:
    """A tabela tem uma linha por cenário e uma coluna por produto."""
    produtos = [produto("cdi", 1.0, "pos_fixado"), produto("variavel", 0.0, "acoes")]
    tabela = retornos_antes_do_ir(produtos, dois_cenarios)
    assert tabela.shape == (2, 2)
    assert tabela[:, 0] == pytest.approx([0.10, 0.12])
    assert tabela[:, 1] == pytest.approx([-0.20, 0.30])


def test_prateleira_de_exemplo_inteira(premissas: Premissas, curva: CurvaDI, exemplos: Path) -> None:
    """Todos os 15 produtos de exemplo geram retornos válidos (acima de -100%)."""
    from talos.modelos import carregar_prateleira

    produtos = carregar_prateleira(exemplos / "prateleira.csv")
    tabela = retornos_antes_do_ir(produtos, gerar_cenarios(premissas, curva, 60))
    assert tabela.shape == (5000, 15)
    assert np.isfinite(tabela).all()
    assert (tabela > -1).all()
