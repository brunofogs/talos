"""Testes de `explicacao.py`: cada regra gera a frase esperada, com os números certos."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from talos import AVISO_SIMULACAO
from talos.explicacao import (
    explicar,
    formatar_percentual,
    formatar_percentual_cdi,
    formatar_prazo,
    formatar_probabilidade,
)
from talos.mercado import carregar_curva_di
from talos.modelos import (
    Cliente,
    Meta,
    Premissas,
    Produto,
    carregar_cliente,
    carregar_prateleira,
    carregar_premissas,
)
from talos.otimizador import (
    ATINGIVEL,
    DIFICIL,
    SEM_SOLUCAO,
    Alocacao,
    ProdutoExcluido,
    Recomendacao,
    ResultadoMeta,
    otimizar_cliente,
)


@pytest.fixture
def premissas(exemplos: Path) -> Premissas:
    """Premissas de exemplo."""
    return carregar_premissas(exemplos / "premissas.json")


def produto(nome: str = "CDB X", **mudancas: object) -> Produto:
    """CDB coberto pelo FGC, 2 anos, que pode ser alterado campo a campo."""
    campos = dict(
        nome=nome, emissor="Banco X", conglomerado="Grupo X", indexador="cdi", taxa=1.10,
        taxa_adm_aa=0.0, vencimento_meses=24, carencia_meses=0, liquidez_diaria=False,
        tributacao="regressivo", classe="pos_fixado", risco=2, coberto_fgc=True, aplicacao_minima=0.0, setor="financeiro",
    )
    campos.update(mudancas)
    return Produto(**campos)


def resultado(
    alocacoes: tuple[Alocacao, ...] = (),
    situacao: str = ATINGIVEL,
    meta: Meta = Meta("Carro", 10_000, 12_000, 24, False),
    cvar: float = -0.10,
    probabilidade: float = 0.9,
    uso_fgc: dict | None = None,
    excluidos: tuple[ProdutoExcluido, ...] = (),
) -> ResultadoMeta:
    """Resultado de meta montado à mão (necessário 9,54% a.a., esperado 11% a.a.)."""
    return ResultadoMeta(
        meta=meta, situacao=situacao, alocacoes=alocacoes, excluidos=excluidos,
        retorno_necessario_aa=(1.2 ** 0.5 - 1), retorno_esperado_aa=0.11, cvar=cvar,
        probabilidade_sucesso=probabilidade, cdi_projetado_aa=0.13, uso_fgc=uso_fgc or {},
        semente=42, reamostragem=False,
    )


def alocacao(prod: Produto, peso: float = 0.3, exposicao: float = 3_600) -> Alocacao:
    """Alocação de `peso` de uma meta de R$ 10 mil."""
    return Alocacao(prod, peso, peso * 10_000, 0.11, exposicao if prod.coberto_fgc else 0.0)


def explicar_um(res: ResultadoMeta, premissas: Premissas, dividas: bool = False):
    """Explica uma recomendação com um único resultado."""
    cliente = Cliente("c1", "moderado", 3_000, dividas, (res.meta,))
    recomendacao = Recomendacao(cliente, premissas.perfil("moderado"), (res,), res.uso_fgc, 42, False)
    return explicar(recomendacao, premissas)


def frases_do_produto(res: ResultadoMeta, premissas: Premissas, nome: str) -> tuple[str, ...]:
    """Frases de um produto específico."""
    return explicar_um(res, premissas).metas[0].produtos[nome]


# --- Formatação -------------------------------------------------------------------------------

@pytest.mark.parametrize(("fracao", "texto"), [(0.1159, "11,59%"), (0.0, "0,00%"), (-0.05, "-5,00%")])
def test_formatar_percentual(fracao: float, texto: str) -> None:
    """Porcentagem com vírgula decimal."""
    assert formatar_percentual(fracao) == texto


@pytest.mark.parametrize(("fracao", "texto"), [(0.94, "94%"), (1.10588, "110,6%"), (1.0, "100%")])
def test_formatar_percentual_cdi(fracao: float, texto: str) -> None:
    """Percentual do CDI com no máximo uma casa, sem ",0" sobrando."""
    assert formatar_percentual_cdi(fracao) == texto


@pytest.mark.parametrize(("meses", "texto"), [(1, "1 mês"), (12, "1 ano"), (18, "18 meses"), (48, "4 anos")])
def test_formatar_prazo(meses: int, texto: str) -> None:
    """Prazos em anos quando dá conta exata."""
    assert formatar_prazo(meses) == texto


@pytest.mark.parametrize(("chance", "texto"), [(0.0, "menos de 0,1%"), (1.0, "mais de 99,9%"), (0.987, "98,7%")])
def test_formatar_probabilidade(chance: float, texto: str) -> None:
    """A chance nunca aparece como 0% ou 100% exatos."""
    assert formatar_probabilidade(chance) == texto


# --- Resumo da meta ----------------------------------------------------------------------------

def test_dividas_caras_vem_antes_de_tudo(premissas: Premissas) -> None:
    """Com dívidas caras, a primeira frase manda quitá-las; sem dívidas, não há essa frase."""
    com_dividas = explicar_um(resultado(), premissas, dividas=True)
    assert com_dividas.antes_de_tudo[0].startswith("Antes de investir, quite as dívidas caras")
    assert explicar_um(resultado(), premissas).antes_de_tudo == ()


def test_resumo_de_meta_atingivel(premissas: Premissas) -> None:
    """Meta atingível traz retorno esperado, necessário, caminho e chance."""
    resumo = explicar_um(resultado(), premissas).metas[0].resumo
    assert resumo[0] == (
        "Meta atingível: a carteira rende em média 11,00% ao ano, já descontado o IR, acima dos "
        "9,54% ao ano necessários para ir de R$ 10.000,00 a R$ 12.000,00 em 2 anos."
    )
    assert resumo[1] == "Chance de atingir a meta: 90,0% dos cenários simulados."


def test_resumo_de_meta_dificil(premissas: Premissas) -> None:
    """Meta difícil explica o que falta e sugere o que fazer."""
    resumo = explicar_um(resultado(situacao=DIFICIL, probabilidade=0.02), premissas).metas[0].resumo
    assert resumo[0].startswith("Meta difícil: para ir de R$ 10.000,00 a R$ 12.000,00 em 2 anos seriam necessários 9,54%")
    assert "perfil moderado" in resumo[0]
    assert resumo[1].endswith("alongar o prazo, reduzir o valor-alvo ou fazer aportes.")


def test_meta_sem_solucao_lista_motivos(premissas: Premissas) -> None:
    """Sem solução: diz que nada serve e agrupa os produtos por motivo."""
    excluidos = (
        ProdutoExcluido(produto("A"), "renda variável exige prazo de pelo menos 36 meses"),
        ProdutoExcluido(produto("B"), "renda variável exige prazo de pelo menos 36 meses"),
        ProdutoExcluido(produto("C"), "a carência de 12 meses termina depois da meta"),
    )
    res = replace(resultado(situacao=SEM_SOLUCAO, excluidos=excluidos), cvar=None, probabilidade_sucesso=None)
    meta = explicar_um(res, premissas).metas[0]
    assert meta.resumo == ("Nenhum produto da prateleira serve para esta meta. Veja abaixo os motivos.",)
    assert meta.nao_usados == (
        "Renda variável exige prazo de pelo menos 36 meses: A, B.",
        "A carência de 12 meses termina depois da meta: C.",
    )


def test_frase_de_perda_nos_piores_cenarios(premissas: Premissas) -> None:
    """CVaR de 8%: perda média de R$ 800 numa meta de R$ 10 mil."""
    resumo = explicar_um(resultado(cvar=0.08), premissas).metas[0].resumo
    assert resumo[-1] == "Nos 5% piores cenários, a perda média seria de R$ 800,00 (8,00% do valor aplicado) no fim do prazo."


def test_frase_sem_perda_nos_piores_cenarios(premissas: Premissas) -> None:
    """CVaR de -10%: mesmo nos piores cenários, R$ 10 mil viram R$ 11 mil."""
    resumo = explicar_um(resultado(cvar=-0.10), premissas).metas[0].resumo
    assert resumo[-1] == (
        "Mesmo nos 5% piores cenários não há perda: os R$ 10.000,00 aplicados viram, em média, "
        "R$ 11.000,00 no fim do prazo."
    )


def test_reserva_de_emergencia_explica_a_regra(premissas: Premissas) -> None:
    """A reserva de emergência começa explicando a regra de liquidez."""
    reserva = resultado(meta=Meta("Reserva", 10_000, 11_000, 12, True))
    assert explicar_um(reserva, premissas).metas[0].resumo[0].startswith("É a reserva de emergência")


# --- Frases por produto ------------------------------------------------------------------------

def test_lca_equivalente_a_cdb(premissas: Premissas) -> None:
    """LCA de 94% do CDI em 2 anos (IR de 15%): equivale a CDB de 94 / 0,85 = 110,6% do CDI."""
    lca = produto("LCA", taxa=0.94, tributacao="isento")
    frases = frases_do_produto(resultado((alocacao(lca),)), premissas, "LCA")
    assert "É isento de IR: 94% do CDI isento equivale a um CDB de 110,6% do CDI no mesmo prazo." in frases


def test_isento_sem_cdi(premissas: Premissas) -> None:
    """Debênture incentivada (IPCA+) diz só que é isenta."""
    debenture = produto("Deb", indexador="ipca", taxa=0.078, tributacao="isento", classe="inflacao", coberto_fgc=False)
    assert "É isento de IR." in frases_do_produto(resultado((alocacao(debenture),)), premissas, "Deb")


def test_limite_do_fgc_atingido(premissas: Premissas) -> None:
    """Se o conglomerado chegou a R$ 250 mil (com juros), a frase avisa que o limite foi atingido."""
    cdb = produto("CDB")
    res = resultado((alocacao(cdb),), uso_fgc={"Grupo X": 250_000.0})
    frases = frases_do_produto(res, premissas, "CDB")
    assert (
        "Chegou ao limite do FGC no Grupo X (R$ 250.000,00, contando os juros até o vencimento); "
        "por isso não foi colocado mais dinheiro nele." in frases
    )


def test_fgc_abaixo_do_limite_e_sem_fgc(premissas: Premissas) -> None:
    """Abaixo do limite: diz que é coberto. Sem cobertura: diz quem assume o risco."""
    cdb = produto("CDB")
    tesouro = produto("Tesouro", emissor="Tesouro Nacional", coberto_fgc=False, indexador="selic", taxa=0.0)
    res = resultado((alocacao(cdb), alocacao(tesouro)), uso_fgc={"Grupo X": 3_600.0})
    assert "Coberto pelo FGC até R$ 250.000,00 por conglomerado (Grupo X)." in frases_do_produto(res, premissas, "CDB")
    assert "Não tem garantia do FGC: o risco de crédito é do emissor (Tesouro Nacional)." in frases_do_produto(
        res, premissas, "Tesouro"
    )


@pytest.mark.parametrize(
    ("vencimento", "liquidez", "trecho"),
    [
        (12, False, "Vence em 1 ano, antes do fim da meta"),
        (24, False, "Vence junto com a meta."),
        (60, True, "Vence depois da meta, mas tem liquidez diária"),
    ],
)
def test_vencimento_em_relacao_a_meta(premissas: Premissas, vencimento: int, liquidez: bool, trecho: str) -> None:
    """O vencimento é explicado em relação ao prazo da meta (2 anos)."""
    cdb = produto("CDB", vencimento_meses=vencimento, liquidez_diaria=liquidez)
    frases = frases_do_produto(resultado((alocacao(cdb),)), premissas, "CDB")
    assert any(frase.startswith(trecho) for frase in frases)


def test_fundo_com_come_cotas_taxa_e_carencia(premissas: Premissas) -> None:
    """Fundo: come-cotas em maio e novembro e taxa de administração; e a carência, quando existe."""
    fundo = produto(
        "Fundo", tributacao="fundo_longo_prazo", taxa_adm_aa=0.004, coberto_fgc=False,
        vencimento_meses=None, carencia_meses=6, liquidez_diaria=True,
    )
    frases = frases_do_produto(resultado((alocacao(fundo),)), premissas, "Fundo")
    assert any("come-cotas: parte do IR é cobrada antes, em maio e novembro" in f for f in frases)
    assert "Cobra taxa de administração de 0,40% ao ano (já descontada)." in frases
    assert "Tem carência de 6 meses: o dinheiro fica preso nesse período." in frases


def test_renda_variavel_e_limite_por_produto(premissas: Premissas) -> None:
    """ETF com 40% da meta: avisa que oscila e que está no limite por produto."""
    etf = produto("ETF", indexador="variavel", taxa=0.0, tributacao="acoes", classe="acoes",
                  risco=4, coberto_fgc=False, vencimento_meses=None, liquidez_diaria=True)
    frases = frases_do_produto(resultado((alocacao(etf, peso=0.4),), meta=Meta("Longa", 10_000, 30_000, 120, False)), premissas, "ETF")
    assert any(f.startswith("É renda variável") and "metas de 3 anos ou mais" in f for f in frases)
    assert "Ficou no limite de 40% por produto, para não concentrar demais." in frases


def test_tesouro_selic_nao_tem_limite_por_produto(premissas: Premissas) -> None:
    """Tesouro Selic com 100% da meta não recebe a frase de limite por produto."""
    selic = produto("Selic", indexador="selic", taxa=0.0, coberto_fgc=False, liquidez_diaria=True)
    frases = frases_do_produto(resultado((alocacao(selic, peso=1.0),)), premissas, "Selic")
    assert not any("limite de 40%" in f for f in frases)


# --- Cliente de exemplo -------------------------------------------------------------------------

def test_explicacao_do_cliente_de_exemplo(exemplos: Path, premissas: Premissas) -> None:
    """Toda meta tem resumo, todo produto escolhido tem frases e tudo termina com o aviso."""
    curva = carregar_curva_di(exemplos / "curva_di.csv", premissas)
    recomendacao = otimizar_cliente(
        carregar_cliente(exemplos / "cliente.json"), carregar_prateleira(exemplos / "prateleira.csv"),
        premissas, curva, quantidade=1000,
    )
    explicacao = explicar(recomendacao, premissas)
    assert len(explicacao.metas) == 4
    for meta, res in zip(explicacao.metas, recomendacao.resultados):
        assert meta.resumo
        assert set(meta.produtos) == {a.produto.nome for a in res.alocacoes}
        assert all(meta.produtos.values())
    assert explicacao.aviso == AVISO_SIMULACAO


# --- Emprego do cliente ---------------------------------------------------------------------------

def test_frases_de_emprego_e_setor(premissas: Premissas) -> None:
    """Empregador e setor informados geram frases no começo e no produto do mesmo setor."""
    res = resultado((alocacao(produto("CDB")),))
    cliente = Cliente("c1", "moderado", 3_000, False, (res.meta,), empregador="Grupo Y", setor_trabalho="financeiro")
    explicacao = explicar(Recomendacao(cliente, premissas.perfil("moderado"), (res,), {}, 42, False), premissas)
    assert explicacao.antes_de_tudo == (
        "Como o cliente trabalha no Grupo Y, nenhum produto desse grupo entra na carteira: se o grupo "
        "tiver problemas, o emprego e os investimentos não são atingidos juntos.",
        "Como o cliente trabalha no setor financeiro, os produtos desse setor ficam em no máximo 30% de cada meta.",
    )
    assert (
        "É do mesmo setor em que o cliente trabalha (financeiro); produtos desse setor somam no máximo 30% "
        "da meta, para não depender duas vezes do mesmo setor." in explicacao.metas[0].produtos["CDB"]
    )
