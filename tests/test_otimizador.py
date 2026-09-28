"""Testes de `otimizador.py`: contas à mão em problemas pequenos e o cliente de exemplo de ponta a ponta."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from talos.fgc import TOLERANCIA_REAIS, RestricaoLinear, violacoes_fgc
from talos.mercado import CurvaDI, carregar_curva_di
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
    ProblemaCarteira,
    Recomendacao,
    cvar,
    filtrar_produtos,
    maximizar_retorno,
    minimizar_cvar,
    otimizar_cliente,
    otimizar_meta,
    retorno_necessario_aa,
)

CENARIOS_NOS_TESTES = 1000  # menos cenários que o padrão, para os testes rodarem rápido


@pytest.fixture
def premissas(exemplos: Path) -> Premissas:
    """Premissas de exemplo."""
    return carregar_premissas(exemplos / "premissas.json")


@pytest.fixture
def curva(exemplos: Path, premissas: Premissas) -> CurvaDI:
    """Curva de juros de exemplo."""
    return carregar_curva_di(exemplos / "curva_di.csv", premissas)


@pytest.fixture
def produtos(exemplos: Path) -> list[Produto]:
    """Os 15 produtos da prateleira de exemplo."""
    return carregar_prateleira(exemplos / "prateleira.csv")


@pytest.fixture
def recomendacao(exemplos: Path, produtos: list[Produto], premissas: Premissas, curva: CurvaDI) -> Recomendacao:
    """Recomendação completa para o cliente de exemplo."""
    cliente = carregar_cliente(exemplos / "cliente.json")
    return otimizar_cliente(cliente, produtos, premissas, curva, quantidade=CENARIOS_NOS_TESTES)


def meta(nome: str = "Meta", atual: float = 10_000, alvo: float = 11_000, meses: int = 12, reserva: bool = False) -> Meta:
    """Cria uma meta simples."""
    return Meta(nome, atual, alvo, meses, reserva)


def por_nome(recomendacao: Recomendacao, nome: str):
    """Resultado da meta com o nome pedido."""
    return next(r for r in recomendacao.resultados if r.meta.nome == nome)


# --- Contas básicas ----------------------------------------------------------------

def test_retorno_necessario_a_mao() -> None:
    """De R$ 25 mil para R$ 36 mil em 4 anos: 1,44^(1/4) - 1 = 9,54% a.a."""
    assert retorno_necessario_aa(meta(atual=25_000, alvo=36_000, meses=48)) == pytest.approx(1.44 ** 0.25 - 1)
    assert retorno_necessario_aa(meta(atual=25_000, alvo=36_000, meses=48)) == pytest.approx(0.0954, abs=1e-4)


def test_cvar_a_mao() -> None:
    """Perdas de 1 a 100: os 5% piores são 96, 97, 98, 99 e 100, com média 98."""
    assert cvar(np.arange(1, 101), 0.95) == pytest.approx(98)
    assert cvar(np.arange(1, 11), 0.90) == pytest.approx(10)


# --- Programas lineares pequenos, resolvidos à mão --------------------------------------

@pytest.fixture
def seguro_e_arriscado() -> ProblemaCarteira:
    """Dois produtos e quatro cenários (nível de 75%: o CVaR é o pior cenário).

    Seguro: R$ 1 vira 1,10 em todos os cenários.
    Arriscado: vira 0,80; 1,30; 1,40; 1,50 (média 1,25).
    """
    fatores = np.array([[1.10, 0.80], [1.10, 1.30], [1.10, 1.40], [1.10, 1.50]])
    return ProblemaCarteira(fatores, 0.75, np.array([1.0, 1.0]), ())


def test_minimo_cvar_a_mao(seguro_e_arriscado: ProblemaCarteira) -> None:
    """Para a média chegar a 1,20: 1,10 (1 - w) + 1,25 w >= 1,20, ou seja, w >= 2/3.
    Como o arriscado piora o pior cenário, o ótimo é exatamente w = 2/3,
    com perda no pior cenário de 1 - (1,10 x 1/3 + 0,80 x 2/3) = -0,10 + 0,30 x 2/3 = 0,10.
    """
    pesos = minimizar_cvar(seguro_e_arriscado, fator_alvo=1.20)
    assert pesos == pytest.approx([1 / 3, 2 / 3], abs=1e-6)
    assert cvar(1 - seguro_e_arriscado.fatores @ pesos, 0.75) == pytest.approx(0.10, abs=1e-6)


def test_meta_impossivel_no_programa_linear(seguro_e_arriscado: ProblemaCarteira) -> None:
    """Nenhuma carteira tem média 1,30 (o máximo é 1,25): devolve None, sem erro."""
    assert minimizar_cvar(seguro_e_arriscado, fator_alvo=1.30) is None


def test_maximo_retorno_com_teto_de_cvar_a_mao(seguro_e_arriscado: ProblemaCarteira) -> None:
    """Com perda máxima de 5% no pior cenário: -0,10 + 0,30 w <= 0,05, ou seja, w <= 0,5.
    Como a média cresce com w, o ótimo é w = 0,5 (média 1,175).
    """
    pesos = maximizar_retorno(seguro_e_arriscado, cvar_maximo=0.05)
    assert pesos == pytest.approx([0.5, 0.5], abs=1e-6)


def test_limite_por_produto_e_restricao_extra(seguro_e_arriscado: ProblemaCarteira) -> None:
    """Com o arriscado limitado a 40% e uma restrição extra (arriscado <= 30%), o máximo retorno usa 30%."""
    problema = replace(
        seguro_e_arriscado,
        limites_superiores=np.array([1.0, 0.4]),
        restricoes=(RestricaoLinear(np.array([0.0, 1.0]), 0.3, "teste"),),
    )
    assert maximizar_retorno(problema, cvar_maximo=1.0) == pytest.approx([0.7, 0.3], abs=1e-6)


# --- Filtro de produtos --------------------------------------------------------------------

def nomes_elegiveis(produtos: list[Produto], alvo: Meta, perfil: str, premissas: Premissas) -> set[str]:
    """Nomes dos produtos que passam no filtro."""
    elegiveis, _ = filtrar_produtos(produtos, alvo, premissas.perfil(perfil), premissas)
    return {p.nome for p in elegiveis}


def test_reserva_de_emergencia_so_liquidez_diaria_e_risco_1(produtos: list[Produto], premissas: Premissas) -> None:
    """Na reserva só entram Tesouro Selic, CDB de liquidez diária e Fundo DI."""
    assert nomes_elegiveis(produtos, meta(reserva=True), "arrojado", premissas) == {
        "Tesouro Selic 2031", "CDB Aurora Liquidez Diária", "Fundo DI Horizonte",
    }


def test_filtro_de_prazo_para_meta_de_um_ano(produtos: list[Produto], premissas: Premissas) -> None:
    """Meta de 12 meses, perfil arrojado:
    - entram: pós-fixados líquidos (mesmo vencendo depois) e a LCI de 12 meses;
    - saem: CDBs sem liquidez que vencem depois, prefixados/IPCA+ que vencem depois,
      risco 3 ou mais (prazo mínimo de 24 meses) e renda variável (36 meses).
    """
    assert nomes_elegiveis(produtos, meta(meses=12), "arrojado", premissas) == {
        "Tesouro Selic 2031", "CDB Aurora Liquidez Diária", "Fundo DI Horizonte", "LCI Horizonte 1 ano",
    }


def test_filtro_explica_os_motivos(produtos: list[Produto], premissas: Premissas) -> None:
    """Cada produto excluído vem com o motivo em português."""
    _, excluidos = filtrar_produtos(produtos, meta(meses=12), premissas.perfil("arrojado"), premissas)
    motivos = {e.produto.nome: e.motivo for e in excluidos}
    assert motivos["CDB Aurora 2 anos"] == "vence depois da meta e não permite resgate antes"
    assert motivos["Tesouro Prefixado 2029"] == "vence depois da meta, e vender antes teria preço incerto"
    assert motivos["ETF Ações Brasil Atlas"] == "renda variável exige prazo de pelo menos 36 meses"
    assert motivos["CDB Boreal 3 anos"].startswith("vence depois da meta")


def test_filtro_de_perfil_e_de_carencia(produtos: list[Produto], premissas: Premissas) -> None:
    """Conservador (risco até 2) em meta de 6 meses: a LCI tem carência de 12 meses e sai."""
    _, excluidos = filtrar_produtos(produtos, meta(meses=6), premissas.perfil("conservador"), premissas)
    motivos = {e.produto.nome: e.motivo for e in excluidos}
    assert motivos["LCI Horizonte 1 ano"] == "a carência de 12 meses termina depois da meta"
    assert motivos["Fundo de Ações Small Caps Atlas"] == "risco acima do máximo do perfil conservador (até 2)"


def test_aplicacao_minima_maior_que_a_meta(produtos: list[Produto], premissas: Premissas) -> None:
    """Numa meta de R$ 300, o fundo de ações (mínimo de R$ 500) fica de fora."""
    _, excluidos = filtrar_produtos(produtos, meta(atual=300, alvo=400, meses=60), premissas.perfil("arrojado"), premissas)
    motivos = {e.produto.nome: e.motivo for e in excluidos}
    assert motivos["Fundo de Ações Small Caps Atlas"] == "a aplicação mínima é maior que o valor da meta"


# --- Cliente de exemplo de ponta a ponta ---------------------------------------------------

def test_reserva_de_emergencia_vem_primeiro(recomendacao: Recomendacao) -> None:
    """A reserva é otimizada antes das outras metas."""
    assert recomendacao.resultados[0].meta.reserva_emergencia


def test_todas_as_carteiras_somam_100_por_cento(recomendacao: Recomendacao) -> None:
    """Em cada meta com carteira, os pesos somam 1 e os valores somam o valor atual."""
    for resultado in recomendacao.resultados:
        assert sum(a.peso for a in resultado.alocacoes) == pytest.approx(1.0)
        assert sum(a.valor for a in resultado.alocacoes) == pytest.approx(resultado.meta.valor_atual)


def test_reserva_usa_so_produtos_permitidos(recomendacao: Recomendacao) -> None:
    """A carteira da reserva tem só produtos de liquidez diária e risco 1."""
    reserva = por_nome(recomendacao, "Reserva de emergência")
    assert reserva.situacao == ATINGIVEL
    assert all(a.produto.liquidez_diaria and a.produto.risco == 1 for a in reserva.alocacoes)


def test_limites_por_produto_e_renda_variavel(recomendacao: Recomendacao, premissas: Premissas) -> None:
    """Nenhum produto passa de 40% (exceto Tesouro Selic) e a renda variável fica em até 20% (moderado)."""
    for resultado in recomendacao.resultados:
        for alocacao in resultado.alocacoes:
            if alocacao.produto.indexador != "selic":
                assert alocacao.peso <= premissas.otimizacao.limite_por_produto + 1e-6
        renda_variavel = sum(a.peso for a in resultado.alocacoes if a.produto.renda_variavel)
        assert renda_variavel <= 0.20 + 1e-6


def test_aplicacao_minima_respeitada(recomendacao: Recomendacao) -> None:
    """Todo produto escolhido recebe pelo menos a sua aplicação mínima."""
    for resultado in recomendacao.resultados:
        for alocacao in resultado.alocacoes:
            assert alocacao.valor >= alocacao.produto.aplicacao_minima - 1e-6


def test_metas_atingiveis_cumprem_o_retorno_necessario(recomendacao: Recomendacao) -> None:
    """Nas metas atingíveis, o retorno esperado é pelo menos o necessário e o CVaR respeita o perfil."""
    for resultado in recomendacao.resultados:
        if resultado.situacao == ATINGIVEL:
            assert resultado.retorno_esperado_aa >= resultado.retorno_necessario_aa - 1e-6
            assert resultado.cvar <= recomendacao.perfil.cvar_maximo + 1e-6
            assert 0.5 < resultado.probabilidade_sucesso <= 1


def test_meta_impossivel_tratada_sem_erro(recomendacao: Recomendacao) -> None:
    """A Viagem precisa de 29% a.a.: vira meta difícil, com a carteira de maior retorno
    dentro do risco do perfil e chance de sucesso baixa.
    """
    viagem = por_nome(recomendacao, "Viagem")
    assert viagem.situacao == DIFICIL
    assert viagem.retorno_esperado_aa < viagem.retorno_necessario_aa
    assert viagem.cvar <= recomendacao.perfil.cvar_maximo + 1e-6
    assert viagem.probabilidade_sucesso < 0.05


def test_mesma_semente_mesma_carteira(exemplos: Path, produtos: list[Produto], premissas: Premissas, curva: CurvaDI) -> None:
    """Rodar duas vezes com a mesma semente dá a mesma carteira."""
    cliente = carregar_cliente(exemplos / "cliente.json")
    a = otimizar_cliente(cliente, produtos, premissas, curva, quantidade=CENARIOS_NOS_TESTES)
    b = otimizar_cliente(cliente, produtos, premissas, curva, quantidade=CENARIOS_NOS_TESTES)
    assert [[(x.produto.nome, x.peso) for x in r.alocacoes] for r in a.resultados] == \
           [[(x.produto.nome, x.peso) for x in r.alocacoes] for r in b.resultados]


# --- FGC -------------------------------------------------------------------------------------

def test_fgc_respeitado_numa_meta_grande(produtos: list[Produto], premissas: Premissas, curva: CurvaDI) -> None:
    """Meta de R$ 1 milhão: nenhum conglomerado passa de R$ 250 mil (contando os juros)."""
    grande = meta(atual=1_000_000, alvo=1_200_000, meses=24)
    resultado = otimizar_meta(grande, premissas.perfil("moderado"), produtos, premissas, curva, quantidade=CENARIOS_NOS_TESTES)
    assert resultado.situacao == ATINGIVEL
    assert all(valor <= 250_000 + TOLERANCIA_REAIS for valor in resultado.uso_fgc.values())
    assert violacoes_fgc([(a.produto, a.exposicao_fgc) for a in resultado.alocacoes], premissas.fgc) == []


def test_fgc_somado_entre_metas(produtos: list[Produto], premissas: Premissas, curva: CurvaDI) -> None:
    """Duas metas de R$ 400 mil: somando as duas, cada conglomerado fica em até R$ 250 mil."""
    cliente = Cliente("c-teste", "moderado", 50_000, False, (
        meta("Meta A", 400_000, 480_000, 24), meta("Meta B", 400_000, 480_000, 24),
    ))
    recomendacao = otimizar_cliente(cliente, produtos, premissas, curva, quantidade=CENARIOS_NOS_TESTES)
    assert all(r.alocacoes for r in recomendacao.resultados)
    assert all(valor <= 250_000 + TOLERANCIA_REAIS for valor in recomendacao.uso_fgc.values())
    usados_na_b = recomendacao.resultados[1].uso_fgc
    for conglomerado, valor in usados_na_b.items():
        assert valor + recomendacao.resultados[0].uso_fgc.get(conglomerado, 0) <= 250_000 + TOLERANCIA_REAIS


# --- Casos sem solução e reamostragem ----------------------------------------------------------

def test_sem_produto_elegivel_nao_quebra(produtos: list[Produto], premissas: Premissas, curva: CurvaDI) -> None:
    """Reserva de emergência numa prateleira só com renda variável: sem solução, sem erro."""
    acoes = [p for p in produtos if p.renda_variavel]
    resultado = otimizar_meta(meta(reserva=True), premissas.perfil("moderado"), acoes, premissas, curva)
    assert resultado.situacao == SEM_SOLUCAO
    assert resultado.alocacoes == ()
    assert resultado.probabilidade_sucesso is None
    assert len(resultado.excluidos) == len(acoes)


def test_reamostragem(produtos: list[Produto], premissas: Premissas, curva: CurvaDI) -> None:
    """Com reamostragem, a carteira continua válida e é reproduzível."""
    poucas = replace(premissas, otimizacao=replace(premissas.otimizacao, reamostragens=5))
    entrada = meta("Entrada", 25_000, 36_000, 48)
    perfil = poucas.perfil("moderado")

    def rodar():
        return otimizar_meta(entrada, perfil, produtos, poucas, curva, reamostragem=True, quantidade=500)

    primeira, segunda = rodar(), rodar()
    assert primeira.reamostragem is True
    assert sum(a.peso for a in primeira.alocacoes) == pytest.approx(1.0)
    assert all(a.valor >= a.produto.aplicacao_minima - 1e-6 for a in primeira.alocacoes)
    assert all(a.peso <= 0.40 + 1e-6 or a.produto.indexador == "selic" for a in primeira.alocacoes)
    assert [(a.produto.nome, a.peso) for a in primeira.alocacoes] == [(a.produto.nome, a.peso) for a in segunda.alocacoes]


def test_retorno_esperado_por_produto_e_anual(recomendacao: Recomendacao) -> None:
    """O retorno líquido esperado de cada produto fica entre 0% e 30% a.a. nos exemplos."""
    for resultado in recomendacao.resultados:
        for alocacao in resultado.alocacoes:
            assert 0 < alocacao.retorno_liquido_esperado_aa < 0.30
            assert math.isfinite(alocacao.retorno_liquido_esperado_aa)


# --- Emprego do cliente ------------------------------------------------------------------------

@pytest.fixture
def bancario_do_aurora(exemplos: Path, produtos: list[Produto], premissas: Premissas, curva: CurvaDI) -> Recomendacao:
    """Cliente de exemplo, mas trabalhando no Conglomerado Aurora (setor financeiro)."""
    cliente = replace(
        carregar_cliente(exemplos / "cliente.json"),
        empregador="Conglomerado Aurora", setor_trabalho="financeiro",
    )
    return otimizar_cliente(cliente, produtos, premissas, curva, quantidade=CENARIOS_NOS_TESTES)


def test_empregador_fica_fora_da_carteira(bancario_do_aurora: Recomendacao) -> None:
    """Quem trabalha no Aurora não recebe nenhum produto do Aurora, em nenhuma meta."""
    for resultado in bancario_do_aurora.resultados:
        assert all(a.produto.conglomerado != "Conglomerado Aurora" for a in resultado.alocacoes)
        motivos = {e.produto.nome: e.motivo for e in resultado.excluidos}
        assert motivos["CDB Aurora Liquidez Diária"].startswith("o cliente trabalha no Conglomerado Aurora")


def test_setor_do_trabalho_limitado_a_30_por_cento(bancario_do_aurora: Recomendacao) -> None:
    """Quem trabalha no setor financeiro fica com no máximo 30% de cada meta em produtos financeiros."""
    for resultado in bancario_do_aurora.resultados:
        financeiro = sum(a.peso for a in resultado.alocacoes if a.produto.setor == "financeiro")
        assert financeiro <= 0.30 + 1e-6


def test_sem_emprego_informado_nada_muda(recomendacao: Recomendacao) -> None:
    """Sem empregador nem setor, a Entrada do apartamento pode ter mais de 30% em bancos (como antes)."""
    entrada = por_nome(recomendacao, "Entrada do apartamento")
    assert sum(a.peso for a in entrada.alocacoes if a.produto.setor == "financeiro") > 0.30
