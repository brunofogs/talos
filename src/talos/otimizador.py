"""Otimizador de carteiras, uma meta de cada vez.

Para cada meta:
1. filtra os produtos que servem para o prazo, o perfil e o tipo de meta;
2. simula quanto R$ 1 vira no fim da meta, líquido de IR, em cada cenário;
3. minimiza o CVaR (perda média nos piores cenários) exigindo que o
   patrimônio esperado alcance o valor-alvo, com a formulação linear de
   Rockafellar e Uryasev resolvida pelo `linprog` (HiGHS);
4. se isso não for possível dentro do risco do perfil, maximiza o retorno
   esperado com o CVaR limitado ao teto do perfil e marca a meta como difícil;
5. calcula a chance de atingir a meta nos cenários.

As perdas são medidas no fim do prazo da meta, como fração do valor
aplicado: perda = 1 - (quanto R$ 1 virou).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import linprog

from talos.fgc import RestricaoLinear, UsoFGC, restricoes_fgc, somar_usos
from talos.impostos import MESES_POR_ANO, retorno_liquido_anualizado
from talos.mercado import Cenarios, CurvaDI, gerar_cenarios, retorno_antes_do_ir
from talos.modelos import RISCO_MINIMO, Cliente, ErroDeDados, Meta, Perfil, Premissas, Produto

Matriz = npt.NDArray[np.float64]
Vetor = npt.NDArray[np.float64]

INDEXADORES_POS_FIXADOS = ("cdi", "selic")
TRIBUTACAO_REINVESTIMENTO = "regressivo"
RISCO_COM_PRAZO_MINIMO = 3
TOLERANCIA_NUMERICA = 1e-9
PERCENTIS_VALOR_FINAL = (5, 25, 50, 75, 95)

ATINGIVEL = "atingivel"
DIFICIL = "dificil"
SEM_SOLUCAO = "sem_solucao"

MENOR_RISCO = "menor_risco"
CRESCIMENTO = "crescimento"
ESTRATEGIAS = (MENOR_RISCO, CRESCIMENTO)


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProdutoExcluido:
    """Produto da prateleira que não pode ser usado numa meta, e o motivo."""

    produto: Produto
    motivo: str


@dataclass(frozen=True)
class Alocacao:
    """Quanto de uma meta vai para um produto.

    `exposicao_fgc` é o valor esperado no vencimento (ou no fim da meta)
    que conta para o limite do FGC; é zero para produtos sem cobertura.
    """

    produto: Produto
    peso: float
    valor: float
    retorno_liquido_esperado_aa: float
    exposicao_fgc: float


@dataclass(frozen=True)
class ResultadoMeta:
    """Carteira e estatísticas de uma meta.

    `situacao` é `atingivel`, `dificil` (a carteira mostrada é a de maior
    retorno dentro do risco do perfil) ou `sem_solucao` (nenhuma carteira
    possível). Nos dois primeiros casos as estatísticas estão preenchidas.
    `valor_final_por_percentil` diz quanto a carteira vale no fim do prazo,
    em reais, nos percentis 5, 25, 50, 75 e 95 dos cenários.
    """

    meta: Meta
    situacao: str
    alocacoes: tuple[Alocacao, ...]
    excluidos: tuple[ProdutoExcluido, ...]
    retorno_necessario_aa: float
    retorno_esperado_aa: float | None
    cvar: float | None
    probabilidade_sucesso: float | None
    cdi_projetado_aa: float | None
    uso_fgc: UsoFGC
    semente: int
    reamostragem: bool
    valor_final_por_percentil: dict[int, float] | None = None


@dataclass(frozen=True)
class Recomendacao:
    """Resultado de todas as metas de um cliente, na ordem em que foram otimizadas."""

    cliente: Cliente
    perfil: Perfil
    resultados: tuple[ResultadoMeta, ...]
    uso_fgc: UsoFGC
    semente: int
    reamostragem: bool
    estrategia: str = MENOR_RISCO


# ---------------------------------------------------------------------------
# Contas básicas
# ---------------------------------------------------------------------------

def retorno_necessario_aa(meta: Meta) -> float:
    """Retorno ao ano que leva o valor atual ao valor-alvo no prazo:
    (valor_alvo / valor_atual) ^ (1 / anos) - 1.
    """
    anos = meta.prazo_meses / MESES_POR_ANO
    return (meta.valor_alvo / meta.valor_atual) ** (1 / anos) - 1


def cvar(perdas: npt.ArrayLike, nivel: float) -> float:
    """Perda média nos piores `1 - nivel` dos cenários (CVaR).

    Usa a fórmula de Rockafellar-Uryasev com o VaR como ponto de corte:
    CVaR = VaR + média(max(perda - VaR, 0)) / (1 - nivel).
    Exemplo: perdas de 1 a 100 e nível 95% dão a média de 96 a 100 = 98.
    """
    perdas = np.asarray(perdas, dtype=float)
    ordenadas = np.sort(perdas)
    var = ordenadas[int(np.ceil(nivel * len(perdas))) - 1]
    return float(var + np.mean(np.maximum(perdas - var, 0)) / (1 - nivel))


# ---------------------------------------------------------------------------
# Filtro de produtos
# ---------------------------------------------------------------------------

def motivo_de_exclusao(
    produto: Produto, meta: Meta, perfil: Perfil, premissas: Premissas, empregador: str | None = None
) -> str | None:
    """Diz por que o produto não serve para a meta, ou `None` se ele serve.

    `empregador` é o conglomerado onde o cliente trabalha: os produtos dele
    ficam de fora, porque uma crise no grupo atingiria emprego e investimento
    ao mesmo tempo.
    """
    otimizacao = premissas.otimizacao
    prazo = meta.prazo_meses
    if empregador is not None and produto.conglomerado == empregador:
        return (
            f"o cliente trabalha no {empregador}; se emprego e investimento dependem do mesmo "
            "grupo, uma crise atinge os dois ao mesmo tempo"
        )
    if meta.reserva_emergencia and not produto.liquidez_diaria:
        return "a reserva de emergência exige liquidez diária"
    if meta.reserva_emergencia and produto.risco > RISCO_MINIMO:
        return f"a reserva de emergência aceita só produtos de risco {RISCO_MINIMO}"
    if produto.risco > perfil.risco_maximo:
        return f"risco acima do máximo do perfil {perfil.nome} (até {perfil.risco_maximo})"
    if produto.carencia_meses > prazo:
        return f"a carência de {produto.carencia_meses} meses termina depois da meta"
    if produto.vencimento_meses is not None and produto.vencimento_meses > prazo:
        if not produto.liquidez_diaria:
            return "vence depois da meta e não permite resgate antes"
        if produto.indexador not in INDEXADORES_POS_FIXADOS:
            return "vence depois da meta, e vender antes teria preço incerto"
    if produto.renda_variavel and prazo < otimizacao.prazo_minimo_renda_variavel_meses:
        return f"renda variável exige prazo de pelo menos {otimizacao.prazo_minimo_renda_variavel_meses} meses"
    if produto.risco >= RISCO_COM_PRAZO_MINIMO and prazo < otimizacao.prazo_minimo_risco_3_meses:
        return (
            f"produtos de risco {RISCO_COM_PRAZO_MINIMO} ou mais exigem prazo de pelo menos "
            f"{otimizacao.prazo_minimo_risco_3_meses} meses"
        )
    if produto.aplicacao_minima > meta.valor_atual:
        return "a aplicação mínima é maior que o valor da meta"
    return None


def filtrar_produtos(
    produtos: Sequence[Produto],
    meta: Meta,
    perfil: Perfil,
    premissas: Premissas,
    empregador: str | None = None,
) -> tuple[list[Produto], list[ProdutoExcluido]]:
    """Separa os produtos que servem para a meta dos que não servem (com o motivo)."""
    elegiveis: list[Produto] = []
    excluidos: list[ProdutoExcluido] = []
    for produto in produtos:
        motivo = motivo_de_exclusao(produto, meta, perfil, premissas, empregador)
        if motivo is None:
            elegiveis.append(produto)
        else:
            excluidos.append(ProdutoExcluido(produto, motivo))
    return elegiveis, excluidos


# ---------------------------------------------------------------------------
# Retornos líquidos nos cenários
# ---------------------------------------------------------------------------

def fator_liquido(retorno_aa: Vetor, tributacao: str, meses: int, premissas: Premissas) -> Vetor:
    """Quanto R$ 1 vira em `meses`, depois do IR, para cada retorno anual."""
    liquido = retorno_liquido_anualizado(retorno_aa, tributacao, meses, premissas)
    return (1 + np.asarray(liquido)) ** (meses / MESES_POR_ANO)


def fatores_liquidos(produtos: Sequence[Produto], cenarios: Cenarios, premissas: Premissas) -> Matriz:
    """Quanto R$ 1 vira no fim da meta, líquido de IR: uma linha por cenário, uma coluna por produto.

    Se o produto vence antes da meta, o dinheiro é reaplicado a 100% do CDI
    do cenário (um CDB comum, com IR regressivo) até o fim da meta.
    """
    prazo = cenarios.prazo_meses
    cdi = cenarios.retornos_aa["pos_fixado"]
    colunas = []
    for produto in produtos:
        bruto = retorno_antes_do_ir(produto, cenarios)
        vencimento = produto.vencimento_meses
        if vencimento is not None and vencimento < prazo:
            ate_vencer = fator_liquido(bruto, produto.tributacao, vencimento, premissas)
            reaplicado = fator_liquido(cdi, TRIBUTACAO_REINVESTIMENTO, prazo - vencimento, premissas)
            colunas.append(ate_vencer * reaplicado)
        else:
            colunas.append(fator_liquido(bruto, produto.tributacao, prazo, premissas))
    return np.column_stack(colunas)


def fatores_crescimento_fgc(produtos: Sequence[Produto], cenarios: Cenarios) -> Vetor:
    """Quanto R$ 1 aplicado deve valer, em média, no vencimento (ou no fim da meta).

    É esse valor, com juros, que conta para o limite do FGC.
    """
    fatores = []
    for produto in produtos:
        meses = min(produto.vencimento_meses or cenarios.prazo_meses, cenarios.prazo_meses)
        bruto = retorno_antes_do_ir(produto, cenarios)
        fatores.append(float(np.mean((1 + bruto) ** (meses / MESES_POR_ANO))))
    return np.array(fatores)


# ---------------------------------------------------------------------------
# Programas lineares
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProblemaCarteira:
    """Dados numéricos de um problema de carteira.

    `fatores[s, i]` é quanto R$ 1 no produto `i` vira no cenário `s`. Os
    pesos ficam entre 0 e `limites_superiores`, somam 1 e respeitam as
    `restricoes` (FGC, teto de renda variável).
    """

    fatores: Matriz
    nivel_cvar: float
    limites_superiores: Vetor
    restricoes: tuple[RestricaoLinear, ...]


def minimizar_cvar(problema: ProblemaCarteira, fator_alvo: float) -> Vetor | None:
    """Pesos de menor CVaR cujo patrimônio esperado chega a `fator_alvo`, ou `None`.

    Variáveis: pesos w, o VaR `a` e o excesso de perda u[s] de cada cenário.
    Minimiza  a + soma(u) / ((1 - nível) * S)
    sujeito a u[s] >= perda[s] - a,  u >= 0,  média(fatores) . w >= fator_alvo.
    """
    quantidade, n = problema.fatores.shape
    custo = np.concatenate([np.zeros(n), [1.0], np.full(quantidade, _peso_excesso(problema))])
    retorno_minimo = (np.concatenate([-problema.fatores.mean(axis=0), np.zeros(1 + quantidade)]), -fator_alvo)
    return _resolver(problema, custo, [retorno_minimo])


def maximizar_retorno(problema: ProblemaCarteira, cvar_maximo: float) -> Vetor | None:
    """Pesos de maior patrimônio esperado com CVaR até `cvar_maximo`, ou `None`."""
    quantidade, n = problema.fatores.shape
    custo = np.concatenate([-problema.fatores.mean(axis=0), np.zeros(1 + quantidade)])
    teto_cvar = (np.concatenate([np.zeros(n), [1.0], np.full(quantidade, _peso_excesso(problema))]), cvar_maximo)
    return _resolver(problema, custo, [teto_cvar])


def maximizar_crescimento(problema: ProblemaCarteira, cvar_maximo: float, nivel_tipico: float) -> Vetor | None:
    """Pesos que maximizam a média dos piores `1 - nivel_tipico` dos cenários, com CVaR até `cvar_maximo`.

    Com `nivel_tipico` = 0,5, maximiza a média da metade pior dos cenários:
    premia carteiras que crescem de forma consistente, e não as que só vão bem
    em poucos cenários de sorte. São dois blocos de Rockafellar-Uryasev: um
    para o teto de risco (nível do problema) e outro para o objetivo.
    Variáveis: pesos w, (a1, u1) do teto e (a2, u2) do objetivo.
    """
    quantidade, n = problema.fatores.shape
    zeros = np.zeros(quantidade)
    peso_teto = _peso_excesso(problema)
    peso_objetivo = 1 / ((1 - nivel_tipico) * quantidade)
    custo = np.concatenate([np.zeros(n), [0.0], zeros, [1.0], np.full(quantidade, peso_objetivo)])

    cenarios_negativos = sparse.csr_matrix(-problema.fatores)
    menos_um = sparse.csr_matrix(-np.ones((quantidade, 1)))
    identidade = -sparse.identity(quantidade, format="csr")
    vazio_coluna = sparse.csr_matrix((quantidade, 1))
    vazio_bloco = sparse.csr_matrix((quantidade, quantidade))
    bloco_teto = sparse.hstack([cenarios_negativos, menos_um, identidade, vazio_coluna, vazio_bloco])
    bloco_objetivo = sparse.hstack([cenarios_negativos, vazio_coluna, vazio_bloco, menos_um, identidade])

    tamanho_extra = 2 + 2 * quantidade
    linhas = [(np.concatenate([r.coeficientes, np.zeros(tamanho_extra)]), r.limite) for r in problema.restricoes]
    linhas.append((np.concatenate([np.zeros(n), [1.0], np.full(quantidade, peso_teto), [0.0], zeros]), cvar_maximo))
    a_ub = sparse.vstack([bloco_teto, bloco_objetivo, sparse.csr_matrix(np.array([l for l, _ in linhas]))])
    b_ub = np.concatenate([-np.ones(2 * quantidade), [limite for _, limite in linhas]])
    a_eq = sparse.csr_matrix(np.concatenate([np.ones(n), np.zeros(tamanho_extra)])[None, :])
    limites = (
        [(0.0, float(teto)) for teto in problema.limites_superiores]
        + [(None, None)] + [(0.0, None)] * quantidade
        + [(None, None)] + [(0.0, None)] * quantidade
    )
    resultado = linprog(custo, A_ub=a_ub.tocsr(), b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=limites, method="highs")
    if resultado.status != 0:
        return None
    return _limpar_pesos(resultado.x[:n])


def _peso_excesso(problema: ProblemaCarteira) -> float:
    """Peso de cada excesso de perda na fórmula do CVaR: 1 / ((1 - nível) * S)."""
    return 1 / ((1 - problema.nivel_cvar) * problema.fatores.shape[0])


def _resolver(
    problema: ProblemaCarteira, custo: Vetor, linhas_extras: list[tuple[Vetor, float]]
) -> Vetor | None:
    """Monta e resolve o programa linear; devolve os pesos limpos ou `None` se não há solução."""
    quantidade, n = problema.fatores.shape
    # Linhas dos cenários: -fatores . w - a - u[s] <= -1  (ou seja, u[s] >= perda[s] - a)
    linhas_cenarios = sparse.hstack([
        sparse.csr_matrix(-problema.fatores),
        sparse.csr_matrix(-np.ones((quantidade, 1))),
        -sparse.identity(quantidade, format="csr"),
    ])
    linhas = [(np.concatenate([r.coeficientes, np.zeros(1 + quantidade)]), r.limite) for r in problema.restricoes]
    linhas += linhas_extras
    a_ub = sparse.vstack([linhas_cenarios, sparse.csr_matrix(np.array([coeficientes for coeficientes, _ in linhas]))])
    b_ub = np.concatenate([-np.ones(quantidade), [limite for _, limite in linhas]])
    a_eq = sparse.csr_matrix(np.concatenate([np.ones(n), np.zeros(1 + quantidade)])[None, :])
    limites = (
        [(0.0, float(teto)) for teto in problema.limites_superiores]
        + [(None, None)]
        + [(0.0, None)] * quantidade
    )
    resultado = linprog(custo, A_ub=a_ub.tocsr(), b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=limites, method="highs")
    if resultado.status != 0:
        return None
    return _limpar_pesos(resultado.x[:n])


def _limpar_pesos(pesos: Vetor) -> Vetor:
    """Zera pesos minúsculos (ruído numérico) e faz a soma voltar a 1."""
    pesos = np.where(pesos < TOLERANCIA_NUMERICA, 0.0, pesos)
    return pesos / pesos.sum()


# ---------------------------------------------------------------------------
# Otimização de uma meta
# ---------------------------------------------------------------------------

Resolvedor = Callable[[list[int], Matriz], Optional[Vetor]]


def otimizar_meta(
    meta: Meta,
    perfil: Perfil,
    produtos: Sequence[Produto],
    premissas: Premissas,
    curva: CurvaDI,
    uso_fgc: UsoFGC | None = None,
    reamostragem: bool = False,
    semente: int | None = None,
    quantidade: int | None = None,
    empregador: str | None = None,
    setor_trabalho: str | None = None,
    estrategia: str = MENOR_RISCO,
) -> ResultadoMeta:
    """Monta a carteira de uma meta.

    Estratégias:
    - `menor_risco` (padrão): a carteira de menor CVaR que chega ao valor-alvo;
      se não houver, a de maior retorno dentro do risco do perfil (meta difícil);
    - `crescimento`: a carteira que mais cresce nos cenários comuns: maximiza a
      média dos piores 50% dos cenários (nível das premissas), dentro do risco
      aceito pelo perfil. Não aposta nos poucos cenários de muita sorte.

    `uso_fgc` é o quanto das garantias do FGC as metas anteriores já usaram.
    Os produtos do `empregador` (conglomerado) ficam fora da carteira, e os do
    `setor_trabalho` do cliente ficam limitados ao teto das premissas.
    Com `reamostragem=True`, a carteira é a média das carteiras ótimas em
    várias reamostragens dos cenários (reamostragem de Michaud).
    """
    if estrategia not in ESTRATEGIAS:
        raise ErroDeDados(f"estratégia '{estrategia}' não reconhecida. Use: {', '.join(ESTRATEGIAS)}.")
    uso_fgc = uso_fgc or {}
    semente = premissas.cenarios.semente if semente is None else semente
    necessario = retorno_necessario_aa(meta)
    elegiveis, excluidos = filtrar_produtos(produtos, meta, perfil, premissas, empregador)
    sem_solucao = ResultadoMeta(
        meta, SEM_SOLUCAO, (), tuple(excluidos), necessario, None, None, None, None, {}, semente, reamostragem,
    )
    if not elegiveis:
        return sem_solucao

    cenarios = gerar_cenarios(premissas, curva, meta.prazo_meses, semente, quantidade)
    fatores = fatores_liquidos(elegiveis, cenarios, premissas)
    crescimento = fatores_crescimento_fgc(elegiveis, cenarios)
    fator_alvo = meta.valor_alvo / meta.valor_atual
    nivel = premissas.otimizacao.nivel_cvar

    def problema(indices: list[int], fatores_usados: Matriz) -> ProblemaCarteira:
        return _montar_problema(
            [elegiveis[i] for i in indices], fatores_usados[:, indices], crescimento[indices],
            meta, perfil, premissas, uso_fgc, setor_trabalho,
        )

    def menor_risco(indices: list[int], fatores_usados: Matriz) -> Vetor | None:
        pesos = minimizar_cvar(problema(indices, fatores_usados), fator_alvo)
        if pesos is None or cvar(1 - fatores_usados[:, indices] @ pesos, nivel) > perfil.cvar_maximo:
            return None
        return pesos

    def maior_retorno(indices: list[int], fatores_usados: Matriz) -> Vetor | None:
        return maximizar_retorno(problema(indices, fatores_usados), perfil.cvar_maximo)

    def mais_crescimento(indices: list[int], fatores_usados: Matriz) -> Vetor | None:
        return maximizar_crescimento(
            problema(indices, fatores_usados), perfil.cvar_maximo, premissas.otimizacao.nivel_crescimento
        )

    peso_minimo = premissas.otimizacao.peso_minimo_por_produto
    minimos = np.array([max(p.aplicacao_minima / meta.valor_atual, peso_minimo) for p in elegiveis])
    if estrategia == CRESCIMENTO:
        situacao, resolvedor = ATINGIVEL, mais_crescimento
        solucao = _com_aplicacao_minima(lambda indices: mais_crescimento(indices, fatores), minimos)
    else:
        situacao, resolvedor = ATINGIVEL, menor_risco
        solucao = _com_aplicacao_minima(lambda indices: menor_risco(indices, fatores), minimos)
    if solucao is None:
        situacao, resolvedor = DIFICIL, maior_retorno
        solucao = _com_aplicacao_minima(lambda indices: maior_retorno(indices, fatores), minimos)
    if solucao is None:
        return sem_solucao

    if reamostragem:
        gerador = np.random.default_rng([semente, 1])
        repeticoes = premissas.otimizacao.reamostragens
        reamostrada = _com_aplicacao_minima(
            lambda indices: _media_reamostrada(resolvedor, indices, fatores, repeticoes, gerador), minimos
        )
        solucao = reamostrada or solucao

    indices, pesos = solucao
    if resolvedor is mais_crescimento:
        chega_ao_alvo = np.mean(fatores[:, indices] @ pesos) >= fator_alvo - TOLERANCIA_NUMERICA
        situacao = ATINGIVEL if chega_ao_alvo else DIFICIL
    return _resultado(
        meta, situacao, elegiveis, indices, pesos, fatores, crescimento, tuple(excluidos),
        necessario, cenarios, nivel, semente, reamostragem,
    )


def _montar_problema(
    produtos: Sequence[Produto],
    fatores: Matriz,
    crescimento: Vetor,
    meta: Meta,
    perfil: Perfil,
    premissas: Premissas,
    uso_fgc: UsoFGC,
    setor_trabalho: str | None = None,
) -> ProblemaCarteira:
    """Junta limites por produto, FGC, teto de renda variável e teto do setor do trabalho."""
    otimizacao = premissas.otimizacao
    limites = np.array([limite_do_produto(p, premissas) for p in produtos])
    restricoes = restricoes_fgc(produtos, meta.valor_atual, uso_fgc, premissas.fgc, crescimento)
    renda_variavel = np.array([p.renda_variavel for p in produtos], dtype=float)
    if renda_variavel.any():
        restricoes.append(RestricaoLinear(renda_variavel, perfil.teto_renda_variavel, "teto de renda variável do perfil"))
    for conglomerado in sorted({p.conglomerado for p in produtos} - set(otimizacao.conglomerados_sem_limite)):
        do_grupo = np.array([p.conglomerado == conglomerado for p in produtos], dtype=float)
        if do_grupo @ limites > otimizacao.limite_por_conglomerado_na_meta:
            restricoes.append(RestricaoLinear(
                do_grupo, otimizacao.limite_por_conglomerado_na_meta, f"concentração no {conglomerado}"
            ))
    mesmo_setor = np.array([p.setor == setor_trabalho for p in produtos], dtype=float)
    if setor_trabalho is not None and mesmo_setor.any():
        restricoes.append(RestricaoLinear(
            mesmo_setor, otimizacao.limite_setor_do_trabalho, "teto do setor em que o cliente trabalha"
        ))
    return ProblemaCarteira(fatores, otimizacao.nivel_cvar, limites, tuple(restricoes))


def limite_do_produto(produto: Produto, premissas: Premissas) -> float:
    """Fração máxima da meta num produto: sem limite (Tesouro Selic), 40% ou, para ações, 10%."""
    otimizacao = premissas.otimizacao
    if produto.indexador in otimizacao.indexadores_sem_limite:
        return 1.0
    if produto.acao_individual:
        return min(otimizacao.limite_por_produto, otimizacao.limite_por_acao)
    return otimizacao.limite_por_produto


def _com_aplicacao_minima(
    resolver: Callable[[list[int]], Vetor | None], minimos: Vetor
) -> tuple[list[int], Vetor] | None:
    """Resolve e, se algum produto ficou abaixo da aplicação mínima, tira-o e resolve de novo.

    O `linprog` não aceita a regra "zero ou pelo menos o mínimo", então este
    laço é uma aproximação: pode deixar de fora uma carteira que usaria o
    produto acima do mínimo.
    """
    indices = list(range(len(minimos)))
    while indices:
        pesos = resolver(indices)
        if pesos is None:
            return None
        abaixo = {i for i, peso in zip(indices, pesos) if 0 < peso < minimos[i] - TOLERANCIA_NUMERICA}
        if not abaixo:
            return indices, pesos
        indices = [i for i in indices if i not in abaixo]
    return None


def _media_reamostrada(
    resolvedor: Resolvedor, indices: list[int], fatores: Matriz, repeticoes: int, gerador: np.random.Generator
) -> Vetor | None:
    """Média das carteiras ótimas em `repeticoes` reamostragens dos cenários (com reposição).

    Reamostrar os cenários perturba as estimativas de retorno e risco; a
    média das carteiras fica mais estável e diversificada (Michaud).
    """
    quantidade = fatores.shape[0]
    carteiras = []
    for _ in range(repeticoes):
        amostra = gerador.integers(0, quantidade, size=quantidade)
        pesos = resolvedor(indices, fatores[amostra])
        if pesos is not None:
            carteiras.append(pesos)
    if not carteiras:
        return None
    return _limpar_pesos(np.mean(carteiras, axis=0))


def _resultado(
    meta: Meta,
    situacao: str,
    elegiveis: Sequence[Produto],
    indices: list[int],
    pesos: Vetor,
    fatores: Matriz,
    crescimento: Vetor,
    excluidos: tuple[ProdutoExcluido, ...],
    necessario: float,
    cenarios: Cenarios,
    nivel: float,
    semente: int,
    reamostragem: bool,
) -> ResultadoMeta:
    """Calcula as estatísticas da carteira nos cenários originais e monta o `ResultadoMeta`."""
    anos = meta.prazo_meses / MESES_POR_ANO
    patrimonio = fatores[:, indices] @ pesos
    alocacoes = []
    uso: UsoFGC = {}
    for i, peso in zip(indices, pesos):
        if peso <= 0:
            continue
        produto = elegiveis[i]
        valor = float(peso * meta.valor_atual)
        exposicao = float(valor * crescimento[i]) if produto.coberto_fgc else 0.0
        if produto.coberto_fgc:
            uso = somar_usos(uso, {produto.conglomerado: exposicao})
        alocacoes.append(Alocacao(
            produto=produto,
            peso=float(peso),
            valor=valor,
            retorno_liquido_esperado_aa=float(np.mean(fatores[:, i]) ** (1 / anos) - 1),
            exposicao_fgc=exposicao,
        ))
    alocacoes.sort(key=lambda a: a.peso, reverse=True)
    return ResultadoMeta(
        meta=meta,
        situacao=situacao,
        alocacoes=tuple(alocacoes),
        excluidos=excluidos,
        retorno_necessario_aa=necessario,
        retorno_esperado_aa=float(np.mean(patrimonio) ** (1 / anos) - 1),
        cvar=cvar(1 - patrimonio, nivel),
        probabilidade_sucesso=float(np.mean(patrimonio >= meta.valor_alvo / meta.valor_atual)),
        cdi_projetado_aa=cenarios.cdi_projetado_aa,
        uso_fgc=uso,
        semente=semente,
        reamostragem=reamostragem,
        valor_final_por_percentil={
            percentil: float(np.percentile(patrimonio, percentil) * meta.valor_atual)
            for percentil in PERCENTIS_VALOR_FINAL
        },
    )


# ---------------------------------------------------------------------------
# Todas as metas de um cliente
# ---------------------------------------------------------------------------

def otimizar_cliente(
    cliente: Cliente,
    produtos: Sequence[Produto],
    premissas: Premissas,
    curva: CurvaDI,
    reamostragem: bool = False,
    semente: int | None = None,
    quantidade: int | None = None,
    estrategia: str = MENOR_RISCO,
) -> Recomendacao:
    """Otimiza todas as metas do cliente, começando pela reserva de emergência.

    O uso do FGC de cada meta é descontado dos limites das metas seguintes.
    O empregador e o setor de trabalho do cliente valem para todas as metas.
    """
    perfil = premissas.perfil(cliente.perfil)
    semente = premissas.cenarios.semente if semente is None else semente
    uso: UsoFGC = {}
    resultados = []
    for meta in sorted(cliente.metas, key=lambda m: not m.reserva_emergencia):
        resultado = otimizar_meta(
            meta, perfil, produtos, premissas, curva, uso, reamostragem, semente, quantidade,
            empregador=cliente.empregador, setor_trabalho=cliente.setor_trabalho, estrategia=estrategia,
        )
        resultados.append(resultado)
        uso = somar_usos(uso, resultado.uso_fgc)
    return Recomendacao(cliente, perfil, tuple(resultados), uso, semente, reamostragem, estrategia)
