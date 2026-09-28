"""Mercado: curva de juros, inflação implícita e cenários de retorno.

Todos os retornos são anualizados. Um cenário diz quanto cada classe de
ativo rende, em média por ano, do dia da aplicação até o fim da meta.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from talos.impostos import MESES_POR_ANO
from talos.modelos import (
    ClasseCenario,
    ErroDeDados,
    Premissas,
    Produto,
    inteiro_csv,
    ler_csv,
    numero_csv,
)

COLUNAS_CURVA = ("prazo_dias_uteis", "taxa_aa")


# ---------------------------------------------------------------------------
# Curva de juros
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurvaDI:
    """Vértices da curva de DI futuro: prazo em dias úteis e taxa ao ano."""

    prazos_du: tuple[int, ...]
    taxas_aa: tuple[float, ...]
    dias_uteis_por_ano: int

    def __post_init__(self) -> None:
        """Confere se há vértices, em ordem crescente de prazo e com taxas possíveis."""
        if not self.prazos_du or len(self.prazos_du) != len(self.taxas_aa):
            raise ErroDeDados("curva DI: informe pelo menos um vértice, com prazo e taxa.")
        if self.prazos_du[0] <= 0 or any(b <= a for a, b in zip(self.prazos_du, self.prazos_du[1:])):
            raise ErroDeDados("curva DI: os prazos precisam ser positivos e em ordem crescente, sem repetição.")
        if any(taxa <= -1 for taxa in self.taxas_aa):
            raise ErroDeDados("curva DI: taxa de -100% ou menor não é possível.")


def carregar_curva_di(caminho: Path, premissas: Premissas) -> CurvaDI:
    """Lê o CSV da curva de DI futuro (linhas com `#` são ignoradas)."""
    linhas = ler_csv(caminho, COLUNAS_CURVA, "curva DI")
    return CurvaDI(
        prazos_du=tuple(inteiro_csv(linha["prazo_dias_uteis"], "curva DI, prazo_dias_uteis") for linha in linhas),
        taxas_aa=tuple(numero_csv(linha["taxa_aa"], "curva DI, taxa_aa") for linha in linhas),
        dias_uteis_por_ano=premissas.dias_uteis_por_ano,
    )


def meses_para_dias_uteis(prazo_meses: int, dias_uteis_por_ano: int) -> int:
    """Converte meses em dias úteis (12 meses = 252 dias úteis)."""
    return round(prazo_meses * dias_uteis_por_ano / MESES_POR_ANO)


def taxa_na_curva(curva: CurvaDI, dias_uteis: int) -> float:
    """Taxa ao ano da curva num prazo qualquer, por interpolação "flat forward".

    Entre dois vértices, a taxa diária a termo fica constante: interpola-se
    em linha reta o logaritmo do fator acumulado. Antes do primeiro vértice
    e depois do último, repete a taxa da ponta.
    """
    if dias_uteis <= 0:
        raise ErroDeDados("curva DI: o prazo precisa ser de pelo menos 1 dia útil.")
    prazos = np.array(curva.prazos_du, dtype=float)
    taxas = np.array(curva.taxas_aa, dtype=float)
    if dias_uteis <= prazos[0]:
        return float(taxas[0])
    if dias_uteis >= prazos[-1]:
        return float(taxas[-1])
    base = curva.dias_uteis_por_ano
    log_fatores = np.log1p(taxas) * prazos / base
    log_fator = np.interp(dias_uteis, prazos, log_fatores)
    return float(np.expm1(log_fator * base / dias_uteis))


def cdi_medio_projetado(curva: CurvaDI, prazo_meses: int) -> float:
    """CDI médio ao ano esperado pelo mercado entre hoje e daqui a `prazo_meses`."""
    return taxa_na_curva(curva, meses_para_dias_uteis(prazo_meses, curva.dias_uteis_por_ano))


def inflacao_implicita(curva: CurvaDI, prazo_meses: int, premissas: Premissas) -> float:
    """Inflação ao ano embutida nos juros: (1 + CDI projetado) / (1 + juro real) - 1."""
    return (1 + cdi_medio_projetado(curva, prazo_meses)) / (1 + premissas.juro_real_aa) - 1


# ---------------------------------------------------------------------------
# Cenários
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cenarios:
    """Retornos anualizados de cada classe, um valor por cenário, para um prazo.

    `medias_aa` guarda o retorno central de cada classe (CDI projetado mais
    o prêmio da classe). Para as classes de renda fixa, a média é o próprio
    CDI projetado: a curva já embute o que o mercado espera.
    """

    prazo_meses: int
    semente: int
    cdi_projetado_aa: float
    juro_real_aa: float
    dias_uteis_por_ano: int
    medias_aa: dict[str, float]
    retornos_aa: dict[str, npt.NDArray[np.float64]]
    graus_liberdade: float | None = None

    @property
    def anos(self) -> float:
        """Prazo em anos."""
        return self.prazo_meses / MESES_POR_ANO

    @property
    def quantidade(self) -> int:
        """Número de cenários."""
        return len(next(iter(self.retornos_aa.values())))

    def ipca_aa(self) -> npt.NDArray[np.float64]:
        """IPCA ao ano em cada cenário, tirado do retorno da classe de inflação.

        A classe `inflacao` representa um título que paga IPCA + juro real,
        então IPCA = (1 + retorno) / (1 + juro real) - 1.
        """
        return (1 + self.retornos_aa["inflacao"]) / (1 + self.juro_real_aa) - 1


def volatilidade_no_prazo(classe: ClasseCenario, anos: float) -> float:
    """Desvio-padrão do retorno anualizado para um prazo em anos."""
    if classe.escala_prazo == "raiz_do_prazo":
        return classe.volatilidade_aa / np.sqrt(anos)
    return classe.volatilidade_aa


def choques_t_correlacionados(
    quantidade: int,
    correlacoes: Sequence[Sequence[float]],
    graus_liberdade: float,
    gerador: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Sorteia choques de uma t de Student multivariada, com média 0 e variância 1.

    Devolve uma tabela com uma linha por cenário e uma coluna por classe.
    Um mesmo sorteio de "susto" divide todas as colunas de um cenário, então
    quedas fortes tendem a acontecer juntas, como nas crises.
    """
    matriz = np.asarray(correlacoes, dtype=float)
    try:
        fator = np.linalg.cholesky(matriz)
    except np.linalg.LinAlgError as erro:
        raise ErroDeDados(
            "cenarios: a matriz de correlações é impossível (não é positiva definida)."
        ) from erro
    normais = gerador.standard_normal((quantidade, len(matriz))) @ fator.T
    susto = np.sqrt(gerador.chisquare(graus_liberdade, size=quantidade) / graus_liberdade)
    ajuste_variancia = np.sqrt((graus_liberdade - 2) / graus_liberdade)
    return normais / susto[:, None] * ajuste_variancia


def choques_mensais_somados(
    quantidade: int,
    meses: int,
    correlacoes: Sequence[Sequence[float]],
    graus_liberdade: float,
    gerador: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Sorteia um choque t de Student correlacionado por mês e soma os meses.

    O resultado é dividido por raiz(meses) para ter variância 1. Num prazo de
    1 mês fica igual a um choque t (caudas gordas); em prazos longos os meses
    bons e ruins se compensam e a soma fica mais parecida com a normal, como
    acontece com retornos de verdade.
    """
    matriz = np.asarray(correlacoes, dtype=float)
    try:
        fator = np.linalg.cholesky(matriz)
    except np.linalg.LinAlgError as erro:
        raise ErroDeDados(
            "cenarios: a matriz de correlações é impossível (não é positiva definida)."
        ) from erro
    normais = gerador.standard_normal((meses, quantidade, len(matriz))) @ fator.T
    susto = np.sqrt(gerador.chisquare(graus_liberdade, size=(meses, quantidade)) / graus_liberdade)
    ajuste_variancia = np.sqrt((graus_liberdade - 2) / graus_liberdade)
    mensais = normais / susto[:, :, None] * ajuste_variancia
    return mensais.sum(axis=0) / np.sqrt(meses)


def gerar_cenarios(
    premissas: Premissas,
    curva: CurvaDI,
    prazo_meses: int,
    semente: int | None = None,
    quantidade: int | None = None,
) -> Cenarios:
    """Gera cenários de retorno anualizado de cada classe até `prazo_meses`.

    Para cada classe: log(1 + retorno) = log(1 + média) + desvio * choque,
    depois deslocado por uma constante (`centrar_na_media`) para que, nos
    cenários sorteados, R$ 1 vire em média exatamente (1 + média) ^ anos.
    Trabalhar no logaritmo impede perdas acima de 100%; o deslocamento evita
    que os poucos cenários extremos da t de Student inflem a média. A semente
    e a quantidade vêm das premissas quando não são informadas.
    """
    if prazo_meses < 1:
        raise ErroDeDados("cenarios: o prazo precisa ser de pelo menos 1 mês.")
    parametros = premissas.cenarios
    semente = parametros.semente if semente is None else semente
    quantidade = parametros.quantidade if quantidade is None else quantidade
    anos = prazo_meses / MESES_POR_ANO
    cdi = cdi_medio_projetado(curva, prazo_meses)

    gerador = np.random.default_rng(semente)
    choques = choques_mensais_somados(
        quantidade, prazo_meses, parametros.correlacoes, parametros.graus_liberdade, gerador
    )
    medias: dict[str, float] = {}
    retornos: dict[str, npt.NDArray[np.float64]] = {}
    for coluna, nome in enumerate(parametros.ordem):
        classe = parametros.classes[nome]
        medias[nome] = cdi + classe.premio_sobre_cdi_aa
        desvio = volatilidade_no_prazo(classe, anos)
        log_retorno = centrar_na_media(np.log1p(medias[nome]) + desvio * choques[:, coluna], medias[nome], anos)
        retornos[nome] = np.expm1(log_retorno)
        if np.any(retornos[nome] <= -1):
            raise ErroDeDados(
                f"cenarios, classe '{nome}': volatilidade alta demais para um prazo de "
                f"{prazo_meses} meses; alguns cenários chegaram a perder 100%."
            )

    return Cenarios(
        prazo_meses=prazo_meses,
        semente=semente,
        cdi_projetado_aa=cdi,
        juro_real_aa=premissas.juro_real_aa,
        dias_uteis_por_ano=premissas.dias_uteis_por_ano,
        medias_aa=medias,
        retornos_aa=retornos,
        graus_liberdade=parametros.graus_liberdade,
    )


def centrar_na_media(
    log_retorno_aa: npt.NDArray[np.float64], media_aa: float, anos: float
) -> npt.NDArray[np.float64]:
    """Desloca os log-retornos anuais para o patrimônio médio no fim do prazo ser (1 + média) ^ anos.

    O patrimônio de cada cenário é exp(anos * log-retorno). Somar a mesma
    constante a todos os cenários muda a média sem mudar a forma da
    distribuição (as caudas gordas continuam). A média dos exponenciais é
    calculada tirando o maior valor antes, para não estourar.
    """
    total = anos * log_retorno_aa
    maior = total.max()
    log_media_atual = maior + np.log(np.mean(np.exp(total - maior)))
    return log_retorno_aa + (anos * np.log1p(media_aa) - log_media_atual) / anos


# ---------------------------------------------------------------------------
# Retorno de cada produto nos cenários
# ---------------------------------------------------------------------------

def retorno_percentual_cdi(
    cdi_aa: npt.ArrayLike, fracao: float, dias_uteis_por_ano: int
) -> npt.NDArray[np.float64]:
    """Retorno ao ano de um papel que paga uma fração do CDI (ex.: 1.10 = 110%).

    A fração incide sobre a taxa de cada dia útil, como no mercado:
    taxa diária = (1 + CDI)^(1/252) - 1; retorno = (1 + fração * diária)^252 - 1.
    """
    diaria = (1 + np.asarray(cdi_aa, dtype=float)) ** (1 / dias_uteis_por_ano) - 1
    return (1 + fracao * diaria) ** dias_uteis_por_ano - 1


def retorno_antes_do_ir(produto: Produto, cenarios: Cenarios) -> npt.NDArray[np.float64]:
    """Retorno anual do produto em cada cenário, já sem a taxa de administração e antes do IR.

    - `cdi`: fração do CDI do cenário;
    - `selic`: CDI do cenário mais o spread (Selic e CDI andam quase juntos);
    - `prefixado`: a taxa contratada, mais o desvio da classe prefixada
      (zero nas premissas de exemplo, pois quem leva ao vencimento recebe a taxa);
    - `ipca`: IPCA do cenário mais o spread;
    - `variavel`: o retorno da própria classe do produto; para ações
      individuais, o retorno da bolsa ajustado pelo beta, pelo alfa e pela
      oscilação própria da empresa (veja `retorno_da_acao`).
    """
    retornos = cenarios.retornos_aa
    if produto.indexador == "cdi":
        bruto = retorno_percentual_cdi(retornos["pos_fixado"], produto.taxa, cenarios.dias_uteis_por_ano)
    elif produto.indexador == "selic":
        bruto = (1 + retornos["pos_fixado"]) * (1 + produto.taxa) - 1
    elif produto.indexador == "prefixado":
        desvio = (1 + retornos["prefixado"]) / (1 + cenarios.medias_aa["prefixado"])
        bruto = (1 + produto.taxa) * desvio - 1
    elif produto.indexador == "ipca":
        bruto = (1 + cenarios.ipca_aa()) * (1 + produto.taxa) - 1
    elif produto.indexador == "variavel" and produto.acao_individual:
        bruto = retorno_da_acao(produto, cenarios)
    elif produto.indexador == "variavel":
        bruto = retornos[produto.classe]
    else:
        raise ErroDeDados(f"produto '{produto.nome}': indexador '{produto.indexador}' não reconhecido.")
    return (1 + bruto) * (1 - produto.taxa_adm_aa) - 1


def retorno_da_acao(produto: Produto, cenarios: Cenarios) -> npt.NDArray[np.float64]:
    """Retorno anual de uma ação individual em cada cenário.

    Em logaritmos, com f = CDI do cenário e m = bolsa (classe da ação):
    log(1 + r) = log(1 + f) + beta * (log(1 + m) - log(1 + f)) + desvio * choque,
    onde desvio = volatilidade própria / raiz(anos). O choque é próprio da
    empresa (independente da bolsa e das outras ações), com as mesmas caudas
    gordas dos cenários, e sempre igual para a mesma ação, semente e prazo.

    Depois, os cenários são centrados (`centrar_na_media`) no retorno esperado
    pela análise: CDI projetado + beta * (média da bolsa - CDI projetado) + alfa.
    """
    livre = np.log1p(cenarios.retornos_aa["pos_fixado"])
    bolsa = np.log1p(cenarios.retornos_aa[produto.classe])
    desvio = produto.volatilidade_propria_aa / np.sqrt(cenarios.anos)
    log_retorno = livre + produto.beta * (bolsa - livre) + desvio * choques_proprios(produto, cenarios)
    cdi = cenarios.cdi_projetado_aa
    esperado = cdi + produto.beta * (cenarios.medias_aa[produto.classe] - cdi) + produto.alfa_aa
    return np.expm1(centrar_na_media(log_retorno, esperado, cenarios.anos))


def choques_proprios(produto: Produto, cenarios: Cenarios) -> npt.NDArray[np.float64]:
    """Choques da própria empresa, um por cenário, com variância 1.

    Como na bolsa, é um choque t de Student por mês, somado ao longo do prazo.
    A semente combina a semente dos cenários com um código fixo do nome do
    produto (CRC32), então cada ação tem sua sequência, sempre a mesma.
    """
    if produto.volatilidade_propria_aa == 0:
        return np.zeros(cenarios.quantidade)
    if cenarios.graus_liberdade is None:
        raise ErroDeDados("cenarios: faltam os graus de liberdade para sortear os choques das ações.")
    return _choques_da_empresa(
        produto.nome, cenarios.semente, cenarios.prazo_meses, cenarios.quantidade, cenarios.graus_liberdade
    )


@lru_cache(maxsize=256)
def _choques_da_empresa(
    nome: str, semente: int, meses: int, quantidade: int, graus_liberdade: float
) -> npt.NDArray[np.float64]:
    """Sorteia (uma vez só, depois reaproveita) os choques próprios de uma empresa.

    O resultado fica somente leitura, porque a mesma tabela é devolvida a
    cada chamada com os mesmos argumentos.
    """
    gerador = np.random.default_rng([semente, zlib.crc32(nome.encode("utf-8"))])
    choques = choques_mensais_somados(quantidade, meses, ((1.0,),), graus_liberdade, gerador)[:, 0]
    choques.setflags(write=False)
    return choques


def retornos_antes_do_ir(
    produtos: Sequence[Produto], cenarios: Cenarios
) -> npt.NDArray[np.float64]:
    """Tabela de retornos antes do IR: uma linha por cenário e uma coluna por produto."""
    return np.column_stack([retorno_antes_do_ir(produto, cenarios) for produto in produtos])
