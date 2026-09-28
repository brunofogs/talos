"""Explicações em português para cada recomendação, geradas por regras fixas.

Nada aqui usa IA generativa: cada frase vem de um molde preenchido com os
números da carteira, dos produtos e das premissas. Assim, a mesma entrada
sempre gera o mesmo texto, e dá para auditar de onde veio cada afirmação.
"""

from __future__ import annotations

from dataclasses import dataclass

from talos import AVISO_SIMULACAO
from talos.fgc import UsoFGC, formatar_reais, somar_usos
from talos.impostos import MESES_POR_ANO, TRIBUTACOES_COM_COME_COTAS, percentual_cdi_equivalente
from talos.modelos import Meta, Premissas
from talos.otimizador import (
    ATINGIVEL,
    CRESCIMENTO,
    DIFICIL,
    Alocacao,
    Recomendacao,
    ResultadoMeta,
    limite_do_produto,
)

NOMES_DOS_MESES = (
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
)
NOMES_DOS_SETORES = {
    "financeiro": "financeiro",
    "governo_federal": "governo federal",
    "energia": "de energia",
    "diversificado": "diversificado",
    "mineracao": "de mineração",
    "petroleo": "de petróleo",
    "varejo": "de varejo",
    "saude": "de saúde",
    "industria": "de indústria",
    "telecom": "de telecomunicações",
    "agronegocio": "de agronegócio",
    "saneamento": "de saneamento",
}
PROBABILIDADE_MINIMA_EXIBIDA = 0.001
TOLERANCIA_LIMITE = 1e-6


@dataclass(frozen=True)
class ExplicacaoMeta:
    """Frases de uma meta: resumo, uma lista por produto escolhido e os produtos não usados."""

    meta: str
    resumo: tuple[str, ...]
    produtos: dict[str, tuple[str, ...]]
    nao_usados: tuple[str, ...]


@dataclass(frozen=True)
class Explicacao:
    """Todas as frases de uma recomendação, terminando sempre com o aviso de simulação."""

    antes_de_tudo: tuple[str, ...]
    metas: tuple[ExplicacaoMeta, ...]
    aviso: str


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------

def formatar_percentual(fracao: float, casas: int = 2) -> str:
    """Escreve uma fração como porcentagem brasileira: 0.1159 -> "11,59%"."""
    return f"{fracao * 100:.{casas}f}".replace(".", ",") + "%"


def formatar_percentual_cdi(fracao: float) -> str:
    """Escreve um percentual do CDI com até uma casa: 0.94 -> "94%", 1.1059 -> "110,6%"."""
    texto = f"{fracao * 100:.1f}".replace(".", ",")
    return texto.removesuffix(",0") + "%"


def formatar_prazo(meses: int) -> str:
    """Escreve um prazo em anos quando dá conta exata: 12 -> "1 ano", 18 -> "18 meses"."""
    if meses % MESES_POR_ANO == 0:
        anos = meses // MESES_POR_ANO
        return "1 ano" if anos == 1 else f"{anos} anos"
    return "1 mês" if meses == 1 else f"{meses} meses"


def formatar_probabilidade(probabilidade: float) -> str:
    """Escreve a chance de sucesso sem arredondar para 0% ou 100% enganosos."""
    if probabilidade < PROBABILIDADE_MINIMA_EXIBIDA:
        return f"menos de {formatar_percentual(PROBABILIDADE_MINIMA_EXIBIDA, 1)}"
    if probabilidade > 1 - PROBABILIDADE_MINIMA_EXIBIDA:
        return f"mais de {formatar_percentual(1 - PROBABILIDADE_MINIMA_EXIBIDA, 1)}"
    return formatar_percentual(probabilidade, 1)


# ---------------------------------------------------------------------------
# Explicação completa
# ---------------------------------------------------------------------------

def explicar(recomendacao: Recomendacao, premissas: Premissas) -> Explicacao:
    """Gera as frases de toda a recomendação, meta a meta, na ordem em que foram otimizadas."""
    antes_de_tudo = []
    if recomendacao.cliente.dividas_caras:
        antes_de_tudo.append(
            "Antes de investir, quite as dívidas caras (cartão de crédito, cheque especial, "
            "empréstimos com juros altos). Os juros delas costumam ser bem maiores do que o "
            "rendimento de qualquer produto desta carteira."
        )
    antes_de_tudo.extend(_frases_de_trabalho(recomendacao, premissas))
    if recomendacao.estrategia == CRESCIMENTO:
        antes_de_tudo.append(
            "Estratégia de crescimento: em vez da carteira de menor risco que atinge cada meta, o Talos "
            "buscou a que mais cresce nos cenários comuns (a média da metade pior dos cenários), sem "
            "passar do risco nem do teto de renda variável do perfil."
        )
    uso_acumulado: UsoFGC = {}
    metas = []
    for resultado in recomendacao.resultados:
        uso_acumulado = somar_usos(uso_acumulado, resultado.uso_fgc)
        metas.append(explicar_meta(
            resultado, recomendacao.perfil.nome, premissas, uso_acumulado, recomendacao.cliente.setor_trabalho
        ))
    return Explicacao(tuple(antes_de_tudo), tuple(metas), AVISO_SIMULACAO)


def explicar_meta(
    resultado: ResultadoMeta,
    nome_perfil: str,
    premissas: Premissas,
    uso_fgc_acumulado: UsoFGC,
    setor_trabalho: str | None = None,
) -> ExplicacaoMeta:
    """Frases de uma meta. `uso_fgc_acumulado` inclui esta meta e as anteriores."""
    return ExplicacaoMeta(
        meta=resultado.meta.nome,
        resumo=tuple(_resumo(resultado, nome_perfil, premissas)),
        produtos={
            alocacao.produto.nome: tuple(
                _frases_do_produto(alocacao, resultado.meta, premissas, uso_fgc_acumulado, setor_trabalho)
            )
            for alocacao in resultado.alocacoes
        },
        nao_usados=tuple(_nao_usados(resultado)),
    )


def _resumo(resultado: ResultadoMeta, nome_perfil: str, premissas: Premissas) -> list[str]:
    """Situação da meta, chance de sucesso e risco."""
    meta = resultado.meta
    caminho = (
        f"ir de {formatar_reais(meta.valor_atual)} a {formatar_reais(meta.valor_alvo)} "
        f"em {formatar_prazo(meta.prazo_meses)}"
    )
    necessario = formatar_percentual(resultado.retorno_necessario_aa)
    frases = []
    if meta.reserva_emergencia:
        frases.append(
            "É a reserva de emergência: só usa produtos de liquidez diária e risco mínimo, "
            "para o dinheiro estar disponível a qualquer momento."
        )
    if resultado.situacao == ATINGIVEL:
        frases.append(
            f"Meta atingível: a carteira rende em média {formatar_percentual(resultado.retorno_esperado_aa)} "
            f"ao ano, já descontado o IR, acima dos {necessario} ao ano necessários para {caminho}."
        )
    elif resultado.situacao == DIFICIL:
        frases.append(
            f"Meta difícil: para {caminho} seriam necessários {necessario} ao ano. A carteira de maior "
            f"retorno dentro do risco aceito pelo perfil {nome_perfil} rende em média "
            f"{formatar_percentual(resultado.retorno_esperado_aa)} ao ano, já descontado o IR."
        )
    else:
        frases.append("Nenhum produto da prateleira serve para esta meta. Veja abaixo os motivos.")
        return frases

    chance = f"Chance de atingir a meta: {formatar_probabilidade(resultado.probabilidade_sucesso)} dos cenários simulados."
    if resultado.situacao == DIFICIL:
        chance += " Para aumentar a chance, dá para alongar o prazo, reduzir o valor-alvo ou fazer aportes."
    frases.append(chance)
    frases.append(_frase_de_risco(resultado, premissas))
    return frases


def _frase_de_risco(resultado: ResultadoMeta, premissas: Premissas) -> str:
    """Perda média nos piores cenários (CVaR), em reais e em porcentagem."""
    piores = formatar_percentual(1 - premissas.otimizacao.nivel_cvar, 0)
    if resultado.cvar > 0:
        perda = formatar_reais(resultado.cvar * resultado.meta.valor_atual)
        return (
            f"Nos {piores} piores cenários, a perda média seria de {perda} "
            f"({formatar_percentual(resultado.cvar)} do valor aplicado) no fim do prazo."
        )
    atual = resultado.meta.valor_atual
    return (
        f"Mesmo nos {piores} piores cenários não há perda: os {formatar_reais(atual)} aplicados "
        f"viram, em média, {formatar_reais(atual * (1 - resultado.cvar))} no fim do prazo."
    )


def _frases_do_produto(
    alocacao: Alocacao,
    meta: Meta,
    premissas: Premissas,
    uso_fgc_acumulado: UsoFGC,
    setor_trabalho: str | None = None,
) -> list[str]:
    """Por que e como cada produto entra na carteira."""
    produto = alocacao.produto
    otimizacao = premissas.otimizacao
    limite_fgc = premissas.fgc.limite_por_conglomerado
    frases = [
        f"{formatar_reais(alocacao.valor)} ({formatar_percentual(alocacao.peso, 1)} da meta). "
        f"Rendimento médio esperado de {formatar_percentual(alocacao.retorno_liquido_esperado_aa)} "
        "ao ano, já descontados IR e taxas."
    ]

    if produto.tributacao == "isento" and produto.indexador == "cdi":
        prazo = min(produto.vencimento_meses or meta.prazo_meses, meta.prazo_meses)
        equivalente = percentual_cdi_equivalente(produto.taxa, prazo, premissas)
        frases.append(
            f"É isento de IR: {formatar_percentual_cdi(produto.taxa)} do CDI isento equivale a um CDB "
            f"de {formatar_percentual_cdi(equivalente)} do CDI no mesmo prazo."
        )
    elif produto.tributacao == "isento":
        frases.append("É isento de IR.")

    if produto.coberto_fgc:
        if uso_fgc_acumulado.get(produto.conglomerado, 0.0) >= limite_fgc * (1 - TOLERANCIA_LIMITE):
            frases.append(
                f"Chegou ao limite do FGC no {produto.conglomerado} ({formatar_reais(limite_fgc)}, "
                "contando os juros até o vencimento); por isso não foi colocado mais dinheiro nele."
            )
        else:
            frases.append(f"Coberto pelo FGC até {formatar_reais(limite_fgc)} por conglomerado ({produto.conglomerado}).")
    else:
        frases.append(f"Não tem garantia do FGC: o risco de crédito é do emissor ({produto.emissor}).")

    vencimento = produto.vencimento_meses
    if vencimento is not None and vencimento < meta.prazo_meses:
        frases.append(
            f"Vence em {formatar_prazo(vencimento)}, antes do fim da meta; a simulação supõe que o "
            "dinheiro é reaplicado a 100% do CDI até lá."
        )
    elif vencimento is not None and vencimento > meta.prazo_meses:
        frases.append("Vence depois da meta, mas tem liquidez diária: dá para resgatar no fim da meta.")
    elif vencimento == meta.prazo_meses:
        frases.append("Vence junto com a meta.")

    if produto.carencia_meses > 0:
        frases.append(f"Tem carência de {formatar_prazo(produto.carencia_meses)}: o dinheiro fica preso nesse período.")

    if produto.acao_individual:
        frases.append(
            f"É uma ação individual ({produto.ticker}, setor {NOMES_DOS_SETORES[produto.setor]}), da lista "
            "aprovada pela área de análise da corretora. Oscila mais que um fundo ou ETF, porque depende "
            "de uma empresa só."
        )

    if produto.renda_variavel:
        frases.append(
            "É renda variável: o valor oscila e pode cair no caminho, por isso só entra em metas de "
            f"{formatar_prazo(otimizacao.prazo_minimo_renda_variavel_meses)} ou mais."
        )

    if produto.tributacao in TRIBUTACOES_COM_COME_COTAS:
        meses = " e ".join(NOMES_DOS_MESES[m - 1] for m in premissas.impostos.meses_come_cotas)
        frases.append(
            f"É um fundo com come-cotas: parte do IR é cobrada antes, em {meses}, o que reduz um pouco o rendimento."
        )

    if produto.taxa_adm_aa > 0:
        frases.append(f"Cobra taxa de administração de {formatar_percentual(produto.taxa_adm_aa)} ao ano (já descontada).")

    if setor_trabalho is not None and produto.setor == setor_trabalho:
        frases.append(
            f"É do mesmo setor em que o cliente trabalha ({NOMES_DOS_SETORES[produto.setor]}); "
            f"produtos desse setor somam no máximo {formatar_percentual(otimizacao.limite_setor_do_trabalho, 0)} "
            "da meta, para não depender duas vezes do mesmo setor."
        )

    limite = limite_do_produto(produto, premissas)
    if limite < 1 and alocacao.peso >= limite - TOLERANCIA_LIMITE:
        unidade = "ação" if produto.acao_individual else "produto"
        frases.append(
            f"Ficou no limite de {formatar_percentual(limite, 0)} por {unidade}, para não concentrar demais."
        )
    return frases


def _frases_de_trabalho(recomendacao: Recomendacao, premissas: Premissas) -> list[str]:
    """Explica como o emprego do cliente muda a carteira."""
    cliente = recomendacao.cliente
    frases = []
    if cliente.empregador is not None:
        frases.append(
            f"Como o cliente trabalha no {cliente.empregador}, nenhum produto desse grupo entra na carteira: "
            "se o grupo tiver problemas, o emprego e os investimentos não são atingidos juntos."
        )
    if cliente.setor_trabalho is not None:
        frases.append(
            f"Como o cliente trabalha no setor {NOMES_DOS_SETORES[cliente.setor_trabalho]}, os produtos desse setor "
            f"ficam em no máximo {formatar_percentual(premissas.otimizacao.limite_setor_do_trabalho, 0)} de cada meta."
        )
    return frases


def _nao_usados(resultado: ResultadoMeta) -> list[str]:
    """Agrupa os produtos excluídos pelo motivo: "Motivo: produto A, produto B."."""
    por_motivo: dict[str, list[str]] = {}
    for excluido in resultado.excluidos:
        por_motivo.setdefault(excluido.motivo, []).append(excluido.produto.nome)
    return [f"{motivo[0].upper()}{motivo[1:]}: {', '.join(nomes)}." for motivo, nomes in por_motivo.items()]
