"""Limites de garantia do FGC (Fundo Garantidor de Créditos).

Duas regras, com valores lidos de `premissas.json`:
- limite por conglomerado financeiro (R$ 250 mil na premissa de exemplo),
  somando todos os bancos do mesmo grupo;
- teto global por CPF (R$ 1 milhão a cada 4 anos). Como não sabemos o
  histórico do cliente, o Talos trata o teto de forma conservadora: o total
  aplicado em produtos cobertos não passa dele.

Como o otimizador trabalha uma meta por vez, o "uso" do FGC já comprometido
pelas metas anteriores é passado adiante como um dicionário
conglomerado -> valor coberto.

A garantia cobre principal mais juros, por isso as restrições aceitam um
fator de crescimento por produto: aplicar R$ 200 mil que viram R$ 260 mil
estoura o limite no vencimento.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from talos.modelos import ErroDeDados, PremissasFGC, Produto

UsoFGC = dict[str, float]


@dataclass(frozen=True)
class RestricaoLinear:
    """Restrição no formato `soma(coeficientes * pesos) <= limite`, pronta para o linprog."""

    coeficientes: npt.NDArray[np.float64]
    limite: float
    descricao: str


def formatar_reais(valor: float) -> str:
    """Escreve um valor em reais no padrão brasileiro: R$ 1.234,56."""
    texto = f"{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"R$ {texto}"


def uso_por_conglomerado(aplicacoes: Iterable[tuple[Produto, float]]) -> UsoFGC:
    """Soma, por conglomerado, os valores aplicados em produtos cobertos pelo FGC."""
    uso: UsoFGC = {}
    for produto, valor in aplicacoes:
        if produto.coberto_fgc:
            uso[produto.conglomerado] = uso.get(produto.conglomerado, 0.0) + valor
    return uso


def somar_usos(*usos: Mapping[str, float]) -> UsoFGC:
    """Junta usos do FGC de várias metas num só."""
    total: UsoFGC = {}
    for uso in usos:
        for conglomerado, valor in uso.items():
            total[conglomerado] = total.get(conglomerado, 0.0) + valor
    return total


def folga_no_conglomerado(conglomerado: str, uso: Mapping[str, float], fgc: PremissasFGC) -> float:
    """Quanto ainda cabe no limite do conglomerado (nunca negativo)."""
    return max(fgc.limite_por_conglomerado - uso.get(conglomerado, 0.0), 0.0)


def folga_global(uso: Mapping[str, float], fgc: PremissasFGC) -> float:
    """Quanto ainda cabe no teto global por CPF (nunca negativo)."""
    return max(fgc.teto_global - sum(uso.values()), 0.0)


def restricoes_fgc(
    produtos: Sequence[Produto],
    valor_investido: float,
    uso_previo: Mapping[str, float],
    fgc: PremissasFGC,
    fatores_crescimento: Sequence[float] | None = None,
) -> list[RestricaoLinear]:
    """Monta as restrições do FGC sobre os pesos de uma carteira.

    O peso `w[i]` é a fração de `valor_investido` no produto `i`. Para cada
    conglomerado com produtos cobertos, e para o teto global, vale:
    `soma(valor_investido * crescimento[i] * w[i]) <= folga`.
    Sem `fatores_crescimento`, o limite vale sobre o valor aplicado.
    """
    if valor_investido <= 0:
        raise ErroDeDados("fgc: o valor investido precisa ser maior que zero.")
    crescimento = np.ones(len(produtos)) if fatores_crescimento is None else np.asarray(fatores_crescimento, dtype=float)
    if crescimento.shape != (len(produtos),):
        raise ErroDeDados("fgc: informe um fator de crescimento para cada produto.")

    cobertos = np.array([produto.coberto_fgc for produto in produtos], dtype=bool)
    exposicao = np.where(cobertos, valor_investido * crescimento, 0.0)

    restricoes = []
    for conglomerado in sorted({p.conglomerado for p in produtos if p.coberto_fgc}):
        do_grupo = np.array([p.conglomerado == conglomerado for p in produtos], dtype=bool)
        restricoes.append(RestricaoLinear(
            coeficientes=np.where(do_grupo, exposicao, 0.0),
            limite=folga_no_conglomerado(conglomerado, uso_previo, fgc),
            descricao=f"FGC: limite do {conglomerado}",
        ))
    if cobertos.any():
        restricoes.append(RestricaoLinear(
            coeficientes=exposicao,
            limite=folga_global(uso_previo, fgc),
            descricao="FGC: teto global por CPF",
        ))
    return restricoes


def violacoes_fgc(
    aplicacoes: Iterable[tuple[Produto, float]],
    fgc: PremissasFGC,
    uso_previo: Mapping[str, float] | None = None,
) -> list[str]:
    """Lista, em português, os limites do FGC estourados por uma carteira em reais.

    Devolve uma lista vazia quando tudo está dentro dos limites.
    """
    uso = somar_usos(uso_previo or {}, uso_por_conglomerado(aplicacoes))
    mensagens = [
        f"{conglomerado}: {formatar_reais(valor)} em produtos cobertos, acima do limite "
        f"de {formatar_reais(fgc.limite_por_conglomerado)} por conglomerado."
        for conglomerado, valor in sorted(uso.items())
        if valor > fgc.limite_por_conglomerado
    ]
    total = sum(uso.values())
    if total > fgc.teto_global:
        mensagens.append(
            f"Total de {formatar_reais(total)} em produtos cobertos, acima do teto global "
            f"de {formatar_reais(fgc.teto_global)} por CPF."
        )
    return mensagens
