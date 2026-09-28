"""Validação: a carteira do Talos contra duas referências simples, nos mesmos cenários.

Referências:
- **100% do CDI**: um CDB comum que rende o CDI, com IR regressivo, até o fim da meta;
- **Pesos iguais**: o mesmo valor em cada produto que serve para a meta (mesmo
  filtro do otimizador), sem otimização nenhuma.

Os cenários são gerados com a mesma semente e quantidade do otimizador, então
as três carteiras enfrentam exatamente os mesmos futuros.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from talos.explicacao import formatar_percentual, formatar_probabilidade
from talos.impostos import MESES_POR_ANO
from talos.mercado import CurvaDI, gerar_cenarios
from talos.modelos import Meta, Premissas, Produto
from talos.otimizador import Recomendacao, ResultadoMeta, cvar, fatores_liquidos, filtrar_produtos

NOME_TALOS = "Talos"
NOME_CDI = "100% do CDI"
NOME_PESOS_IGUAIS = "Pesos iguais"

# Referência 100% do CDI: um CDB sem vencimento dentro da meta, com IR regressivo.
PRODUTO_CDI = Produto(
    nome=NOME_CDI, emissor="Referência", conglomerado="Referência", indexador="cdi",
    taxa=1.0, taxa_adm_aa=0.0, vencimento_meses=None, carencia_meses=0, liquidez_diaria=True,
    tributacao="regressivo", classe="pos_fixado", risco=1, coberto_fgc=False,
    aplicacao_minima=0.0, setor="financeiro",
)


@dataclass(frozen=True)
class Estatisticas:
    """Números de uma carteira nos cenários de uma meta.

    `resultado_piores` é a variação média do valor aplicado nos piores
    cenários (o contrário do CVaR): -0,08 é perda de 8%, +0,48 é ganho de 48%.
    """

    carteira: str
    retorno_esperado_aa: float
    resultado_piores: float
    probabilidade_sucesso: float


@dataclass(frozen=True)
class ComparacaoMeta:
    """As três carteiras de uma meta, lado a lado. Carteiras impossíveis ficam como `None`."""

    meta: Meta
    situacao: str
    talos: Estatisticas | None
    cdi: Estatisticas
    pesos_iguais: Estatisticas | None

    @property
    def linhas(self) -> tuple[Estatisticas, ...]:
        """As carteiras que existem, na ordem Talos, 100% do CDI, pesos iguais."""
        return tuple(e for e in (self.talos, self.cdi, self.pesos_iguais) if e is not None)


def estatisticas(
    carteira: str, patrimonio: npt.ArrayLike, meta: Meta, nivel_cvar: float
) -> Estatisticas:
    """Calcula retorno esperado, resultado nos piores cenários e chance de sucesso.

    `patrimonio` diz quanto R$ 1 vira no fim da meta em cada cenário.
    Exemplo: [1,2; 1,1; 0,9; 1,3], alvo 1,15 em 1 ano e nível 75%: retorno
    esperado 12,5%, pior cenário -10% e chance de 50%.
    """
    patrimonio = np.asarray(patrimonio, dtype=float)
    anos = meta.prazo_meses / MESES_POR_ANO
    return Estatisticas(
        carteira=carteira,
        retorno_esperado_aa=float(np.mean(patrimonio) ** (1 / anos) - 1),
        resultado_piores=-cvar(1 - patrimonio, nivel_cvar),
        probabilidade_sucesso=float(np.mean(patrimonio >= meta.valor_alvo / meta.valor_atual)),
    )


def validar(
    recomendacao: Recomendacao,
    produtos: Sequence[Produto],
    premissas: Premissas,
    curva: CurvaDI,
    quantidade: int | None = None,
) -> tuple[ComparacaoMeta, ...]:
    """Compara cada meta da recomendação com 100% do CDI e com pesos iguais.

    `quantidade` precisa ser a mesma usada no otimizador para os cenários
    serem idênticos.
    """
    return tuple(
        comparar_meta(resultado, recomendacao, produtos, premissas, curva, quantidade)
        for resultado in recomendacao.resultados
    )


def comparar_meta(
    resultado: ResultadoMeta,
    recomendacao: Recomendacao,
    produtos: Sequence[Produto],
    premissas: Premissas,
    curva: CurvaDI,
    quantidade: int | None = None,
) -> ComparacaoMeta:
    """Monta a comparação de uma meta."""
    meta = resultado.meta
    nivel = premissas.otimizacao.nivel_cvar
    cenarios = gerar_cenarios(premissas, curva, meta.prazo_meses, resultado.semente, quantidade)

    talos = None
    if resultado.alocacoes:
        pesos = np.array([a.peso for a in resultado.alocacoes])
        fatores = fatores_liquidos([a.produto for a in resultado.alocacoes], cenarios, premissas)
        talos = estatisticas(NOME_TALOS, fatores @ pesos, meta, nivel)

    cdi = estatisticas(NOME_CDI, fatores_liquidos([PRODUTO_CDI], cenarios, premissas)[:, 0], meta, nivel)

    elegiveis, _ = filtrar_produtos(
        produtos, meta, recomendacao.perfil, premissas, recomendacao.cliente.empregador
    )
    pesos_iguais = None
    if elegiveis:
        fatores = fatores_liquidos(elegiveis, cenarios, premissas)
        pesos_iguais = estatisticas(NOME_PESOS_IGUAIS, fatores.mean(axis=1), meta, nivel)

    return ComparacaoMeta(meta, resultado.situacao, talos, cdi, pesos_iguais)


# ---------------------------------------------------------------------------
# Conclusões em português (por regras)
# ---------------------------------------------------------------------------

# Diferenças menores que estas contam como empate (0,1 ponto percentual de
# chance e 0,05 ponto percentual de retorno ou resultado).
EMPATE_PROBABILIDADE = 0.001
EMPATE_RETORNO = 0.0005


def conclusao(comparacao: ComparacaoMeta) -> str:
    """Compara o Talos com a melhor referência em cada quesito, numa frase."""
    if comparacao.talos is None:
        return (
            "O Talos não montou carteira para esta meta. Com 100% do CDI, a chance de atingi-la "
            f"seria de {formatar_probabilidade(comparacao.cdi.probabilidade_sucesso)}."
        )
    talos = comparacao.talos
    referencias = [e for e in (comparacao.cdi, comparacao.pesos_iguais) if e is not None]
    quesitos = (
        ("chance de sucesso", "probabilidade_sucesso", EMPATE_PROBABILIDADE, formatar_probabilidade),
        ("resultado nos 5% piores cenários", "resultado_piores", EMPATE_RETORNO, percentual_com_sinal),
        ("retorno esperado", "retorno_esperado_aa", EMPATE_RETORNO, formatar_percentual),
    )
    partes = []
    empates = 0
    for nome, campo, empate, formatar in quesitos:
        melhor = max(referencias, key=lambda e: getattr(e, campo))
        valor_talos, valor_ref = getattr(talos, campo), getattr(melhor, campo)
        if abs(valor_talos - valor_ref) <= empate:
            palavra = "empata em"
            empates += 1
        elif valor_talos > valor_ref:
            palavra = "é melhor em"
        else:
            palavra = "é pior em"
        partes.append(
            f"{palavra} {nome} ({formatar(valor_talos)} contra {formatar(valor_ref)} "
            f"de {_nome_na_frase(melhor.carteira)})"
        )
    if empates == len(quesitos):
        return (
            "O Talos empata com a melhor referência em chance de sucesso, resultado nos 5% piores "
            "cenários e retorno esperado: aqui as carteiras simples já resolvem bem."
        )
    return "Frente à melhor referência em cada quesito, o Talos " + "; ".join(partes) + "."


def percentual_com_sinal(valor: float) -> str:
    """Porcentagem com "+" na frente quando é ganho: 0,48 -> "+48,00%"."""
    texto = formatar_percentual(valor)
    return f"+{texto}" if valor >= 0 else texto


def _nome_na_frase(carteira: str) -> str:
    """Nome da carteira no meio da frase: "Pesos iguais" -> "pesos iguais"."""
    return carteira[0].lower() + carteira[1:]


def resumo_validacao(comparacoes: Sequence[ComparacaoMeta]) -> str:
    """Em quantas metas o Talos tem chance e resultado nos piores cenários iguais ou melhores
    que as duas referências (diferenças dentro da margem de empate contam como iguais).
    """
    com_talos = [c for c in comparacoes if c.talos is not None]

    def pelo_menos_igual(campo: str, empate: float) -> int:
        return sum(
            all(getattr(c.talos, campo) >= getattr(r, campo) - empate for r in c.linhas)
            for c in com_talos
        )

    total = len(comparacoes)
    return (
        f"Resumo: em {pelo_menos_igual('probabilidade_sucesso', EMPATE_PROBABILIDADE)} de {total} metas o "
        "Talos tem chance de sucesso igual ou maior que as duas referências, e em "
        f"{pelo_menos_igual('resultado_piores', EMPATE_RETORNO)} de {total} tem resultado igual ou melhor "
        "nos 5% piores cenários."
    )
