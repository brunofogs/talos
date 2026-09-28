"""Linha de comando do Talos: `python -m talos <comando>`.

    python -m talos recomendar --cliente exemplos/cliente.json --prateleira exemplos/prateleira.csv

Por padrão, as premissas e a curva de juros são lidas da mesma pasta da
prateleira (`premissas.json` e `curva_di.csv`), e o registro de auditoria
é gravado em `saidas/`.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from talos import AVISO_SIMULACAO, __version__
from talos.auditoria import PASTA_PADRAO, registrar
from talos.explicacao import (
    Explicacao,
    ExplicacaoMeta,
    explicar,
    formatar_percentual,
    formatar_prazo,
    formatar_probabilidade,
)
from talos.fgc import formatar_reais
from talos.mercado import carregar_curva_di
from talos.modelos import ErroDeDados, carregar_cliente, carregar_prateleira, carregar_premissas
from talos.otimizador import ATINGIVEL, DIFICIL, Recomendacao, ResultadoMeta, otimizar_cliente

LARGURA = 88
CODIGO_ERRO_DE_DADOS = 2
NOME_PREMISSAS = "premissas.json"
NOME_CURVA = "curva_di.csv"
ROTULOS_SITUACAO = {ATINGIVEL: "ATINGÍVEL", DIFICIL: "DIFÍCIL"}
ROTULO_SEM_SOLUCAO = "SEM SOLUÇÃO"


def usar_utf8_no_terminal() -> None:
    """Faz a saída do terminal usar UTF-8 em todos os sistemas.

    Sem isso, no Windows os acentos saem trocados quando a saída é
    redirecionada para um arquivo ou outro programa.
    """
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8", errors="replace")


def criar_parser() -> argparse.ArgumentParser:
    """Monta o leitor de argumentos com os comandos `recomendar` e `validar`."""
    parser = argparse.ArgumentParser(
        prog="talos",
        description="Talos: carteiras de investimento por meta (simulação).",
    )
    parser.add_argument("--versao", action="version", version=f"talos {__version__}")
    comandos = parser.add_subparsers(dest="comando", required=True)
    for nome, ajuda in (
        ("recomendar", "gera uma carteira para cada meta do cliente"),
        ("validar", "compara a carteira com referências simples"),
    ):
        sub = comandos.add_parser(nome, help=ajuda)
        sub.add_argument("--cliente", required=True, type=Path, help="arquivo JSON do cliente")
        sub.add_argument("--prateleira", required=True, type=Path, help="arquivo CSV de produtos")
        sub.add_argument("--premissas", type=Path, help=f"arquivo de premissas (padrão: {NOME_PREMISSAS} na pasta da prateleira)")
        sub.add_argument("--curva", type=Path, help=f"curva de DI (padrão: {NOME_CURVA} na pasta da prateleira)")
        sub.add_argument("--semente", type=int, help="semente dos cenários (padrão: a das premissas)")
        sub.add_argument("--cenarios", type=int, help="quantidade de cenários (padrão: a das premissas)")
        sub.add_argument("--reamostragem", action="store_true", help="usa a reamostragem de Michaud (mais lento)")
        sub.add_argument("--saidas", type=Path, default=PASTA_PADRAO, help="pasta do registro de auditoria (padrão: saidas)")
    return parser


def main(argumentos: Sequence[str] | None = None) -> int:
    """Executa a linha de comando e devolve o código de saída (0 = sucesso, 2 = erro nos dados)."""
    usar_utf8_no_terminal()
    args = criar_parser().parse_args(argumentos)
    try:
        if args.comando == "recomendar":
            print(recomendar(args))
        else:
            print(f"O comando '{args.comando}' ainda não foi implementado (etapa 8).\n\n{AVISO_SIMULACAO}")
    except FileNotFoundError as erro:
        print(f"Arquivo não encontrado: {erro.filename}", file=sys.stderr)
        return CODIGO_ERRO_DE_DADOS
    except ErroDeDados as erro:
        print(f"Erro nos dados: {erro}", file=sys.stderr)
        return CODIGO_ERRO_DE_DADOS
    return 0


def recomendar(args: argparse.Namespace) -> str:
    """Lê as entradas, otimiza, explica, grava a auditoria e devolve o relatório em texto."""
    pasta = args.prateleira.parent
    premissas = carregar_premissas(args.premissas or pasta / NOME_PREMISSAS)
    curva = carregar_curva_di(args.curva or pasta / NOME_CURVA, premissas)
    produtos = carregar_prateleira(args.prateleira)
    cliente = carregar_cliente(args.cliente)
    premissas.perfil(cliente.perfil)
    if args.cenarios is not None and args.cenarios < 1:
        raise ErroDeDados("--cenarios precisa ser de pelo menos 1.")

    recomendacao = otimizar_cliente(
        cliente, produtos, premissas, curva,
        reamostragem=args.reamostragem, semente=args.semente, quantidade=args.cenarios,
    )
    explicacao = explicar(recomendacao, premissas)
    caminho = registrar(recomendacao, explicacao, produtos, curva, premissas, args.saidas)
    quantidade = f"{args.cenarios or premissas.cenarios.quantidade:,}".replace(",", ".")
    cabecalho = (
        f"Premissas de {premissas.data_referencia:%d/%m/%Y} · {quantidade} cenários · "
        f"semente {recomendacao.semente}" + (" · com reamostragem" if recomendacao.reamostragem else "")
    )
    return formatar_relatorio(recomendacao, explicacao, cabecalho, caminho)


# ---------------------------------------------------------------------------
# Relatório em texto
# ---------------------------------------------------------------------------

def formatar_relatorio(
    recomendacao: Recomendacao, explicacao: Explicacao, cabecalho: str, caminho_auditoria: Path
) -> str:
    """Monta o relatório completo: cabeçalho, avisos gerais, uma seção por meta e o aviso final."""
    linhas = [
        f"TALOS · Recomendação para {recomendacao.cliente.identificador} (perfil {recomendacao.perfil.nome})",
        cabecalho,
    ]
    for frase in explicacao.antes_de_tudo:
        linhas += [""] + _paragrafo(frase, "» ")
    total = len(recomendacao.resultados)
    for numero, (resultado, texto) in enumerate(zip(recomendacao.resultados, explicacao.metas), start=1):
        linhas += [""] + _secao_meta(resultado, texto, numero, total)
    linhas += ["", "=" * LARGURA, f"Registro de auditoria: {caminho_auditoria}", ""]
    linhas += _paragrafo(explicacao.aviso)
    return "\n".join(linhas)


def _secao_meta(resultado: ResultadoMeta, texto: ExplicacaoMeta, numero: int, total: int) -> list[str]:
    """Seção de uma meta: título, números principais, tabela da carteira e explicações."""
    meta = resultado.meta
    rotulo = ROTULOS_SITUACAO.get(resultado.situacao, ROTULO_SEM_SOLUCAO)
    titulo = f"Meta {numero} de {total}: {meta.nome}"
    linhas = [
        "=" * LARGURA,
        f"{titulo}{rotulo:>{LARGURA - len(titulo)}}",
        f"{formatar_reais(meta.valor_atual)} → {formatar_reais(meta.valor_alvo)} em {formatar_prazo(meta.prazo_meses)}"
        + (" · reserva de emergência" if meta.reserva_emergencia else ""),
        "-" * LARGURA,
    ]
    if resultado.alocacoes:
        linhas += _paragrafo(
            f"Precisa render {formatar_percentual(resultado.retorno_necessario_aa)} a.a. · "
            f"carteira rende {formatar_percentual(resultado.retorno_esperado_aa)} a.a. · "
            f"chance de atingir {formatar_probabilidade(resultado.probabilidade_sucesso)}"
        )
        if resultado.valor_final_por_percentil:
            valores = resultado.valor_final_por_percentil
            linhas += _paragrafo(
                f"Valor no fim do prazo: 5% piores cenários até {formatar_reais(valores[5])} · "
                f"cenário do meio {formatar_reais(valores[50])} · "
                f"5% melhores acima de {formatar_reais(valores[95])}"
            )
        linhas += [""] + _tabela_carteira(resultado)
    linhas.append("")
    for frase in texto.resumo:
        linhas += _paragrafo(frase, "• ")
    for nome, frases in texto.produtos.items():
        linhas += ["", f"  {nome}"]
        for frase in frases[1:]:
            linhas += _paragrafo(frase, "    - ")
    if texto.nao_usados:
        linhas += ["", "  Não entraram nesta meta:"]
        for frase in texto.nao_usados:
            linhas += _paragrafo(frase, "    · ")
    return linhas


def _tabela_carteira(resultado: ResultadoMeta) -> list[str]:
    """Tabela com produto, peso, valor aplicado e rendimento líquido esperado."""
    largura_nome = max(len("Produto"), *(len(a.produto.nome) for a in resultado.alocacoes))
    linhas = [f"  {'Produto':<{largura_nome}}  {'Peso':>6}  {'Valor aplicado':>16}  {'Rende a.a.':>10}"]
    for alocacao in resultado.alocacoes:
        linhas.append(
            f"  {alocacao.produto.nome:<{largura_nome}}  {formatar_percentual(alocacao.peso, 1):>6}  "
            f"{formatar_reais(alocacao.valor):>16}  {formatar_percentual(alocacao.retorno_liquido_esperado_aa):>10}"
        )
    return linhas


def _paragrafo(texto: str, marcador: str = "") -> list[str]:
    """Quebra um texto longo em linhas, com o marcador na primeira e recuo nas seguintes."""
    return textwrap.wrap(
        texto, width=LARGURA, initial_indent=marcador, subsequent_indent=" " * len(marcador)
    )


if __name__ == "__main__":
    raise SystemExit(main())
