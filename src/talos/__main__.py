"""Linha de comando do Talos: `python -m talos <comando>`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from talos import AVISO_SIMULACAO, __version__


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
    parser.add_argument(
        "--versao", action="version", version=f"talos {__version__}"
    )
    comandos = parser.add_subparsers(dest="comando", required=True)
    for nome, ajuda in (
        ("recomendar", "gera uma carteira para cada meta do cliente"),
        ("validar", "compara a carteira com referências simples"),
    ):
        sub = comandos.add_parser(nome, help=ajuda)
        sub.add_argument("--cliente", required=True, help="arquivo JSON do cliente")
        sub.add_argument("--prateleira", required=True, help="arquivo CSV de produtos")
    return parser


def main(argumentos: Sequence[str] | None = None) -> int:
    """Executa a linha de comando e devolve o código de saída (0 = sucesso)."""
    usar_utf8_no_terminal()
    args = criar_parser().parse_args(argumentos)
    print(f"O comando '{args.comando}' ainda não foi implementado (etapa 7).")
    print()
    print(AVISO_SIMULACAO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
