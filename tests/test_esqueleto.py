"""Testes da etapa 1: o pacote importa, as dependências funcionam e a CLI responde."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import talos

RAIZ = Path(__file__).resolve().parent.parent


def test_versao_definida() -> None:
    """O pacote expõe uma versão no formato X.Y.Z."""
    assert len(talos.__version__.split(".")) == 3


def test_linprog_highs_disponivel() -> None:
    """O solver HiGHS do SciPy resolve um problema linear feito à mão.

    Minimizar x + 2y com x + y = 1, x, y >= 0: a resposta é x = 1, y = 0.
    """
    resultado = linprog(
        c=[1, 2], A_eq=[[1, 1]], b_eq=[1], bounds=[(0, None)] * 2, method="highs"
    )
    assert resultado.success
    assert np.allclose(resultado.x, [1.0, 0.0])


def test_cli_mostra_a_versao() -> None:
    """`python -m talos --versao` roda como programa separado e mostra a versão."""
    ambiente = {**os.environ, "PYTHONPATH": str(RAIZ / "src")}
    processo = subprocess.run(
        [sys.executable, "-m", "talos", "--versao"],
        capture_output=True, text=True, encoding="utf-8", env=ambiente, cwd=RAIZ,
    )
    assert processo.returncode == 0
    assert processo.stdout.strip() == f"talos {talos.__version__}"
