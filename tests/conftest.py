"""Caminhos compartilhados pelos testes."""

from __future__ import annotations

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
PASTA_EXEMPLOS = RAIZ / "exemplos"


@pytest.fixture
def exemplos() -> Path:
    """Pasta com os arquivos de exemplo do projeto."""
    return PASTA_EXEMPLOS
