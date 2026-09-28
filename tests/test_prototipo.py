"""Testes da ponte do protótipo web: a página recebe do motor o mesmo resultado do Python."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ponte():
    """Carrega `prototipo/ponte.py` como módulo."""
    especificacao = importlib.util.spec_from_file_location("ponte", RAIZ / "prototipo" / "ponte.py")
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


def test_dados_da_prateleira(ponte) -> None:
    """A página recebe os 15 produtos, os perfis e o cliente de exemplo (sem o campo de aviso)."""
    dados = json.loads(ponte.dados_da_prateleira(RAIZ / "exemplos"))
    assert len(dados["produtos"]) == 15
    assert set(dados["perfis"]) == {"conservador", "moderado", "arrojado"}
    assert dados["cdi_12_meses"] == pytest.approx(0.14)
    assert "_aviso" not in dados["cliente_exemplo"]


def test_simular_cliente_de_exemplo(ponte) -> None:
    """A simulação devolve as 4 metas, com faixa de valor final em ordem e o aviso no fim."""
    cliente = json.loads((RAIZ / "exemplos" / "cliente.json").read_text(encoding="utf-8"))
    cliente["quantidade"] = 500
    resposta = json.loads(ponte.simular(json.dumps(cliente), RAIZ / "exemplos"))
    assert [m["nome"] for m in resposta["metas"]][0] == "Reserva de emergência"
    for meta in resposta["metas"]:
        faixa = meta["faixa_valor_final"]
        assert faixa["p5"] <= faixa["p25"] <= faixa["p50"] <= faixa["p75"] <= faixa["p95"]
        assert sum(a["peso"] for a in meta["alocacoes"]) == pytest.approx(1.0)
        assert all(a["frases"] for a in meta["alocacoes"])
    assert resposta["aviso"].startswith("Aviso:")


@pytest.mark.parametrize(
    ("mudanca", "trecho"),
    [({"perfil": "agressivo"}, "perfil do cliente"), ({"metas": []}, "pelo menos uma meta")],
)
def test_erros_voltam_como_mensagem(ponte, mudanca: dict, trecho: str) -> None:
    """Dados inválidos voltam como {"erro": ...} em português, sem quebrar a página."""
    cliente = json.loads((RAIZ / "exemplos" / "cliente.json").read_text(encoding="utf-8"))
    cliente.update(mudanca)
    resposta = json.loads(ponte.simular(json.dumps(cliente), RAIZ / "exemplos"))
    assert trecho in resposta["erro"]
