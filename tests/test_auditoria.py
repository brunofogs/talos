"""Testes de `auditoria.py`: o registro JSON guarda tudo e não sobrescreve registros anteriores."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from talos import AVISO_SIMULACAO, __version__
from talos.auditoria import impressao_digital, ler_registro, registrar
from talos.explicacao import explicar
from talos.mercado import CurvaDI, carregar_curva_di
from talos.modelos import Premissas, Produto, carregar_cliente, carregar_prateleira, carregar_premissas
from talos.otimizador import Recomendacao, otimizar_cliente

MOMENTO = datetime(2026, 9, 27, 14, 30, 5, tzinfo=timezone(timedelta(hours=-3)))


@pytest.fixture(scope="module")
def dados() -> tuple[Recomendacao, list[Produto], CurvaDI, Premissas]:
    """Recomendação do cliente de exemplo, com as entradas usadas."""
    exemplos = Path(__file__).resolve().parent.parent / "exemplos"
    premissas = carregar_premissas(exemplos / "premissas.json")
    curva = carregar_curva_di(exemplos / "curva_di.csv", premissas)
    produtos = carregar_prateleira(exemplos / "prateleira.csv")
    recomendacao = otimizar_cliente(carregar_cliente(exemplos / "cliente.json"), produtos, premissas, curva, quantidade=500)
    return recomendacao, produtos, curva, premissas


def gravar(dados, pasta: Path, agora: datetime = MOMENTO) -> Path:
    """Grava o registro da recomendação de exemplo."""
    recomendacao, produtos, curva, premissas = dados
    return registrar(recomendacao, explicar(recomendacao, premissas), produtos, curva, premissas, pasta, agora)


def test_registro_guarda_tudo(dados, tmp_path: Path) -> None:
    """O JSON tem data, versão, semente, entradas, premissas, resultado, explicações e aviso."""
    registro = ler_registro(gravar(dados, tmp_path))
    assert registro["data_hora"] == "2026-09-27T14:30:05-03:00"
    assert registro["versao_talos"] == __version__
    assert registro["semente"] == 42
    assert registro["reamostragem"] is False
    assert registro["entradas"]["cliente"]["identificador"] == "cliente-exemplo-001"
    assert len(registro["entradas"]["prateleira"]) == 15
    assert registro["entradas"]["curva_di"]["prazos_du"][0] == 21
    assert registro["premissas"]["fgc"]["limite_por_conglomerado"] == 250_000
    assert registro["premissas"]["data_referencia"] == "2026-09-25"
    assert len(registro["resultado"]["metas"]) == 4
    assert registro["resultado"]["metas"][0]["alocacoes"][0]["produto"]["nome"]
    assert len(registro["explicacoes"]["metas"]) == 4
    assert registro["aviso"] == AVISO_SIMULACAO


def test_registro_em_utf8_com_acentos(dados, tmp_path: Path) -> None:
    """Os acentos ficam legíveis no arquivo (e não como \\u00ea)."""
    texto = gravar(dados, tmp_path).read_text(encoding="utf-8")
    assert "Reserva de emergência" in texto


def test_nome_do_arquivo_e_sem_sobrescrever(dados, tmp_path: Path) -> None:
    """O nome leva cliente e horário; um segundo registro no mesmo segundo ganha sufixo."""
    primeiro = gravar(dados, tmp_path)
    segundo = gravar(dados, tmp_path)
    assert primeiro.name == "recomendacao_cliente-exemplo-001_20260927-143005.json"
    assert segundo.name == "recomendacao_cliente-exemplo-001_20260927-143005_2.json"


def test_identificador_com_caracteres_proibidos(dados, tmp_path: Path) -> None:
    """Caracteres que o Windows não aceita em nomes de arquivo viram "_"."""
    recomendacao, produtos, curva, premissas = dados
    estranho = replace(recomendacao, cliente=replace(recomendacao.cliente, identificador='a/b\\c:d*e?"f'))
    caminho = registrar(estranho, explicar(estranho, premissas), produtos, curva, premissas, tmp_path, MOMENTO)
    assert caminho.name == "recomendacao_a_b_c_d_e_f_20260927-143005.json"


def test_cria_a_pasta_se_nao_existir(dados, tmp_path: Path) -> None:
    """A pasta de saída é criada automaticamente."""
    caminho = gravar(dados, tmp_path / "nova" / "pasta")
    assert caminho.exists()


def test_impressao_digital_muda_com_as_entradas(dados) -> None:
    """Mesmas entradas, mesma impressão digital; qualquer mudança gera outra."""
    recomendacao, produtos, curva, premissas = dados
    entradas = {"cliente": recomendacao.cliente, "prateleira": produtos, "curva_di": curva}
    original = impressao_digital(entradas, premissas)
    assert impressao_digital(dict(entradas), premissas) == original
    assert len(original) == 64

    outro_cliente = dict(entradas, cliente=replace(recomendacao.cliente, renda_mensal=4501.0))
    assert impressao_digital(outro_cliente, premissas) != original
    outra_semente = replace(premissas, cenarios=replace(premissas.cenarios, semente=43))
    assert impressao_digital(entradas, outra_semente) != original


def test_impressao_digital_no_registro_confere(dados, tmp_path: Path) -> None:
    """A impressão digital gravada é a das entradas usadas."""
    recomendacao, produtos, curva, premissas = dados
    registro = ler_registro(gravar(dados, tmp_path))
    entradas = {"cliente": recomendacao.cliente, "prateleira": produtos, "curva_di": curva}
    assert registro["impressao_digital_entradas"] == impressao_digital(entradas, premissas)
