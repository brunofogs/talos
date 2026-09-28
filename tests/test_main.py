"""Testes da linha de comando (`python -m talos`), de ponta a ponta com os arquivos de exemplo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from talos import AVISO_SIMULACAO
from talos.__main__ import main
from talos.auditoria import ler_registro

RAIZ = Path(__file__).resolve().parent.parent
EXEMPLOS = RAIZ / "exemplos"
POUCOS_CENARIOS = "500"


def argumentos_de_exemplo(pasta_saidas: Path, *extras: str) -> list[str]:
    """Argumentos de `recomendar` com o cliente e a prateleira de exemplo."""
    return [
        "recomendar",
        "--cliente", str(EXEMPLOS / "cliente.json"),
        "--prateleira", str(EXEMPLOS / "prateleira.csv"),
        "--cenarios", POUCOS_CENARIOS,
        "--saidas", str(pasta_saidas),
        *extras,
    ]


def test_recomendar_de_ponta_a_ponta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """O relatório traz as 4 metas, a carteira, as explicações, o registro e termina com o aviso."""
    assert main(argumentos_de_exemplo(tmp_path)) == 0
    saida = capsys.readouterr().out
    assert saida.startswith("TALOS · Recomendação para cliente-exemplo-001 (perfil moderado)")
    assert "Premissas de 25/09/2026 · 500 cenários · semente 42" in saida
    for nome in ("Reserva de emergência", "Entrada do apartamento", "Aposentadoria", "Viagem"):
        assert nome in saida
    assert "Meta 1 de 4: Reserva de emergência" in saida
    assert "ATINGÍVEL" in saida and "DIFÍCIL" in saida
    assert "Tesouro Selic 2031" in saida
    assert "Valor no fim do prazo:" in saida
    assert saida.strip().endswith(AVISO_SIMULACAO.split()[-1])
    assert " ".join(saida.split()).endswith(AVISO_SIMULACAO)


def test_recomendar_grava_a_auditoria(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Um registro JSON é gravado na pasta pedida, e o caminho aparece no relatório."""
    main(argumentos_de_exemplo(tmp_path, "--semente", "7"))
    arquivos = list(tmp_path.glob("recomendacao_cliente-exemplo-001_*.json"))
    assert len(arquivos) == 1
    assert str(arquivos[0]) in capsys.readouterr().out
    registro = ler_registro(arquivos[0])
    assert registro["semente"] == 7
    assert len(registro["resultado"]["metas"]) == 4


def test_linhas_nao_passam_da_largura(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Só a linha com o caminho do registro pode passar de 88 caracteres."""
    main(argumentos_de_exemplo(tmp_path))
    linhas = capsys.readouterr().out.splitlines()
    assert all(len(linha) <= 88 for linha in linhas if not linha.startswith("Registro de auditoria:"))


def test_estrategia_crescimento_com_acoes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Com a lista de ações da pasta de exemplos e --estrategia crescimento, ações aparecem no relatório."""
    dados = json.loads((EXEMPLOS / "cliente.json").read_text(encoding="utf-8"))
    dados["perfil"] = "arrojado"
    cliente = tmp_path / "cliente.json"
    cliente.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    argumentos = argumentos_de_exemplo(tmp_path, "--estrategia", "crescimento")
    argumentos[2] = str(cliente)
    assert main(argumentos) == 0
    saida = capsys.readouterr().out
    assert "estratégia crescimento" in saida
    assert "-F (" in saida


def test_lista_de_acoes_inexistente(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--acoes apontando para um arquivo que não existe gera mensagem clara e código 2."""
    assert main(argumentos_de_exemplo(tmp_path, "--acoes", str(tmp_path / "nao_existe.csv"))) == 2
    assert "Arquivo não encontrado" in capsys.readouterr().err


def test_arquivo_inexistente(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Cliente que não existe: mensagem clara e código de saída 2."""
    argumentos = argumentos_de_exemplo(tmp_path)
    argumentos[2] = str(tmp_path / "nao_existe.json")
    assert main(argumentos) == 2
    assert "Arquivo não encontrado:" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.json"))


def test_dados_invalidos(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Cliente com perfil desconhecido: mensagem em português e código de saída 2."""
    dados = json.loads((EXEMPLOS / "cliente.json").read_text(encoding="utf-8"))
    dados["perfil"] = "agressivo"
    cliente = tmp_path / "cliente.json"
    cliente.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    argumentos = argumentos_de_exemplo(tmp_path)
    argumentos[2] = str(cliente)
    assert main(argumentos) == 2
    assert capsys.readouterr().err.startswith("Erro nos dados: perfil do cliente: valor 'agressivo'")


def test_cenarios_invalidos(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--cenarios 0 é recusado com mensagem clara."""
    argumentos = argumentos_de_exemplo(tmp_path)
    argumentos[argumentos.index(POUCOS_CENARIOS)] = "0"
    assert main(argumentos) == 2
    assert "--cenarios" in capsys.readouterr().err


def test_premissas_e_curva_em_outra_pasta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Com a prateleira em outra pasta, --premissas e --curva apontam os arquivos."""
    prateleira = tmp_path / "prateleira.csv"
    prateleira.write_text((EXEMPLOS / "prateleira.csv").read_text(encoding="utf-8"), encoding="utf-8")
    argumentos = argumentos_de_exemplo(tmp_path)
    argumentos[4] = str(prateleira)
    assert main(argumentos) == 2
    assert "premissas.json" in capsys.readouterr().err
    argumentos += ["--premissas", str(EXEMPLOS / "premissas.json"), "--curva", str(EXEMPLOS / "curva_di.csv")]
    assert main(argumentos) == 0


def test_como_programa_separado(tmp_path: Path) -> None:
    """`python -m talos recomendar ...` roda num processo separado e escreve em UTF-8."""
    ambiente = {**os.environ, "PYTHONPATH": str(RAIZ / "src")}
    ambiente.pop("PYTHONIOENCODING", None)
    processo = subprocess.run(
        [sys.executable, "-m", "talos", *argumentos_de_exemplo(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", env=ambiente, cwd=RAIZ,
    )
    assert processo.returncode == 0, processo.stderr
    assert "Reserva de emergência" in processo.stdout
    assert " ".join(processo.stdout.split()).endswith(AVISO_SIMULACAO)


# --- validar -----------------------------------------------------------------------------------

def test_validar_de_ponta_a_ponta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`validar` mostra as três carteiras por meta, a conclusão, o resumo e o aviso; não grava auditoria."""
    argumentos = argumentos_de_exemplo(tmp_path)
    argumentos[0] = "validar"
    assert main(argumentos) == 0
    saida = capsys.readouterr().out
    assert saida.startswith("TALOS · Validação para cliente-exemplo-001 (perfil moderado)")
    assert saida.count("  Talos ") == 4
    assert saida.count("  100% do CDI ") == 4
    assert saida.count("  Pesos iguais ") == 4
    assert "Resumo: em " in saida
    assert " ".join(saida.split()).endswith(AVISO_SIMULACAO)
    assert all(len(linha) <= 88 for linha in saida.splitlines())
    assert not list(tmp_path.glob("*.json"))
