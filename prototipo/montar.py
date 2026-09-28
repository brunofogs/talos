"""Monta a pasta `prototipo/dist/` com tudo o que a página precisa.

Uso (em qualquer sistema, na raiz do projeto):

    python prototipo/montar.py

Copia a página, a ponte, o código do pacote `talos` e os arquivos de
exemplo, e baixa uma única vez o Python do navegador (Pyodide) com numpy e
scipy. Este é o único passo que usa a internet; depois disso a pasta
`dist/` funciona sozinha. Para testar localmente:

    python -m http.server 8000 --directory prototipo/dist
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

PASTA_PROTOTIPO = Path(__file__).resolve().parent
RAIZ = PASTA_PROTOTIPO.parent
DESTINO = PASTA_PROTOTIPO / "dist"

VERSAO_PYODIDE = "0.27.7"
ENDERECO_PYODIDE = f"https://cdn.jsdelivr.net/pyodide/v{VERSAO_PYODIDE}/full/"
ARQUIVOS_PYODIDE = (
    "pyodide.js",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
    "numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "openblas-0.3.26.zip",
    "scipy-1.14.1-cp312-cp312-pyodide_2024_0_wasm32.whl",
)
ARQUIVOS_DA_PAGINA = ("index.html", "motor.js", "ponte.py")
ARQUIVOS_DE_EXEMPLO = ("cliente.json", "prateleira.csv", "acoes.csv", "curva_di.csv", "premissas.json")


def copiar_codigo() -> list[str]:
    """Copia página, ponte, pacote `talos` e exemplos; devolve a lista de arquivos Python e de dados."""
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS_DA_PAGINA:
        shutil.copy2(PASTA_PROTOTIPO / nome, DESTINO / nome)

    arquivos = ["ponte.py"]
    destino_talos = DESTINO / "talos"
    if destino_talos.exists():
        shutil.rmtree(destino_talos)
    destino_talos.mkdir()
    for fonte in sorted((RAIZ / "src" / "talos").glob("*.py")):
        shutil.copy2(fonte, destino_talos / fonte.name)
        arquivos.append(f"talos/{fonte.name}")

    destino_exemplos = DESTINO / "exemplos"
    destino_exemplos.mkdir(exist_ok=True)
    for nome in ARQUIVOS_DE_EXEMPLO:
        shutil.copy2(RAIZ / "exemplos" / nome, destino_exemplos / nome)
        arquivos.append(f"exemplos/{nome}")
    return arquivos


def baixar_pyodide() -> None:
    """Baixa os arquivos do Pyodide que ainda não estão em `dist/pyodide/`."""
    pasta = DESTINO / "pyodide"
    pasta.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS_PYODIDE:
        destino = pasta / nome
        if destino.exists():
            continue
        print(f"Baixando {nome}...")
        with urllib.request.urlopen(ENDERECO_PYODIDE + nome) as resposta, destino.open("wb") as arquivo:
            shutil.copyfileobj(resposta, arquivo)


def main() -> None:
    """Monta a pasta `dist/` e grava a lista de arquivos que a página deve carregar."""
    arquivos = copiar_codigo()
    (DESTINO / "arquivos.json").write_text(json.dumps(arquivos, indent=2), encoding="utf-8")
    baixar_pyodide()
    print(f"Pronto: {DESTINO}")


if __name__ == "__main__":
    main()
