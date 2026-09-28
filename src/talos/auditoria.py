"""Registro de auditoria: um arquivo JSON por recomendação, na pasta `saidas/`.

Cada registro guarda tudo o que é preciso para refazer e conferir a
recomendação: data e hora, versão do código, todas as entradas (cliente,
prateleira e curva), premissas, semente, resultado e explicações. A
"impressão digital" (SHA-256) das entradas permite verificar depois que
nada foi alterado.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from talos import __version__
from talos.explicacao import Explicacao
from talos.mercado import CurvaDI
from talos.modelos import Premissas, Produto
from talos.otimizador import Recomendacao

PASTA_PADRAO = Path("saidas")
CARACTERES_PERMITIDOS_NO_NOME = re.compile(r"[^A-Za-z0-9_-]+")


def registrar(
    recomendacao: Recomendacao,
    explicacao: Explicacao,
    produtos: Sequence[Produto],
    curva: CurvaDI,
    premissas: Premissas,
    pasta: Path = PASTA_PADRAO,
    agora: datetime | None = None,
) -> Path:
    """Grava o registro JSON da recomendação e devolve o caminho do arquivo criado."""
    agora = agora or datetime.now().astimezone()
    entradas = {
        "cliente": recomendacao.cliente,
        "prateleira": list(produtos),
        "curva_di": curva,
    }
    registro = {
        "data_hora": agora.isoformat(timespec="seconds"),
        "versao_talos": __version__,
        "semente": recomendacao.semente,
        "reamostragem": recomendacao.reamostragem,
        "impressao_digital_entradas": impressao_digital(entradas, premissas),
        "entradas": entradas,
        "premissas": premissas,
        "resultado": {
            "perfil": recomendacao.perfil,
            "metas": recomendacao.resultados,
            "uso_fgc_total": recomendacao.uso_fgc,
        },
        "explicacoes": explicacao,
        "aviso": explicacao.aviso,
    }
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = _caminho_livre(pasta, recomendacao.cliente.identificador, agora)
    caminho.write_text(_para_texto_json(registro), encoding="utf-8")
    return caminho


def ler_registro(caminho: Path) -> dict[str, Any]:
    """Lê um registro de auditoria gravado por `registrar`."""
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def impressao_digital(entradas: dict[str, Any], premissas: Premissas) -> str:
    """SHA-256 das entradas e premissas: muda se qualquer dado de entrada mudar."""
    texto = json.dumps(
        _para_json({"entradas": entradas, "premissas": premissas}),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _caminho_livre(pasta: Path, identificador: str, agora: datetime) -> Path:
    """Nome de arquivo seguro em qualquer sistema e que não sobrescreve registros anteriores."""
    nome_seguro = CARACTERES_PERMITIDOS_NO_NOME.sub("_", identificador).strip("_") or "cliente"
    base = f"recomendacao_{nome_seguro}_{agora:%Y%m%d-%H%M%S}"
    caminho = pasta / f"{base}.json"
    contador = 1
    while caminho.exists():
        contador += 1
        caminho = pasta / f"{base}_{contador}.json"
    return caminho


def _para_texto_json(dados: Any) -> str:
    """Converte o registro em texto JSON legível, com acentos preservados."""
    return json.dumps(_para_json(dados), ensure_ascii=False, indent=2)


def _para_json(valor: Any) -> Any:
    """Converte dataclasses, datas, tuplas e números do numpy em tipos que o JSON aceita."""
    if is_dataclass(valor) and not isinstance(valor, type):
        return _para_json(asdict(valor))
    if isinstance(valor, dict):
        return {str(chave): _para_json(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_para_json(item) for item in valor]
    if isinstance(valor, (date, datetime)):
        return valor.isoformat()
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, np.generic):
        return valor.item()
    return valor
