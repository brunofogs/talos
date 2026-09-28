"""Ponte entre a página do protótipo e o motor Talos.

A página roda Python no navegador (Pyodide) e chama `simular` com o
cliente em JSON. A resposta é um JSON com as carteiras, as estatísticas,
a faixa de valores finais de cada meta e as explicações em português.
Nenhuma conta é feita aqui: tudo vem dos módulos do pacote `talos`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from talos.explicacao import explicar
from talos.mercado import carregar_curva_di, cdi_medio_projetado, gerar_cenarios
from talos.modelos import (
    ErroDeDados,
    carregar_prateleira,
    carregar_premissas,
    cliente_de_dados,
)
from talos.otimizador import ResultadoMeta, fatores_liquidos, otimizar_cliente

PERCENTIS_DA_FAIXA = {"p5": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}


def dados_da_prateleira(pasta_dados: str | Path) -> str:
    """Prateleira, premissas principais e cliente de exemplo, em JSON, para a página exibir."""
    pasta = Path(pasta_dados)
    premissas = carregar_premissas(pasta / "premissas.json")
    curva = carregar_curva_di(pasta / "curva_di.csv", premissas)
    cliente = json.loads((pasta / "cliente.json").read_text(encoding="utf-8"))
    return json.dumps({
        "produtos": [asdict(p) for p in carregar_prateleira(pasta / "prateleira.csv")],
        "perfis": {nome: asdict(perfil) for nome, perfil in premissas.perfis.items()},
        "data_referencia": premissas.data_referencia.isoformat(),
        "cdi_12_meses": cdi_medio_projetado(curva, 12),
        "quantidade_cenarios": premissas.cenarios.quantidade,
        "cliente_exemplo": {k: v for k, v in cliente.items() if not k.startswith("_")},
    }, ensure_ascii=False)


def simular(entrada_json: str, pasta_dados: str | Path) -> str:
    """Roda o Talos para o cliente enviado pela página.

    `entrada_json` tem o formato do `cliente.json` e pode trazer
    `quantidade` (número de cenários). Erros de dados voltam como
    {"erro": "mensagem"} em vez de quebrar a página.
    """
    pasta = Path(pasta_dados)
    try:
        entrada = json.loads(entrada_json)
        quantidade = entrada.pop("quantidade", None)
        cliente = cliente_de_dados(entrada)
        premissas = carregar_premissas(pasta / "premissas.json")
        premissas.perfil(cliente.perfil)
    except (ErroDeDados, json.JSONDecodeError) as erro:
        return json.dumps({"erro": str(erro)}, ensure_ascii=False)

    curva = carregar_curva_di(pasta / "curva_di.csv", premissas)
    produtos = carregar_prateleira(pasta / "prateleira.csv")
    recomendacao = otimizar_cliente(cliente, produtos, premissas, curva, quantidade=quantidade)
    explicacao = explicar(recomendacao, premissas)

    metas = []
    for resultado, texto in zip(recomendacao.resultados, explicacao.metas):
        metas.append({
            "nome": resultado.meta.nome,
            "situacao": resultado.situacao,
            "reserva_emergencia": resultado.meta.reserva_emergencia,
            "valor_atual": resultado.meta.valor_atual,
            "valor_alvo": resultado.meta.valor_alvo,
            "prazo_meses": resultado.meta.prazo_meses,
            "retorno_necessario_aa": resultado.retorno_necessario_aa,
            "retorno_esperado_aa": resultado.retorno_esperado_aa,
            "cvar": resultado.cvar,
            "probabilidade_sucesso": resultado.probabilidade_sucesso,
            "faixa_valor_final": faixa_valor_final(resultado, premissas, curva, quantidade),
            "alocacoes": [
                {
                    "produto": a.produto.nome,
                    "emissor": a.produto.emissor,
                    "classe": a.produto.classe,
                    "peso": a.peso,
                    "valor": a.valor,
                    "retorno_liquido_esperado_aa": a.retorno_liquido_esperado_aa,
                    "frases": list(texto.produtos[a.produto.nome]),
                }
                for a in resultado.alocacoes
            ],
            "resumo": list(texto.resumo),
            "nao_usados": list(texto.nao_usados),
        })
    return json.dumps({
        "perfil": recomendacao.perfil.nome,
        "semente": recomendacao.semente,
        "antes_de_tudo": list(explicacao.antes_de_tudo),
        "metas": metas,
        "aviso": explicacao.aviso,
    }, ensure_ascii=False)


def faixa_valor_final(resultado: ResultadoMeta, premissas, curva, quantidade: int | None) -> dict[str, float] | None:
    """Valores finais da carteira (em reais) nos percentis 5, 25, 50, 75 e 95 dos cenários.

    Usa os mesmos cenários do otimizador (mesma semente, prazo e quantidade).
    """
    if not resultado.alocacoes:
        return None
    cenarios = gerar_cenarios(premissas, curva, resultado.meta.prazo_meses, resultado.semente, quantidade)
    fatores = fatores_liquidos([a.produto for a in resultado.alocacoes], cenarios, premissas)
    pesos = np.array([a.peso for a in resultado.alocacoes])
    valores = fatores @ pesos * resultado.meta.valor_atual
    return {nome: float(np.percentile(valores, p)) for nome, p in PERCENTIS_DA_FAIXA.items()}
