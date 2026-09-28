"""API HTTP do Talos: o jeito de uma corretora pedir recomendações por sistema.

Usa só o servidor HTTP da biblioteca padrão do Python (nenhuma dependência
nova). Por padrão atende apenas no próprio computador (127.0.0.1) e não tem
autenticação: é para demonstração e integração local, não para a internet.

Rotas:
- `GET  /saude`          -> situação do serviço e versão;
- `GET  /prateleira`     -> produtos, perfis e data das premissas;
- `POST /recomendacoes`  -> corpo no formato do `cliente.json`; devolve a
  carteira de cada meta, as estatísticas e as explicações, e grava a auditoria;
- `POST /validacoes`     -> mesmo corpo; devolve a comparação com 100% do CDI
  e com pesos iguais.

`POST` aceita `?cenarios=N&semente=N` na URL. Erros voltam como
{"erro": "mensagem em português"} com o código HTTP adequado.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from talos import AVISO_SIMULACAO, __version__
from talos.auditoria import registrar
from talos.explicacao import explicar
from talos.mercado import CurvaDI, carregar_curva_di
from talos.modelos import (
    ErroDeDados,
    Premissas,
    Produto,
    carregar_prateleira,
    carregar_premissas,
    cliente_de_dados,
)
from talos.otimizador import Recomendacao, otimizar_cliente
from talos.validacao import conclusao, validar

HOST_PADRAO = "127.0.0.1"
PORTA_PADRAO = 8000
TAMANHO_MAXIMO_CORPO = 1_000_000  # 1 MB: um cliente com dezenas de metas cabe com folga
LIMITE_DESCARTE = 10 * TAMANHO_MAXIMO_CORPO  # corpo grande demais é lido e jogado fora até aqui
PEDACO_DESCARTE = 65_536
PARAMETROS_INTEIROS = ("cenarios", "semente")
TEMPO_LIMITE_TESTE = 300  # segundos para o testar-api esperar uma resposta


class ErroDaRequisicao(Exception):
    """Erro causado pela requisição, com o código HTTP a devolver."""

    def __init__(self, status: HTTPStatus, mensagem: str) -> None:
        super().__init__(mensagem)
        self.status = status


@dataclass(frozen=True)
class Motor:
    """Dados carregados uma vez na partida do servidor e usados em todas as requisições."""

    premissas: Premissas
    curva: CurvaDI
    produtos: tuple[Produto, ...]
    pasta_saidas: Path


def carregar_motor(pasta_dados: Path, pasta_saidas: Path) -> Motor:
    """Lê premissas, curva e prateleira da pasta de dados."""
    pasta_dados = Path(pasta_dados)
    premissas = carregar_premissas(pasta_dados / "premissas.json")
    return Motor(
        premissas=premissas,
        curva=carregar_curva_di(pasta_dados / "curva_di.csv", premissas),
        produtos=tuple(carregar_prateleira(pasta_dados / "prateleira.csv")),
        pasta_saidas=Path(pasta_saidas),
    )


# ---------------------------------------------------------------------------
# Respostas (sem HTTP: fáceis de testar e de reaproveitar)
# ---------------------------------------------------------------------------

def resposta_saude() -> dict[str, Any]:
    """Situação do serviço."""
    return {"status": "ok", "versao_talos": __version__}


def resposta_prateleira(motor: Motor) -> dict[str, Any]:
    """Produtos disponíveis, perfis aceitos e data das premissas."""
    return {
        "data_referencia": motor.premissas.data_referencia.isoformat(),
        "perfis": sorted(motor.premissas.perfis),
        "produtos": [asdict(produto) for produto in motor.produtos],
        "aviso": AVISO_SIMULACAO,
    }


def resposta_recomendacao(motor: Motor, dados_cliente: Any, cenarios: int | None = None, semente: int | None = None) -> dict[str, Any]:
    """Otimiza as metas do cliente, grava a auditoria e devolve a recomendação em JSON."""
    recomendacao = _otimizar(motor, dados_cliente, cenarios, semente)
    explicacao = explicar(recomendacao, motor.premissas)
    caminho = registrar(recomendacao, explicacao, motor.produtos, motor.curva, motor.premissas, motor.pasta_saidas)
    metas = []
    for resultado, texto in zip(recomendacao.resultados, explicacao.metas):
        percentis = resultado.valor_final_por_percentil
        metas.append({
            "nome": resultado.meta.nome,
            "situacao": resultado.situacao,
            "reserva_emergencia": resultado.meta.reserva_emergencia,
            "valor_atual": resultado.meta.valor_atual,
            "valor_alvo": resultado.meta.valor_alvo,
            "prazo_meses": resultado.meta.prazo_meses,
            "retorno_necessario_aa": resultado.retorno_necessario_aa,
            "retorno_esperado_aa": resultado.retorno_esperado_aa,
            "perda_media_piores_cenarios": resultado.cvar,
            "probabilidade_sucesso": resultado.probabilidade_sucesso,
            "valor_final_por_percentil": (
                {f"p{percentil}": valor for percentil, valor in percentis.items()} if percentis else None
            ),
            "carteira": [
                {
                    "produto": alocacao.produto.nome,
                    "emissor": alocacao.produto.emissor,
                    "conglomerado": alocacao.produto.conglomerado,
                    "peso": alocacao.peso,
                    "valor": round(alocacao.valor, 2),
                    "retorno_liquido_esperado_aa": alocacao.retorno_liquido_esperado_aa,
                    "explicacoes": list(texto.produtos[alocacao.produto.nome]),
                }
                for alocacao in resultado.alocacoes
            ],
            "resumo": list(texto.resumo),
            "nao_usados": list(texto.nao_usados),
        })
    return {
        "id": caminho.stem,
        "versao_talos": __version__,
        "cliente": recomendacao.cliente.identificador,
        "perfil": recomendacao.perfil.nome,
        "semente": recomendacao.semente,
        "cenarios": cenarios or motor.premissas.cenarios.quantidade,
        "antes_de_tudo": list(explicacao.antes_de_tudo),
        "metas": metas,
        "aviso": explicacao.aviso,
    }


def resposta_validacao(motor: Motor, dados_cliente: Any, cenarios: int | None = None, semente: int | None = None) -> dict[str, Any]:
    """Compara, meta a meta, o Talos com 100% do CDI e com pesos iguais."""
    recomendacao = _otimizar(motor, dados_cliente, cenarios, semente)
    comparacoes = validar(recomendacao, motor.produtos, motor.premissas, motor.curva, cenarios)
    return {
        "versao_talos": __version__,
        "cliente": recomendacao.cliente.identificador,
        "perfil": recomendacao.perfil.nome,
        "semente": recomendacao.semente,
        "cenarios": cenarios or motor.premissas.cenarios.quantidade,
        "metas": [
            {
                "nome": comparacao.meta.nome,
                "situacao": comparacao.situacao,
                "carteiras": [asdict(linha) for linha in comparacao.linhas],
                "conclusao": conclusao(comparacao),
            }
            for comparacao in comparacoes
        ],
        "aviso": AVISO_SIMULACAO,
    }


def _otimizar(motor: Motor, dados_cliente: Any, cenarios: int | None, semente: int | None) -> Recomendacao:
    """Valida o cliente e otimiza todas as metas."""
    cliente = cliente_de_dados(dados_cliente)
    motor.premissas.perfil(cliente.perfil)
    if cenarios is not None and cenarios < 1:
        raise ErroDeDados("cenarios precisa ser de pelo menos 1.")
    return otimizar_cliente(
        cliente, motor.produtos, motor.premissas, motor.curva, semente=semente, quantidade=cenarios
    )


# ---------------------------------------------------------------------------
# Servidor HTTP
# ---------------------------------------------------------------------------

class ManipuladorTalos(BaseHTTPRequestHandler):
    """Atende as rotas da API. O `Motor` fica no servidor (`self.server.motor`)."""

    server_version = f"Talos/{__version__}"
    ROTAS_GET = ("/saude", "/prateleira")
    ROTAS_POST = ("/recomendacoes", "/validacoes")

    def do_GET(self) -> None:  # noqa: N802 (nome exigido pela biblioteca)
        """Responde às rotas de leitura."""
        caminho = urlsplit(self.path).path
        self._atender(lambda: self._responder_get(caminho))

    def do_POST(self) -> None:  # noqa: N802
        """Responde às rotas que calculam carteiras."""
        partes = urlsplit(self.path)
        self._atender(lambda: self._responder_post(partes.path, partes.query))

    def _responder_get(self, caminho: str) -> dict[str, Any]:
        """Escolhe a resposta de um GET."""
        if caminho == "/saude":
            return resposta_saude()
        if caminho == "/prateleira":
            return resposta_prateleira(self.server.motor)
        self._rota_inexistente(caminho, self.ROTAS_POST)

    def _responder_post(self, caminho: str, consulta: str) -> dict[str, Any]:
        """Escolhe a resposta de um POST."""
        if caminho not in self.ROTAS_POST:
            self._rota_inexistente(caminho, self.ROTAS_GET)
        dados = self._ler_json()
        opcoes = _parametros(consulta)
        if caminho == "/recomendacoes":
            funcao = resposta_recomendacao
        else:
            funcao = resposta_validacao
        return self.server.calcular(funcao, dados, **opcoes)

    def _rota_inexistente(self, caminho: str, rotas_do_outro_metodo: tuple[str, ...]) -> None:
        """Rota desconhecida (404) ou existente com outro método (405)."""
        if caminho in rotas_do_outro_metodo:
            raise ErroDaRequisicao(HTTPStatus.METHOD_NOT_ALLOWED, f"A rota {caminho} não aceita {self.command}.")
        rotas = ", ".join(f"GET {r}" for r in self.ROTAS_GET) + ", " + ", ".join(f"POST {r}" for r in self.ROTAS_POST)
        raise ErroDaRequisicao(HTTPStatus.NOT_FOUND, f"Rota {caminho} não existe. Rotas disponíveis: {rotas}.")

    def _ler_json(self) -> Any:
        """Lê o corpo da requisição como JSON em UTF-8, com limite de tamanho."""
        try:
            tamanho = int(self.headers.get("Content-Length", "0"))
        except ValueError as erro:
            raise ErroDaRequisicao(HTTPStatus.BAD_REQUEST, "Cabeçalho Content-Length inválido.") from erro
        if tamanho > TAMANHO_MAXIMO_CORPO:
            self._descartar(min(tamanho, LIMITE_DESCARTE))
            raise ErroDaRequisicao(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"O corpo da requisição passa do limite de {TAMANHO_MAXIMO_CORPO:,} bytes.".replace(",", "."),
            )
        if tamanho == 0:
            raise ErroDaRequisicao(HTTPStatus.BAD_REQUEST, "Envie os dados do cliente em JSON no corpo da requisição.")
        try:
            return json.loads(self.rfile.read(tamanho).decode("utf-8"))
        except UnicodeDecodeError as erro:
            raise ErroDaRequisicao(HTTPStatus.BAD_REQUEST, "O corpo precisa estar em UTF-8.") from erro
        except json.JSONDecodeError as erro:
            raise ErroDaRequisicao(
                HTTPStatus.BAD_REQUEST, f"JSON inválido na linha {erro.lineno}, coluna {erro.colno}."
            ) from erro

    def _descartar(self, tamanho: int) -> None:
        """Lê e joga fora o corpo recusado, para o cliente receber a resposta 413 em vez de
        ver a conexão cair no meio do envio.
        """
        restante = tamanho
        while restante > 0:
            pedaco = self.rfile.read(min(PEDACO_DESCARTE, restante))
            if not pedaco:
                break
            restante -= len(pedaco)

    def _atender(self, calcular: Callable[[], dict[str, Any]]) -> None:
        """Calcula a resposta e a envia; transforma erros em respostas JSON."""
        try:
            status, corpo = HTTPStatus.OK, calcular()
        except ErroDaRequisicao as erro:
            status, corpo = erro.status, {"erro": str(erro)}
        except ErroDeDados as erro:
            status, corpo = HTTPStatus.BAD_REQUEST, {"erro": str(erro)}
        except Exception as erro:  # noqa: BLE001 (o serviço não pode cair por um pedido)
            self.log_error("erro interno: %r", erro)
            status, corpo = HTTPStatus.INTERNAL_SERVER_ERROR, {"erro": "Erro interno do Talos. Veja o terminal do servidor."}
        texto = json.dumps(corpo, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(texto)))
        if status != HTTPStatus.OK:
            # O corpo pode não ter sido lido (ex.: grande demais): encerra a conexão
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(texto)

    def log_message(self, formato: str, *argumentos: Any) -> None:
        """Registra cada requisição no terminal, com data e hora."""
        sys.stderr.write(f"[{datetime.now():%d/%m/%Y %H:%M:%S}] {self.address_string()} {formato % argumentos}\n")


def _parametros(consulta: str) -> dict[str, int]:
    """Lê `cenarios` e `semente` da URL (ex.: ?cenarios=1000&semente=7)."""
    parametros = {}
    for nome, valores in parse_qs(consulta).items():
        if nome not in PARAMETROS_INTEIROS:
            raise ErroDaRequisicao(HTTPStatus.BAD_REQUEST, f"Parâmetro '{nome}' desconhecido. Use: {', '.join(PARAMETROS_INTEIROS)}.")
        try:
            parametros[nome] = int(valores[-1])
        except ValueError as erro:
            raise ErroDaRequisicao(HTTPStatus.BAD_REQUEST, f"Parâmetro '{nome}' precisa ser um número inteiro.") from erro
    return parametros


class ServidorTalos(ThreadingHTTPServer):
    """Servidor HTTP em que cada requisição é atendida na sua própria thread.

    As otimizações, que são pesadas, rodam todas numa única thread de cálculo
    fixa: pedidos simultâneos entram numa fila em vez de disputar processador
    e memória, e rotas leves como /saude continuam respondendo na hora.

    No Windows, a opção de reaproveitar endereço deixa dois programas usarem
    a mesma porta sem erro; por isso ela só fica ligada nos outros sistemas,
    onde serve apenas para religar o serviço logo depois de desligá-lo.
    """

    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True

    def __init__(self, endereco: tuple[str, int], motor: Motor) -> None:
        # Criada antes de abrir a porta: se a porta estiver ocupada, server_close ainda a encontra
        self._fila_de_calculo = ThreadPoolExecutor(max_workers=1, thread_name_prefix="talos-calculo")
        self.motor = motor
        super().__init__(endereco, ManipuladorTalos)

    def calcular(self, funcao: Callable[..., dict[str, Any]], dados: Any, **opcoes: int) -> dict[str, Any]:
        """Roda uma resposta pesada na thread de cálculo e espera o resultado."""
        return self._fila_de_calculo.submit(funcao, self.motor, dados, **opcoes).result()

    def server_close(self) -> None:
        """Fecha a porta e encerra a thread de cálculo."""
        super().server_close()
        self._fila_de_calculo.shutdown(wait=True)


def criar_servidor(motor: Motor, host: str = HOST_PADRAO, porta: int = PORTA_PADRAO) -> ServidorTalos:
    """Cria o servidor (porta 0 escolhe uma porta livre)."""
    return ServidorTalos((host, porta), motor)


# ---------------------------------------------------------------------------
# Cliente de teste: faz o papel da corretora
# ---------------------------------------------------------------------------

def chamar(url: str, metodo: str = "GET", corpo: Any = None) -> tuple[int, Any]:
    """Faz uma requisição à API e devolve (código HTTP, JSON da resposta)."""
    dados = None if corpo is None else json.dumps(corpo, ensure_ascii=False).encode("utf-8")
    requisicao = urllib.request.Request(
        url, data=dados, method=metodo, headers={"Content-Type": "application/json; charset=utf-8"}
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=TEMPO_LIMITE_TESTE) as resposta:
            return resposta.status, json.loads(resposta.read().decode("utf-8"))
    except urllib.error.HTTPError as erro:
        return erro.code, json.loads(erro.read().decode("utf-8"))


def testar_api(url_base: str, caminho_cliente: Path, cenarios: int | None = None) -> str:
    """Chama /saude, /prateleira e /recomendacoes como uma corretora faria e devolve o que aconteceu.

    A resposta de /recomendacoes aparece inteira, em JSON, exatamente como chega.
    """
    url_base = url_base.rstrip("/")
    cliente = json.loads(Path(caminho_cliente).read_text(encoding="utf-8"))
    consulta = f"?cenarios={cenarios}" if cenarios else ""
    linhas = []

    status, saude = chamar(f"{url_base}/saude")
    linhas.append(f"GET {url_base}/saude -> {status} · {json.dumps(saude, ensure_ascii=False)}")

    status, prateleira = chamar(f"{url_base}/prateleira")
    quantidade = len(prateleira.get("produtos", [])) if status == HTTPStatus.OK else 0
    linhas.append(f"GET {url_base}/prateleira -> {status} · {quantidade} produtos, perfis {prateleira.get('perfis')}")

    endereco = f"{url_base}/recomendacoes{consulta}"
    linhas.append(f"POST {endereco}  (corpo: {Path(caminho_cliente).name}, cliente {cliente.get('identificador')})")
    status, recomendacao = chamar(endereco, "POST", cliente)
    linhas.append(f"-> {status}")
    linhas.append(json.dumps(recomendacao, ensure_ascii=False, indent=2))
    return "\n".join(linhas)
