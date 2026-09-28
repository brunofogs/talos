"""Testes da API HTTP (`talos.api`): um servidor de verdade numa porta livre, chamado por HTTP."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from talos import AVISO_SIMULACAO, __version__
from talos.__main__ import main
from talos.api import TAMANHO_MAXIMO_CORPO, carregar_motor, chamar, criar_servidor
from talos.api import testar_api as conversa_de_teste

EXEMPLOS = Path(__file__).resolve().parent.parent / "exemplos"
POUCOS_CENARIOS = "?cenarios=500"


@pytest.fixture(scope="module")
def pasta_saidas(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pasta temporária para os registros de auditoria gravados pela API."""
    return tmp_path_factory.mktemp("saidas")


@pytest.fixture(scope="module")
def url(pasta_saidas: Path) -> Iterator[str]:
    """Liga a API numa porta livre (porta 0) durante os testes deste arquivo."""
    servidor = criar_servidor(carregar_motor(EXEMPLOS, pasta_saidas), "127.0.0.1", 0)
    linha = threading.Thread(target=servidor.serve_forever, daemon=True)
    linha.start()
    host, porta = servidor.server_address[:2]
    yield f"http://{host}:{porta}"
    servidor.shutdown()
    servidor.server_close()


@pytest.fixture(scope="module")
def cliente() -> dict:
    """Cliente de exemplo, como uma corretora enviaria."""
    return json.loads((EXEMPLOS / "cliente.json").read_text(encoding="utf-8"))


def test_saude(url: str) -> None:
    """GET /saude responde que o serviço está no ar e qual a versão."""
    assert chamar(f"{url}/saude") == (200, {"status": "ok", "versao_talos": __version__})


def test_prateleira(url: str) -> None:
    """GET /prateleira devolve os 15 produtos, os perfis aceitos e o aviso."""
    status, corpo = chamar(f"{url}/prateleira")
    assert status == 200
    assert len(corpo["produtos"]) == 15
    assert corpo["perfis"] == ["arrojado", "conservador", "moderado"]
    assert corpo["aviso"] == AVISO_SIMULACAO


def test_recomendacoes(url: str, cliente: dict, pasta_saidas: Path) -> None:
    """POST /recomendacoes devolve as 4 metas com carteira e explicações, e grava a auditoria."""
    status, corpo = chamar(f"{url}/recomendacoes{POUCOS_CENARIOS}", "POST", cliente)
    assert status == 200
    assert corpo["cliente"] == "cliente-exemplo-001"
    assert corpo["cenarios"] == 500
    assert [m["nome"] for m in corpo["metas"]][0] == "Reserva de emergência"
    reserva = corpo["metas"][0]
    assert reserva["situacao"] == "atingivel"
    assert sum(item["peso"] for item in reserva["carteira"]) == pytest.approx(1.0)
    assert all(item["explicacoes"] for item in reserva["carteira"])
    assert set(reserva["valor_final_por_percentil"]) == {"p5", "p25", "p50", "p75", "p95"}
    assert list(corpo)[-1] == "aviso" and corpo["aviso"] == AVISO_SIMULACAO
    assert (pasta_saidas / f"{corpo['id']}.json").exists()


def test_validacoes(url: str, cliente: dict) -> None:
    """POST /validacoes devolve, por meta, Talos, 100% do CDI e pesos iguais, com a conclusão."""
    status, corpo = chamar(f"{url}/validacoes{POUCOS_CENARIOS}", "POST", cliente)
    assert status == 200
    assert len(corpo["metas"]) == 4
    nomes = [c["carteira"] for c in corpo["metas"][1]["carteiras"]]
    assert nomes == ["Talos", "100% do CDI", "Pesos iguais"]
    assert corpo["metas"][1]["conclusao"]


def test_semente_na_url(url: str, cliente: dict) -> None:
    """?semente=7 troca a semente dos cenários."""
    status, corpo = chamar(f"{url}/recomendacoes{POUCOS_CENARIOS}&semente=7", "POST", cliente)
    assert (status, corpo["semente"]) == (200, 7)


@pytest.mark.parametrize(
    ("caminho", "metodo", "corpo", "status", "trecho"),
    [
        ("/carteiras", "GET", None, 404, "Rota /carteiras não existe"),
        ("/recomendacoes", "GET", None, 405, "não aceita GET"),
        ("/saude", "POST", {}, 405, "não aceita POST"),
        ("/recomendacoes?cenarios=muitos", "POST", {}, 400, "precisa ser um número inteiro"),
        ("/recomendacoes?moeda=usd", "POST", {}, 400, "Parâmetro 'moeda' desconhecido"),
        ("/recomendacoes?cenarios=0", "POST", "cliente", 400, "cenarios precisa ser de pelo menos 1"),
    ],
)
def test_erros_com_codigo_http(url: str, cliente: dict, caminho: str, metodo: str, corpo, status: int, trecho: str) -> None:
    """Rotas, métodos e parâmetros errados voltam com o código HTTP certo e mensagem em português."""
    resposta = chamar(f"{url}{caminho}", metodo, cliente if corpo == "cliente" else corpo)
    assert resposta[0] == status
    assert trecho in resposta[1]["erro"]


def test_cliente_invalido(url: str, cliente: dict) -> None:
    """Perfil desconhecido volta como 400 com a lista de perfis aceitos."""
    status, corpo = chamar(f"{url}/recomendacoes{POUCOS_CENARIOS}", "POST", {**cliente, "perfil": "agressivo"})
    assert status == 400
    assert corpo["erro"].startswith("perfil do cliente: valor 'agressivo' não reconhecido")


def enviar_bruto(url: str, dados: bytes) -> tuple[int, dict]:
    """POST com bytes quaisquer no corpo (para testar JSON quebrado e corpo grande)."""
    requisicao = urllib.request.Request(f"{url}/recomendacoes", data=dados, method="POST")
    try:
        with urllib.request.urlopen(requisicao) as resposta:
            return resposta.status, json.loads(resposta.read())
    except urllib.error.HTTPError as erro:
        return erro.code, json.loads(erro.read())


def test_json_quebrado(url: str) -> None:
    """JSON quebrado volta como 400, dizendo linha e coluna."""
    assert enviar_bruto(url, b'{"perfil": ') == (400, {"erro": "JSON inválido na linha 1, coluna 12."})


def test_corpo_vazio(url: str) -> None:
    """POST sem corpo pede os dados do cliente."""
    status, corpo = enviar_bruto(url, b"")
    assert status == 400
    assert "Envie os dados do cliente" in corpo["erro"]


def test_corpo_grande_demais(url: str) -> None:
    """Corpo acima de 1 MB é recusado com 413, sem ser processado."""
    status, corpo = enviar_bruto(url, b" " * (TAMANHO_MAXIMO_CORPO + 1))
    assert status == 413
    assert "limite" in corpo["erro"]


# --- testar-api -------------------------------------------------------------------------------

def test_testar_api_mostra_a_resposta_inteira(url: str) -> None:
    """O cliente de teste chama as três rotas e mostra o JSON da recomendação como chegou."""
    texto = conversa_de_teste(url, EXEMPLOS / "cliente.json", cenarios=500)
    assert f"GET {url}/saude -> 200" in texto
    assert "15 produtos" in texto
    assert f"POST {url}/recomendacoes?cenarios=500" in texto
    corpo = json.loads(texto[texto.index("{\n"):])
    assert len(corpo["metas"]) == 4
    assert corpo["aviso"] == AVISO_SIMULACAO


def test_testar_api_pela_linha_de_comando(url: str, capsys: pytest.CaptureFixture[str]) -> None:
    """`python -m talos testar-api --url ...` imprime a conversa com a API."""
    argumentos = ["testar-api", "--url", url, "--cliente", str(EXEMPLOS / "cliente.json"), "--cenarios", "500"]
    assert main(argumentos) == 0
    assert '"situacao": "atingivel"' in capsys.readouterr().out


def test_testar_api_sem_servidor(capsys: pytest.CaptureFixture[str]) -> None:
    """Sem a API ligada, o testar-api explica como ligá-la e termina com código 2."""
    argumentos = ["testar-api", "--url", "http://127.0.0.1:9", "--cliente", str(EXEMPLOS / "cliente.json")]
    assert main(argumentos) == 2
    assert "python -m talos servir" in capsys.readouterr().err


def test_servir_com_porta_ocupada(url: str, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Se a porta já está em uso, o servir avisa e sugere outra, em vez de quebrar."""
    porta = url.rsplit(":", 1)[1]
    argumentos = ["servir", "--porta", porta, "--dados", str(EXEMPLOS), "--saidas", str(tmp_path)]
    assert main(argumentos) == 2
    assert "Tente outra com --porta" in capsys.readouterr().err
