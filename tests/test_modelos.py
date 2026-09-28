"""Testes de leitura dos arquivos de exemplo e das validações de `modelos.py`."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from talos.modelos import (
    CLASSES,
    Cliente,
    ErroDeDados,
    Meta,
    Produto,
    carregar_cliente,
    carregar_prateleira,
    carregar_premissas,
)

CABECALHO_PRATELEIRA = (
    "nome,emissor,conglomerado,indexador,taxa,taxa_adm_aa,vencimento_meses,"
    "carencia_meses,liquidez_diaria,tributacao,classe,risco,coberto_fgc,aplicacao_minima,setor"
)
LINHA_CDB = "CDB Teste,Banco X,Grupo X,cdi,1.10,0,24,0,não,regressivo,pos_fixado,2,sim,1000,financeiro"


def escrever(caminho: Path, conteudo: str) -> Path:
    """Grava um arquivo de teste em UTF-8 e devolve o caminho."""
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def produto_valido(**mudancas: object) -> Produto:
    """Cria um CDB válido, trocando só os campos pedidos."""
    campos = dict(
        nome="CDB Teste", emissor="Banco X", conglomerado="Grupo X", indexador="cdi",
        taxa=1.10, taxa_adm_aa=0.0, vencimento_meses=24, carencia_meses=0,
        liquidez_diaria=False, tributacao="regressivo", classe="pos_fixado",
        risco=2, coberto_fgc=True, aplicacao_minima=1000.0, setor="financeiro",
    )
    campos.update(mudancas)
    return Produto(**campos)


# --- Arquivos de exemplo ------------------------------------------------------

def test_exemplos_json_avisam_que_sao_ficticios(exemplos: Path) -> None:
    """Todo JSON de exemplo começa com um campo `_aviso` dizendo que é fictício."""
    for nome in ("cliente.json", "premissas.json"):
        dados = json.loads((exemplos / nome).read_text(encoding="utf-8"))
        assert next(iter(dados)) == "_aviso"
        assert "fictício" in dados["_aviso"]


def test_exemplos_csv_avisam_que_sao_ficticios(exemplos: Path) -> None:
    """Todo CSV de exemplo começa com uma linha de comentário dizendo que é fictício."""
    for nome in ("prateleira.csv", "curva_di.csv"):
        primeira_linha = (exemplos / nome).read_text(encoding="utf-8").splitlines()[0]
        assert primeira_linha.startswith("#")
        assert "fictícios" in primeira_linha


def test_carregar_cliente_de_exemplo(exemplos: Path) -> None:
    """O cliente de exemplo tem perfil moderado, 4 metas e uma reserva de emergência."""
    cliente = carregar_cliente(exemplos / "cliente.json")
    assert cliente.perfil == "moderado"
    assert cliente.renda_mensal == 4500.0
    assert cliente.dividas_caras is False
    assert [meta.nome for meta in cliente.metas] == [
        "Reserva de emergência", "Entrada do apartamento", "Aposentadoria", "Viagem",
    ]
    assert [meta.reserva_emergencia for meta in cliente.metas] == [True, False, False, False]
    assert cliente.metas[1] == Meta("Entrada do apartamento", 25000.0, 36000.0, 48, False)


def test_carregar_prateleira_de_exemplo(exemplos: Path) -> None:
    """A prateleira converte textos em números, sim/não e vencimento vazio."""
    produtos = {p.nome: p for p in carregar_prateleira(exemplos / "prateleira.csv")}
    assert len(produtos) == 15

    selic = produtos["Tesouro Selic 2031"]
    assert (selic.indexador, selic.liquidez_diaria, selic.coberto_fgc) == ("selic", True, False)

    lci = produtos["LCI Horizonte 1 ano"]
    assert (lci.taxa, lci.carencia_meses, lci.tributacao) == (0.92, 12, "isento")

    etf = produtos["ETF Ações Brasil Atlas"]
    assert etf.vencimento_meses is None
    assert etf.renda_variavel is True
    assert lci.renda_variavel is False


def test_prateleira_agrupa_emissores_do_mesmo_conglomerado(exemplos: Path) -> None:
    """Dois emissores diferentes podem pertencer ao mesmo conglomerado (importa para o FGC)."""
    produtos = carregar_prateleira(exemplos / "prateleira.csv")
    boreal = {p.emissor for p in produtos if p.conglomerado == "Conglomerado Boreal"}
    assert boreal == {"Banco Boreal", "Financeira Boreal"}


def test_carregar_premissas_de_exemplo(exemplos: Path) -> None:
    """As premissas trazem IR, FGC, perfis e cenários com os valores do arquivo."""
    premissas = carregar_premissas(exemplos / "premissas.json")
    assert premissas.data_referencia == date(2026, 9, 25)

    tabela = premissas.impostos.tabela_regressiva
    assert [(f.ate_dias, f.aliquota) for f in tabela] == [
        (180, 0.225), (360, 0.20), (720, 0.175), (None, 0.15),
    ]
    assert premissas.impostos.meses_come_cotas == (5, 11)

    assert premissas.fgc.limite_por_conglomerado == 250_000
    assert premissas.fgc.teto_global == 1_000_000

    assert premissas.perfil("moderado").teto_renda_variavel == 0.20
    assert premissas.otimizacao.limite_por_produto == 0.40
    assert premissas.cenarios.semente == 42
    assert set(premissas.cenarios.classes) == set(CLASSES)


# --- Validações ---------------------------------------------------------------

def test_perfil_desconhecido_gera_erro(exemplos: Path) -> None:
    """Pedir um perfil que não existe nas premissas gera erro claro."""
    premissas = carregar_premissas(exemplos / "premissas.json")
    with pytest.raises(ErroDeDados, match="perfil do cliente"):
        premissas.perfil("agressivo")


@pytest.mark.parametrize(
    ("campo", "valor"),
    [("valor_atual", 0.0), ("valor_alvo", -1.0), ("prazo_meses", 0)],
)
def test_meta_com_valor_invalido_gera_erro(campo: str, valor: float) -> None:
    """Meta com valor atual zero, alvo negativo ou prazo zero é recusada."""
    campos = dict(nome="Meta", valor_atual=100.0, valor_alvo=200.0, prazo_meses=12, reserva_emergencia=False)
    campos[campo] = valor
    with pytest.raises(ErroDeDados, match=campo):
        Meta(**campos)


def test_cliente_sem_metas_gera_erro() -> None:
    """Um cliente precisa ter pelo menos uma meta."""
    with pytest.raises(ErroDeDados, match="pelo menos uma meta"):
        Cliente("c1", "moderado", 3000.0, False, ())


def test_cliente_json_sem_campo_obrigatorio(tmp_path: Path) -> None:
    """Um JSON de cliente sem o campo `perfil` gera erro dizendo qual campo falta."""
    arquivo = escrever(tmp_path / "cliente.json", json.dumps({
        "identificador": "c1", "renda_mensal": 3000, "dividas_caras": False, "metas": [],
    }))
    with pytest.raises(ErroDeDados, match="'perfil' ausente"):
        carregar_cliente(arquivo)


def test_cliente_json_com_texto_no_lugar_de_booleano(tmp_path: Path) -> None:
    """`dividas_caras` precisa ser true/false, e não o texto "sim"."""
    arquivo = escrever(tmp_path / "cliente.json", json.dumps({
        "identificador": "c1", "perfil": "moderado", "renda_mensal": 3000,
        "dividas_caras": "sim", "metas": [],
    }))
    with pytest.raises(ErroDeDados, match="dividas_caras"):
        carregar_cliente(arquivo)


def test_json_invalido_informa_a_linha(tmp_path: Path) -> None:
    """Um JSON quebrado gera erro com a linha do problema."""
    arquivo = escrever(tmp_path / "cliente.json", '{\n  "perfil": "moderado",\n}')
    with pytest.raises(ErroDeDados, match="linha 3"):
        carregar_cliente(arquivo)


@pytest.mark.parametrize(
    ("mudanca", "trecho_da_mensagem"),
    [
        ({"risco": 6}, "risco"),
        ({"indexador": "dolar"}, "indexador"),
        ({"classe": "cripto"}, "classe"),
        ({"tributacao": "isenta"}, "tributacao"),
        ({"carencia_meses": 36}, "carência"),
    ],
)
def test_produto_invalido_gera_erro(mudanca: dict[str, object], trecho_da_mensagem: str) -> None:
    """Produto com risco fora de 1 a 5, categoria desconhecida ou carência maior que o vencimento é recusado."""
    with pytest.raises(ErroDeDados, match=trecho_da_mensagem):
        produto_valido(**mudanca)


def test_prateleira_sem_coluna_obrigatoria(tmp_path: Path) -> None:
    """CSV sem a coluna `risco` gera erro dizendo qual coluna falta."""
    cabecalho = CABECALHO_PRATELEIRA.replace(",risco", "")
    linha = LINHA_CDB.replace(",2,sim", ",sim")
    arquivo = escrever(tmp_path / "prateleira.csv", f"{cabecalho}\n{linha}\n")
    with pytest.raises(ErroDeDados, match="colunas ausentes: risco"):
        carregar_prateleira(arquivo)


def test_prateleira_com_sim_nao_invalido(tmp_path: Path) -> None:
    """Valor diferente de sim/não numa coluna de verdadeiro/falso gera erro."""
    linha = LINHA_CDB.replace(",sim,1000", ",talvez,1000")
    arquivo = escrever(tmp_path / "prateleira.csv", f"{CABECALHO_PRATELEIRA}\n{linha}\n")
    with pytest.raises(ErroDeDados, match="coberto_fgc"):
        carregar_prateleira(arquivo)


def test_prateleira_com_virgula_decimal(tmp_path: Path) -> None:
    """Número com vírgula decimal gera erro explicando que o separador é o ponto."""
    linha = LINHA_CDB.replace(",1.10,", ',"1,10",')
    arquivo = escrever(tmp_path / "prateleira.csv", f"{CABECALHO_PRATELEIRA}\n{linha}\n")
    with pytest.raises(ErroDeDados, match="use ponto"):
        carregar_prateleira(arquivo)


def test_prateleira_ignora_comentarios(tmp_path: Path) -> None:
    """Linhas começando com # são ignoradas, antes ou no meio dos dados."""
    conteudo = f"# aviso\n{CABECALHO_PRATELEIRA}\n# outro comentário\n{LINHA_CDB}\n"
    produtos = carregar_prateleira(escrever(tmp_path / "prateleira.csv", conteudo))
    assert produtos == [produto_valido()]


def test_prateleira_com_nomes_repetidos(tmp_path: Path) -> None:
    """Dois produtos com o mesmo nome geram erro."""
    conteudo = f"{CABECALHO_PRATELEIRA}\n{LINHA_CDB}\n{LINHA_CDB}\n"
    with pytest.raises(ErroDeDados, match="mesmo nome"):
        carregar_prateleira(escrever(tmp_path / "prateleira.csv", conteudo))


def _premissas_alteradas(
    exemplos: Path, tmp_path: Path, alterar: Callable[[dict], None]
) -> Path:
    """Copia as premissas de exemplo, aplica uma alteração e grava num arquivo temporário."""
    dados = json.loads((exemplos / "premissas.json").read_text(encoding="utf-8"))
    alterar(dados)
    return escrever(tmp_path / "premissas.json", json.dumps(dados, ensure_ascii=False))


def test_tabela_ir_fora_de_ordem(exemplos: Path, tmp_path: Path) -> None:
    """Faixas de IR fora de ordem crescente geram erro."""
    def trocar(dados: dict) -> None:
        faixas = dados["impostos"]["tabela_regressiva"]
        faixas[0], faixas[1] = faixas[1], faixas[0]
    with pytest.raises(ErroDeDados, match="ordem crescente"):
        carregar_premissas(_premissas_alteradas(exemplos, tmp_path, trocar))


def test_aliquota_em_porcentagem_gera_erro(exemplos: Path, tmp_path: Path) -> None:
    """Alíquota escrita como 15 (em vez de 0.15) é recusada."""
    def estragar(dados: dict) -> None:
        dados["impostos"]["aliquota_acoes"] = 15
    with pytest.raises(ErroDeDados, match="entre 0 e 1"):
        carregar_premissas(_premissas_alteradas(exemplos, tmp_path, estragar))


def test_correlacao_assimetrica_gera_erro(exemplos: Path, tmp_path: Path) -> None:
    """Matriz de correlação que não é simétrica é recusada."""
    def estragar(dados: dict) -> None:
        dados["cenarios"]["correlacoes"]["matriz"][0][2] = 0.5
    with pytest.raises(ErroDeDados, match="simétrica"):
        carregar_premissas(_premissas_alteradas(exemplos, tmp_path, estragar))


# --- Emprego e setor -----------------------------------------------------------------------------

def test_cliente_de_exemplo_sem_emprego_informado(exemplos: Path) -> None:
    """No exemplo, empregador e setor de trabalho estão como null."""
    cliente = carregar_cliente(exemplos / "cliente.json")
    assert cliente.empregador is None
    assert cliente.setor_trabalho is None


def test_setores_da_prateleira_e_limite_nas_premissas(exemplos: Path) -> None:
    """Cada produto tem setor; o teto do setor do trabalho vem das premissas (30%)."""
    produtos = {p.nome: p for p in carregar_prateleira(exemplos / "prateleira.csv")}
    assert produtos["CDB Boreal 3 anos"].setor == "financeiro"
    assert produtos["Tesouro Selic 2031"].setor == "governo_federal"
    assert produtos["Debênture Incentivada Energia Sol"].setor == "energia"
    assert carregar_premissas(exemplos / "premissas.json").otimizacao.limite_setor_do_trabalho == 0.30


def test_setor_desconhecido_gera_erro() -> None:
    """Setor de trabalho fora da lista é recusado, com a lista de opções na mensagem."""
    with pytest.raises(ErroDeDados, match="setor_trabalho"):
        Cliente("c1", "moderado", 3000.0, False, (Meta("M", 1.0, 2.0, 12, False),), setor_trabalho="bancos")
    with pytest.raises(ErroDeDados, match="setor"):
        produto_valido(setor="bancos")


def test_cliente_json_precisa_dizer_o_empregador(tmp_path: Path) -> None:
    """O JSON precisa ter o campo `empregador` (mesmo que null), para erros de digitação não passarem."""
    arquivo = escrever(tmp_path / "cliente.json", json.dumps({
        "identificador": "c1", "perfil": "moderado", "renda_mensal": 3000, "dividas_caras": False,
        "setor_trabalho": None, "metas": [],
    }))
    with pytest.raises(ErroDeDados, match="'empregador' ausente"):
        carregar_cliente(arquivo)
