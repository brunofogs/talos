"""Estruturas de dados do Talos e leitura dos arquivos de entrada.

Cada arquivo de entrada vira um objeto com tipos conferidos:
`cliente.json` vira `Cliente`, `prateleira.csv` vira uma lista de `Produto`
e `premissas.json` vira `Premissas`. Qualquer problema nos dados gera um
`ErroDeDados` com mensagem em português dizendo onde está o erro.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

INDEXADORES = ("cdi", "selic", "prefixado", "ipca", "variavel")
TRIBUTACOES = (
    "regressivo", "isento", "acoes", "fundo_longo_prazo", "fundo_curto_prazo",
)
CLASSES = ("pos_fixado", "prefixado", "inflacao", "acoes")
CLASSE_RENDA_VARIAVEL = "acoes"
ESCALAS_PRAZO = ("constante", "raiz_do_prazo")
RISCO_MINIMO = 1
RISCO_MAXIMO = 5
PREFIXO_COMENTARIO = "#"
COLUNAS_PRATELEIRA = (
    "nome", "emissor", "conglomerado", "indexador", "taxa", "taxa_adm_aa",
    "vencimento_meses", "carencia_meses", "liquidez_diaria", "tributacao",
    "classe", "risco", "coberto_fgc", "aplicacao_minima", "setor",
)
SETORES = (
    "financeiro", "governo_federal", "energia", "diversificado", "mineracao", "petroleo",
    "varejo", "saude", "industria", "telecom", "agronegocio", "saneamento",
)
COLUNAS_ACOES = (
    "ticker", "empresa", "conglomerado", "setor", "risco", "beta",
    "volatilidade_propria_aa", "alfa_aa", "aplicacao_minima",
)
TEXTOS_SIM = frozenset({"sim", "s", "true", "1"})
TEXTOS_NAO = frozenset({"não", "nao", "n", "false", "0"})


class ErroDeDados(ValueError):
    """Erro em um arquivo ou valor de entrada, com mensagem em português."""


# ---------------------------------------------------------------------------
# Cliente e metas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Meta:
    """Um objetivo financeiro do cliente, com valor e prazo.

    `valor_atual` é quanto o cliente já tem separado para a meta e
    `valor_alvo` é quanto quer ter ao fim de `prazo_meses`.
    """

    nome: str
    valor_atual: float
    valor_alvo: float
    prazo_meses: int
    reserva_emergencia: bool

    def __post_init__(self) -> None:
        """Confere se os valores da meta fazem sentido."""
        contexto = f"meta '{self.nome}'"
        _exigir(bool(self.nome.strip()), "meta: o nome não pode ficar vazio.")
        _exigir(self.valor_atual > 0, f"{contexto}: valor_atual precisa ser maior que zero.")
        _exigir(self.valor_alvo > 0, f"{contexto}: valor_alvo precisa ser maior que zero.")
        _exigir(self.prazo_meses >= 1, f"{contexto}: prazo_meses precisa ser de pelo menos 1.")


@dataclass(frozen=True)
class Cliente:
    """Investidor fictício com perfil de suitability, renda, trabalho e metas.

    `empregador` é o conglomerado onde o cliente trabalha, com o mesmo nome
    usado na prateleira (ou `None`). `setor_trabalho` é o setor do emprego,
    com os nomes de `SETORES` (ou `None` se não aparece na prateleira). Os
    dois servem para não concentrar emprego e investimentos no mesmo lugar.
    """

    identificador: str
    perfil: str
    renda_mensal: float
    dividas_caras: bool
    metas: tuple[Meta, ...]
    empregador: str | None = None
    setor_trabalho: str | None = None

    def __post_init__(self) -> None:
        """Confere se o cliente tem metas, com nomes diferentes, e se o setor é conhecido."""
        _exigir(bool(self.perfil.strip()), "cliente: o perfil não pode ficar vazio.")
        if self.empregador is not None:
            _exigir(bool(self.empregador.strip()), "cliente: empregador vazio; use null se não quiser informar.")
        if self.setor_trabalho is not None:
            _exigir_opcao(self.setor_trabalho, SETORES, "cliente, setor_trabalho")
        _exigir(self.renda_mensal >= 0, "cliente: renda_mensal não pode ser negativa.")
        _exigir(len(self.metas) > 0, "cliente: é preciso pelo menos uma meta.")
        nomes = [meta.nome for meta in self.metas]
        _exigir(len(nomes) == len(set(nomes)), "cliente: há metas com o mesmo nome.")


# ---------------------------------------------------------------------------
# Produtos da prateleira
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Produto:
    """Um produto de investimento da prateleira da instituição parceira.

    O significado de `taxa` depende do `indexador`:
    - `cdi`: fração do CDI (1.10 quer dizer 110% do CDI);
    - `selic` e `ipca`: spread ao ano somado ao indexador;
    - `prefixado`: taxa ao ano;
    - `variavel`: não é usada, o retorno vem dos cenários da classe.

    `vencimento_meses` é `None` para produtos sem vencimento (fundos, ETFs).
    """

    nome: str
    emissor: str
    conglomerado: str
    indexador: str
    taxa: float
    taxa_adm_aa: float
    vencimento_meses: int | None
    carencia_meses: int
    liquidez_diaria: bool
    tributacao: str
    classe: str
    risco: int
    coberto_fgc: bool
    aplicacao_minima: float
    setor: str
    ticker: str | None = None
    beta: float = 1.0
    volatilidade_propria_aa: float = 0.0
    alfa_aa: float = 0.0

    def __post_init__(self) -> None:
        """Confere se o produto usa categorias conhecidas e valores válidos."""
        contexto = f"produto '{self.nome}'"
        _exigir_opcao(self.setor, SETORES, f"{contexto}, setor")
        _exigir(self.volatilidade_propria_aa >= 0, f"{contexto}: volatilidade_propria_aa não pode ser negativa.")
        _exigir(self.alfa_aa > -1, f"{contexto}: alfa_aa precisa ser maior que -100%.")
        if self.ticker is not None:
            _exigir(
                self.indexador == "variavel" and self.renda_variavel,
                f"{contexto}: ação individual precisa ter indexador variavel e classe {CLASSE_RENDA_VARIAVEL}.",
            )
        _exigir(bool(self.nome.strip()), "produto: o nome não pode ficar vazio.")
        _exigir_opcao(self.indexador, INDEXADORES, f"{contexto}, indexador")
        _exigir_opcao(self.tributacao, TRIBUTACOES, f"{contexto}, tributacao")
        _exigir_opcao(self.classe, CLASSES, f"{contexto}, classe")
        _exigir(
            RISCO_MINIMO <= self.risco <= RISCO_MAXIMO,
            f"{contexto}: risco precisa estar entre {RISCO_MINIMO} e {RISCO_MAXIMO}.",
        )
        _exigir(self.taxa_adm_aa >= 0, f"{contexto}: taxa_adm_aa não pode ser negativa.")
        _exigir(self.carencia_meses >= 0, f"{contexto}: carencia_meses não pode ser negativa.")
        _exigir(self.aplicacao_minima >= 0, f"{contexto}: aplicacao_minima não pode ser negativa.")
        if self.vencimento_meses is not None:
            _exigir(self.vencimento_meses >= 1, f"{contexto}: vencimento_meses precisa ser de pelo menos 1.")
            _exigir(
                self.carencia_meses <= self.vencimento_meses,
                f"{contexto}: a carência não pode passar do vencimento.",
            )

    @property
    def renda_variavel(self) -> bool:
        """Diz se o produto conta no teto de renda variável do perfil."""
        return self.classe == CLASSE_RENDA_VARIAVEL

    @property
    def acao_individual(self) -> bool:
        """Diz se o produto é uma ação de uma empresa (e não um fundo ou ETF)."""
        return self.ticker is not None


# ---------------------------------------------------------------------------
# Premissas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FaixaIR:
    """Uma faixa de uma tabela de IR: vale até `ate_dias` (None = sem limite)."""

    ate_dias: int | None
    aliquota: float


@dataclass(frozen=True)
class PremissasImpostos:
    """Alíquotas de IR e regras de come-cotas."""

    tabela_regressiva: tuple[FaixaIR, ...]
    tabela_fundo_curto_prazo: tuple[FaixaIR, ...]
    aliquota_acoes: float
    meses_come_cotas: tuple[int, ...]
    come_cotas_longo_prazo: float
    come_cotas_curto_prazo: float


@dataclass(frozen=True)
class PremissasFGC:
    """Limites de garantia do FGC (conferir no site do FGC antes do uso real)."""

    limite_por_conglomerado: float
    teto_global: float
    janela_teto_anos: int


@dataclass(frozen=True)
class Perfil:
    """Limites de um perfil de suitability.

    `cvar_maximo` é a perda média aceitável nos piores cenários, como
    fração do valor aplicado.
    """

    nome: str
    risco_maximo: int
    teto_renda_variavel: float
    cvar_maximo: float


@dataclass(frozen=True)
class PremissasOtimizacao:
    """Parâmetros do otimizador de carteiras."""

    nivel_cvar: float
    limite_por_produto: float
    indexadores_sem_limite: tuple[str, ...]
    prazo_minimo_renda_variavel_meses: int
    prazo_minimo_risco_3_meses: int
    reamostragens: int
    limite_setor_do_trabalho: float
    limite_por_acao: float
    peso_minimo_por_produto: float
    limite_por_conglomerado_na_meta: float
    conglomerados_sem_limite: tuple[str, ...]
    nivel_crescimento: float


@dataclass(frozen=True)
class ClasseCenario:
    """Parâmetros de uma classe de ativo para gerar cenários.

    `escala_prazo` diz como a incerteza do retorno anualizado muda com o prazo:
    - `raiz_do_prazo`: cai com a raiz do número de anos, porque anos bons e
      ruins se compensam (ações);
    - `constante`: não cai, porque a média do CDI ou do IPCA num prazo longo
      não fica mais previsível que num prazo curto (renda fixa).
    """

    premio_sobre_cdi_aa: float
    volatilidade_aa: float
    escala_prazo: str


@dataclass(frozen=True)
class PremissasCenarios:
    """Parâmetros da simulação de cenários de retorno.

    `correlacoes[i][j]` é a correlação entre as classes `ordem[i]` e `ordem[j]`.
    """

    quantidade: int
    semente: int
    graus_liberdade: float
    classes: dict[str, ClasseCenario]
    ordem: tuple[str, ...]
    correlacoes: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class Premissas:
    """Todas as premissas de mercado, impostos, FGC e perfis."""

    data_referencia: date
    juro_real_aa: float
    dias_uteis_por_ano: int
    dias_corridos_por_ano: int
    impostos: PremissasImpostos
    fgc: PremissasFGC
    perfis: dict[str, Perfil]
    otimizacao: PremissasOtimizacao
    cenarios: PremissasCenarios

    def perfil(self, nome: str) -> Perfil:
        """Devolve os limites do perfil pedido, ou erro se ele não existir."""
        _exigir_opcao(nome, tuple(self.perfis), "perfil do cliente")
        return self.perfis[nome]


# ---------------------------------------------------------------------------
# Leitura dos arquivos
# ---------------------------------------------------------------------------

def carregar_cliente(caminho: Path) -> Cliente:
    """Lê o arquivo JSON de um cliente e devolve um `Cliente`."""
    return cliente_de_dados(_ler_json(caminho))


def cliente_de_dados(dados: Any) -> Cliente:
    """Monta um `Cliente` a partir de um dicionário no mesmo formato do `cliente.json`."""
    _exigir(isinstance(dados, dict), "cliente: precisa ser um objeto JSON.")
    metas = _campo(dados, "metas", "cliente")
    _exigir(isinstance(metas, list), "cliente: 'metas' precisa ser uma lista.")
    return Cliente(
        identificador=_texto(_campo(dados, "identificador", "cliente"), "cliente, identificador"),
        perfil=_texto(_campo(dados, "perfil", "cliente"), "cliente, perfil"),
        renda_mensal=_numero(_campo(dados, "renda_mensal", "cliente"), "cliente, renda_mensal"),
        dividas_caras=_booleano(_campo(dados, "dividas_caras", "cliente"), "cliente, dividas_caras"),
        metas=tuple(_ler_meta(item, posicao) for posicao, item in enumerate(metas, start=1)),
        empregador=_texto_ou_nulo(_campo(dados, "empregador", "cliente"), "cliente, empregador"),
        setor_trabalho=_texto_ou_nulo(_campo(dados, "setor_trabalho", "cliente"), "cliente, setor_trabalho"),
    )


def carregar_prateleira(caminho: Path) -> list[Produto]:
    """Lê o CSV da prateleira e devolve a lista de produtos.

    Linhas que começam com `#` são comentários (como o aviso de dados
    fictícios) e são ignoradas.
    """
    produtos = [_ler_produto(linha) for linha in ler_csv(caminho, COLUNAS_PRATELEIRA, "prateleira")]
    _exigir(len(produtos) > 0, "prateleira: nenhum produto encontrado.")
    nomes = [produto.nome for produto in produtos]
    _exigir(len(nomes) == len(set(nomes)), "prateleira: há produtos com o mesmo nome.")
    return produtos


def carregar_acoes(caminho: Path) -> list[Produto]:
    """Lê o CSV com a lista de ações aprovada pela área de análise da corretora.

    Cada ação vira um `Produto` de renda variável, tributado como ações, com
    liquidez diária e sem garantia do FGC. As colunas de risco vêm da análise:
    - `beta`: quanto a ação acompanha a bolsa (1 = igual à bolsa);
    - `volatilidade_propria_aa`: oscilação própria da empresa, além da bolsa;
    - `alfa_aa`: retorno esperado a mais que a bolsa explica (ex.: pelo preço-alvo);
    - `aplicacao_minima`: preço de uma ação (mercado fracionário).
    """
    acoes = []
    for linha in ler_csv(caminho, COLUNAS_ACOES, "ações"):
        ticker = (linha["ticker"] or "").strip()
        empresa = (linha["empresa"] or "").strip()
        contexto = f"ações, '{ticker}'"
        _exigir(bool(ticker) and bool(empresa), "ações: toda linha precisa de ticker e empresa.")
        acoes.append(Produto(
            nome=f"{ticker} ({empresa})",
            emissor=empresa,
            conglomerado=(linha["conglomerado"] or "").strip() or empresa,
            indexador="variavel",
            taxa=0.0,
            taxa_adm_aa=0.0,
            vencimento_meses=None,
            carencia_meses=0,
            liquidez_diaria=True,
            tributacao="acoes",
            classe=CLASSE_RENDA_VARIAVEL,
            risco=inteiro_csv(linha["risco"], f"{contexto}, risco"),
            coberto_fgc=False,
            aplicacao_minima=numero_csv(linha["aplicacao_minima"], f"{contexto}, aplicacao_minima"),
            setor=(linha["setor"] or "").strip(),
            ticker=ticker,
            beta=numero_csv(linha["beta"], f"{contexto}, beta"),
            volatilidade_propria_aa=numero_csv(linha["volatilidade_propria_aa"], f"{contexto}, volatilidade_propria_aa"),
            alfa_aa=numero_csv(linha["alfa_aa"], f"{contexto}, alfa_aa"),
        ))
    tickers = [acao.ticker for acao in acoes]
    _exigir(len(tickers) == len(set(tickers)), "ações: há tickers repetidos.")
    return acoes


def carregar_premissas(caminho: Path) -> Premissas:
    """Lê o JSON de premissas e devolve um objeto `Premissas`."""
    dados = _ler_json(caminho)
    data_texto = _texto(_campo(dados, "data_referencia", "premissas"), "premissas, data_referencia")
    try:
        data_referencia = date.fromisoformat(data_texto)
    except ValueError as erro:
        raise ErroDeDados(
            f"premissas: data_referencia '{data_texto}' não está no formato AAAA-MM-DD."
        ) from erro
    return Premissas(
        data_referencia=data_referencia,
        juro_real_aa=_numero(_campo(dados, "juro_real_aa", "premissas"), "premissas, juro_real_aa"),
        dias_uteis_por_ano=_inteiro(_campo(dados, "dias_uteis_por_ano", "premissas"), "premissas, dias_uteis_por_ano"),
        dias_corridos_por_ano=_inteiro(_campo(dados, "dias_corridos_por_ano", "premissas"), "premissas, dias_corridos_por_ano"),
        impostos=_ler_impostos(_campo(dados, "impostos", "premissas")),
        fgc=_ler_fgc(_campo(dados, "fgc", "premissas")),
        perfis=_ler_perfis(_campo(dados, "perfis", "premissas")),
        otimizacao=_ler_otimizacao(_campo(dados, "otimizacao", "premissas")),
        cenarios=_ler_cenarios(_campo(dados, "cenarios", "premissas")),
    )


def ler_csv(caminho: Path, colunas: tuple[str, ...], contexto: str) -> list[dict[str, str]]:
    """Lê um CSV em UTF-8, pulando linhas de comentário (`#`), e confere as colunas.

    Devolve uma lista de linhas, cada uma como dicionário coluna -> texto.
    """
    with Path(caminho).open(encoding="utf-8", newline="") as arquivo:
        linhas = [
            linha for linha in arquivo
            if not linha.lstrip().startswith(PREFIXO_COMENTARIO)
        ]
    leitor = csv.DictReader(linhas)
    faltando = [c for c in colunas if c not in (leitor.fieldnames or [])]
    _exigir(not faltando, f"{contexto}: colunas ausentes: {', '.join(faltando)}.")
    return list(leitor)


# ---------------------------------------------------------------------------
# Leitura das partes (funções internas)
# ---------------------------------------------------------------------------

def _ler_meta(dados: Any, posicao: int) -> Meta:
    """Converte um item da lista de metas do JSON em `Meta`."""
    contexto = f"meta nº {posicao}"
    _exigir(isinstance(dados, dict), f"{contexto}: precisa ser um objeto JSON.")
    return Meta(
        nome=_texto(_campo(dados, "nome", contexto), f"{contexto}, nome"),
        valor_atual=_numero(_campo(dados, "valor_atual", contexto), f"{contexto}, valor_atual"),
        valor_alvo=_numero(_campo(dados, "valor_alvo", contexto), f"{contexto}, valor_alvo"),
        prazo_meses=_inteiro(_campo(dados, "prazo_meses", contexto), f"{contexto}, prazo_meses"),
        reserva_emergencia=_booleano(
            _campo(dados, "reserva_emergencia", contexto), f"{contexto}, reserva_emergencia"
        ),
    )


def _ler_produto(linha: dict[str, str]) -> Produto:
    """Converte uma linha do CSV da prateleira em `Produto`."""
    nome = (linha["nome"] or "").strip()
    contexto = f"prateleira, produto '{nome}'"
    vencimento = (linha["vencimento_meses"] or "").strip()
    return Produto(
        nome=nome,
        emissor=(linha["emissor"] or "").strip(),
        conglomerado=(linha["conglomerado"] or "").strip(),
        indexador=(linha["indexador"] or "").strip(),
        taxa=numero_csv(linha["taxa"], f"{contexto}, taxa"),
        taxa_adm_aa=numero_csv(linha["taxa_adm_aa"], f"{contexto}, taxa_adm_aa"),
        vencimento_meses=inteiro_csv(vencimento, f"{contexto}, vencimento_meses") if vencimento else None,
        carencia_meses=inteiro_csv(linha["carencia_meses"], f"{contexto}, carencia_meses"),
        liquidez_diaria=_booleano_csv(linha["liquidez_diaria"], f"{contexto}, liquidez_diaria"),
        tributacao=(linha["tributacao"] or "").strip(),
        classe=(linha["classe"] or "").strip(),
        risco=inteiro_csv(linha["risco"], f"{contexto}, risco"),
        coberto_fgc=_booleano_csv(linha["coberto_fgc"], f"{contexto}, coberto_fgc"),
        aplicacao_minima=numero_csv(linha["aplicacao_minima"], f"{contexto}, aplicacao_minima"),
        setor=(linha["setor"] or "").strip(),
    )


def _ler_faixas(dados: Any, contexto: str) -> tuple[FaixaIR, ...]:
    """Lê uma tabela de IR e confere se as faixas estão em ordem crescente."""
    _exigir(isinstance(dados, list) and len(dados) > 0, f"{contexto}: precisa ser uma lista com faixas.")
    faixas = []
    for posicao, item in enumerate(dados, start=1):
        local = f"{contexto}, faixa {posicao}"
        _exigir(isinstance(item, dict), f"{local}: precisa ser um objeto JSON.")
        ate_dias = _campo(item, "ate_dias", local)
        faixas.append(FaixaIR(
            ate_dias=None if ate_dias is None else _inteiro(ate_dias, f"{local}, ate_dias"),
            aliquota=_fracao(_campo(item, "aliquota", local), f"{local}, aliquota"),
        ))
    limites = [faixa.ate_dias for faixa in faixas]
    _exigir(limites[-1] is None, f"{contexto}: a última faixa precisa ter ate_dias = null.")
    _exigir(None not in limites[:-1], f"{contexto}: só a última faixa pode ter ate_dias = null.")
    _exigir(limites[:-1] == sorted(set(limites[:-1])), f"{contexto}: as faixas precisam estar em ordem crescente.")
    return tuple(faixas)


def _ler_impostos(dados: Any) -> PremissasImpostos:
    """Lê a seção `impostos` das premissas."""
    contexto = "premissas, impostos"
    _exigir(isinstance(dados, dict), f"{contexto}: precisa ser um objeto JSON.")
    come_cotas = _campo(dados, "come_cotas", contexto)
    local = f"{contexto}, come_cotas"
    _exigir(isinstance(come_cotas, dict), f"{local}: precisa ser um objeto JSON.")
    meses = _campo(come_cotas, "meses", local)
    _exigir(isinstance(meses, list), f"{local}, meses: precisa ser uma lista.")
    meses_lidos = tuple(_inteiro(mes, f"{local}, meses") for mes in meses)
    _exigir(all(1 <= mes <= 12 for mes in meses_lidos), f"{local}, meses: use números de 1 a 12.")
    return PremissasImpostos(
        tabela_regressiva=_ler_faixas(_campo(dados, "tabela_regressiva", contexto), f"{contexto}, tabela_regressiva"),
        tabela_fundo_curto_prazo=_ler_faixas(
            _campo(dados, "tabela_fundo_curto_prazo", contexto), f"{contexto}, tabela_fundo_curto_prazo"
        ),
        aliquota_acoes=_fracao(_campo(dados, "aliquota_acoes", contexto), f"{contexto}, aliquota_acoes"),
        meses_come_cotas=meses_lidos,
        come_cotas_longo_prazo=_fracao(_campo(come_cotas, "aliquota_longo_prazo", local), f"{local}, aliquota_longo_prazo"),
        come_cotas_curto_prazo=_fracao(_campo(come_cotas, "aliquota_curto_prazo", local), f"{local}, aliquota_curto_prazo"),
    )


def _ler_fgc(dados: Any) -> PremissasFGC:
    """Lê a seção `fgc` das premissas."""
    contexto = "premissas, fgc"
    _exigir(isinstance(dados, dict), f"{contexto}: precisa ser um objeto JSON.")
    fgc = PremissasFGC(
        limite_por_conglomerado=_numero(_campo(dados, "limite_por_conglomerado", contexto), f"{contexto}, limite_por_conglomerado"),
        teto_global=_numero(_campo(dados, "teto_global", contexto), f"{contexto}, teto_global"),
        janela_teto_anos=_inteiro(_campo(dados, "janela_teto_anos", contexto), f"{contexto}, janela_teto_anos"),
    )
    _exigir(fgc.limite_por_conglomerado > 0, f"{contexto}: limite_por_conglomerado precisa ser positivo.")
    _exigir(fgc.teto_global >= fgc.limite_por_conglomerado, f"{contexto}: teto_global não pode ser menor que o limite por conglomerado.")
    _exigir(fgc.janela_teto_anos >= 1, f"{contexto}: janela_teto_anos precisa ser de pelo menos 1.")
    return fgc


def _ler_perfis(dados: Any) -> dict[str, Perfil]:
    """Lê a seção `perfis` das premissas."""
    _exigir(isinstance(dados, dict) and len(dados) > 0, "premissas, perfis: precisa ter pelo menos um perfil.")
    perfis = {}
    for nome, valores in dados.items():
        local = f"premissas, perfil '{nome}'"
        _exigir(isinstance(valores, dict), f"{local}: precisa ser um objeto JSON.")
        risco_maximo = _inteiro(_campo(valores, "risco_maximo", local), f"{local}, risco_maximo")
        _exigir(
            RISCO_MINIMO <= risco_maximo <= RISCO_MAXIMO,
            f"{local}: risco_maximo precisa estar entre {RISCO_MINIMO} e {RISCO_MAXIMO}.",
        )
        perfis[nome] = Perfil(
            nome=nome,
            risco_maximo=risco_maximo,
            teto_renda_variavel=_fracao(_campo(valores, "teto_renda_variavel", local), f"{local}, teto_renda_variavel"),
            cvar_maximo=_fracao(_campo(valores, "cvar_maximo", local), f"{local}, cvar_maximo"),
        )
    return perfis


def _ler_otimizacao(dados: Any) -> PremissasOtimizacao:
    """Lê a seção `otimizacao` das premissas."""
    contexto = "premissas, otimizacao"
    _exigir(isinstance(dados, dict), f"{contexto}: precisa ser um objeto JSON.")
    sem_limite = _campo(dados, "indexadores_sem_limite", contexto)
    _exigir(isinstance(sem_limite, list), f"{contexto}, indexadores_sem_limite: precisa ser uma lista.")
    for indexador in sem_limite:
        _exigir_opcao(indexador, INDEXADORES, f"{contexto}, indexadores_sem_limite")
    otimizacao = PremissasOtimizacao(
        nivel_cvar=_fracao(_campo(dados, "nivel_cvar", contexto), f"{contexto}, nivel_cvar"),
        limite_por_produto=_fracao(_campo(dados, "limite_por_produto", contexto), f"{contexto}, limite_por_produto"),
        indexadores_sem_limite=tuple(sem_limite),
        prazo_minimo_renda_variavel_meses=_inteiro(
            _campo(dados, "prazo_minimo_renda_variavel_meses", contexto), f"{contexto}, prazo_minimo_renda_variavel_meses"
        ),
        prazo_minimo_risco_3_meses=_inteiro(
            _campo(dados, "prazo_minimo_risco_3_meses", contexto), f"{contexto}, prazo_minimo_risco_3_meses"
        ),
        reamostragens=_inteiro(_campo(dados, "reamostragens", contexto), f"{contexto}, reamostragens"),
        limite_setor_do_trabalho=_fracao(
            _campo(dados, "limite_setor_do_trabalho", contexto), f"{contexto}, limite_setor_do_trabalho"
        ),
        limite_por_acao=_fracao(_campo(dados, "limite_por_acao", contexto), f"{contexto}, limite_por_acao"),
        peso_minimo_por_produto=_fracao(
            _campo(dados, "peso_minimo_por_produto", contexto), f"{contexto}, peso_minimo_por_produto"
        ),
        limite_por_conglomerado_na_meta=_fracao(
            _campo(dados, "limite_por_conglomerado_na_meta", contexto), f"{contexto}, limite_por_conglomerado_na_meta"
        ),
        conglomerados_sem_limite=tuple(
            _texto(nome, f"{contexto}, conglomerados_sem_limite")
            for nome in _lista(_campo(dados, "conglomerados_sem_limite", contexto), f"{contexto}, conglomerados_sem_limite")
        ),
        nivel_crescimento=_fracao(_campo(dados, "nivel_crescimento", contexto), f"{contexto}, nivel_crescimento"),
    )
    _exigir(0 < otimizacao.nivel_cvar < 1, f"{contexto}: nivel_cvar precisa estar entre 0 e 1.")
    _exigir(otimizacao.limite_por_produto > 0, f"{contexto}: limite_por_produto precisa ser positivo.")
    _exigir(otimizacao.reamostragens >= 1, f"{contexto}: reamostragens precisa ser de pelo menos 1.")
    return otimizacao


def _ler_cenarios(dados: Any) -> PremissasCenarios:
    """Lê a seção `cenarios` e confere a matriz de correlações."""
    contexto = "premissas, cenarios"
    _exigir(isinstance(dados, dict), f"{contexto}: precisa ser um objeto JSON.")
    classes_json = _campo(dados, "classes", contexto)
    _exigir(isinstance(classes_json, dict), f"{contexto}, classes: precisa ser um objeto JSON.")
    _exigir(
        set(classes_json) == set(CLASSES),
        f"{contexto}, classes: informe exatamente as classes {', '.join(CLASSES)}.",
    )
    classes = {}
    for nome, valores in classes_json.items():
        local = f"{contexto}, classe '{nome}'"
        _exigir(isinstance(valores, dict), f"{local}: precisa ser um objeto JSON.")
        classes[nome] = ClasseCenario(
            premio_sobre_cdi_aa=_numero(_campo(valores, "premio_sobre_cdi_aa", local), f"{local}, premio_sobre_cdi_aa"),
            volatilidade_aa=_numero(_campo(valores, "volatilidade_aa", local), f"{local}, volatilidade_aa"),
            escala_prazo=_texto(_campo(valores, "escala_prazo", local), f"{local}, escala_prazo"),
        )
        _exigir_opcao(classes[nome].escala_prazo, ESCALAS_PRAZO, f"{local}, escala_prazo")
        _exigir(classes[nome].volatilidade_aa >= 0, f"{local}: volatilidade_aa não pode ser negativa.")

    correlacoes = _campo(dados, "correlacoes", contexto)
    local = f"{contexto}, correlacoes"
    _exigir(isinstance(correlacoes, dict), f"{local}: precisa ser um objeto JSON.")
    ordem = tuple(_campo(correlacoes, "ordem", local))
    _exigir(sorted(ordem) == sorted(CLASSES), f"{local}, ordem: liste cada classe uma única vez.")
    matriz = _ler_matriz_correlacao(_campo(correlacoes, "matriz", local), len(ordem), local)

    cenarios = PremissasCenarios(
        quantidade=_inteiro(_campo(dados, "quantidade", contexto), f"{contexto}, quantidade"),
        semente=_inteiro(_campo(dados, "semente", contexto), f"{contexto}, semente"),
        graus_liberdade=_numero(_campo(dados, "graus_liberdade", contexto), f"{contexto}, graus_liberdade"),
        classes=classes,
        ordem=ordem,
        correlacoes=matriz,
    )
    _exigir(cenarios.quantidade >= 1, f"{contexto}: quantidade precisa ser de pelo menos 1.")
    _exigir(cenarios.graus_liberdade > 2, f"{contexto}: graus_liberdade precisa ser maior que 2 (variância finita).")
    return cenarios


def _ler_matriz_correlacao(dados: Any, tamanho: int, contexto: str) -> tuple[tuple[float, ...], ...]:
    """Lê a matriz de correlações e confere se é quadrada, simétrica e com 1 na diagonal."""
    local = f"{contexto}, matriz"
    _exigir(
        isinstance(dados, list) and len(dados) == tamanho
        and all(isinstance(linha, list) and len(linha) == tamanho for linha in dados),
        f"{local}: precisa ser uma tabela {tamanho} x {tamanho}.",
    )
    matriz = tuple(tuple(_numero(valor, local) for valor in linha) for linha in dados)
    for i in range(tamanho):
        _exigir(matriz[i][i] == 1, f"{local}: a diagonal precisa ser 1.")
        for j in range(tamanho):
            _exigir(-1 <= matriz[i][j] <= 1, f"{local}: correlações precisam estar entre -1 e 1.")
            _exigir(matriz[i][j] == matriz[j][i], f"{local}: a matriz precisa ser simétrica.")
    return matriz


# ---------------------------------------------------------------------------
# Conversões e conferências (funções internas)
# ---------------------------------------------------------------------------

def _exigir(condicao: bool, mensagem: str) -> None:
    """Levanta `ErroDeDados` com a mensagem se a condição for falsa."""
    if not condicao:
        raise ErroDeDados(mensagem)


def _exigir_opcao(valor: str, opcoes: tuple[str, ...], contexto: str) -> None:
    """Confere se o valor é uma das opções aceitas."""
    _exigir(
        valor in opcoes,
        f"{contexto}: valor '{valor}' não reconhecido. Use um destes: {', '.join(opcoes)}.",
    )


def _ler_json(caminho: Path) -> dict[str, Any]:
    """Abre um arquivo JSON em UTF-8 e confere se ele é um objeto."""
    caminho = Path(caminho)
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
    except json.JSONDecodeError as erro:
        raise ErroDeDados(
            f"{caminho.name}: JSON inválido na linha {erro.lineno}, coluna {erro.colno}."
        ) from erro
    _exigir(isinstance(dados, dict), f"{caminho.name}: o conteúdo precisa ser um objeto JSON.")
    return dados


def _campo(dados: dict[str, Any], chave: str, contexto: str) -> Any:
    """Pega um campo obrigatório de um objeto JSON."""
    _exigir(chave in dados, f"{contexto}: campo obrigatório '{chave}' ausente.")
    return dados[chave]


def _texto(valor: Any, contexto: str) -> str:
    """Confere se o valor é texto."""
    _exigir(isinstance(valor, str), f"{contexto}: precisa ser um texto.")
    return valor


def _lista(valor: Any, contexto: str) -> list[Any]:
    """Confere se o valor é uma lista."""
    _exigir(isinstance(valor, list), f"{contexto}: precisa ser uma lista.")
    return valor


def _texto_ou_nulo(valor: Any, contexto: str) -> str | None:
    """Confere se o valor é texto ou null."""
    _exigir(valor is None or isinstance(valor, str), f"{contexto}: precisa ser um texto ou null.")
    return valor


def _numero(valor: Any, contexto: str) -> float:
    """Confere se o valor é um número (e não verdadeiro/falso)."""
    _exigir(
        isinstance(valor, (int, float)) and not isinstance(valor, bool),
        f"{contexto}: precisa ser um número.",
    )
    return float(valor)


def _inteiro(valor: Any, contexto: str) -> int:
    """Confere se o valor é um número inteiro."""
    _exigir(
        isinstance(valor, int) and not isinstance(valor, bool),
        f"{contexto}: precisa ser um número inteiro.",
    )
    return valor


def _fracao(valor: Any, contexto: str) -> float:
    """Confere se o valor é um número entre 0 e 1 (por exemplo, 0.15 = 15%)."""
    numero = _numero(valor, contexto)
    _exigir(0 <= numero <= 1, f"{contexto}: precisa estar entre 0 e 1 (ex.: 0.15 para 15%).")
    return numero


def _booleano(valor: Any, contexto: str) -> bool:
    """Confere se o valor é true ou false."""
    _exigir(isinstance(valor, bool), f"{contexto}: precisa ser true ou false.")
    return valor


def numero_csv(texto: str | None, contexto: str) -> float:
    """Converte um texto do CSV em número (use ponto como separador decimal)."""
    try:
        return float((texto or "").strip())
    except ValueError as erro:
        raise ErroDeDados(f"{contexto}: '{texto}' não é um número (use ponto, ex.: 0.15).") from erro


def inteiro_csv(texto: str | None, contexto: str) -> int:
    """Converte um texto do CSV em número inteiro."""
    try:
        return int((texto or "").strip())
    except ValueError as erro:
        raise ErroDeDados(f"{contexto}: '{texto}' não é um número inteiro.") from erro


def _booleano_csv(texto: str | None, contexto: str) -> bool:
    """Converte 'sim'/'não' (ou true/false, 1/0) do CSV em verdadeiro/falso."""
    normalizado = (texto or "").strip().lower()
    if normalizado in TEXTOS_SIM:
        return True
    if normalizado in TEXTOS_NAO:
        return False
    raise ErroDeDados(f"{contexto}: '{texto}' não é 'sim' nem 'não'.")
