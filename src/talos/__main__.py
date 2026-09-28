"""Linha de comando do Talos: `python -m talos <comando>`.

    python -m talos recomendar --cliente exemplos/cliente.json --prateleira exemplos/prateleira.csv
    python -m talos validar --cliente exemplos/cliente.json --prateleira exemplos/prateleira.csv
    python -m talos servir          (API HTTP em http://127.0.0.1:8000)
    python -m talos testar-api      (chama a API como uma corretora faria)

Por padrão, as premissas e a curva de juros são lidas da mesma pasta da
prateleira (`premissas.json` e `curva_di.csv`), e o registro de auditoria
é gravado em `saidas/`.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import textwrap
from collections.abc import Sequence
from pathlib import Path

from talos import AVISO_SIMULACAO, __version__
from talos.api import HOST_PADRAO, PORTA_PADRAO, carregar_motor, criar_servidor, testar_api
from talos.auditoria import PASTA_PADRAO, registrar
from talos.explicacao import (
    Explicacao,
    ExplicacaoMeta,
    explicar,
    formatar_percentual,
    formatar_prazo,
    formatar_probabilidade,
)
from talos.fgc import formatar_reais
from talos.mercado import CurvaDI, carregar_curva_di
from talos.modelos import (
    ErroDeDados,
    Premissas,
    Produto,
    carregar_acoes,
    carregar_cliente,
    carregar_prateleira,
    carregar_premissas,
)
from talos.otimizador import (
    ATINGIVEL,
    DIFICIL,
    ESTRATEGIAS,
    MENOR_RISCO,
    Recomendacao,
    ResultadoMeta,
    otimizar_cliente,
)
from talos.validacao import ComparacaoMeta, conclusao, percentual_com_sinal, resumo_validacao, validar

LARGURA = 88
CODIGO_ERRO_DE_DADOS = 2
NOME_PREMISSAS = "premissas.json"
NOME_CURVA = "curva_di.csv"
NOME_ACOES = "acoes.csv"
PASTA_EXEMPLOS = Path("exemplos")
ROTULOS_SITUACAO = {ATINGIVEL: "ATINGÍVEL", DIFICIL: "DIFÍCIL"}
ROTULO_SEM_SOLUCAO = "SEM SOLUÇÃO"


def usar_utf8_no_terminal() -> None:
    """Faz a saída do terminal usar UTF-8 em todos os sistemas.

    Sem isso, no Windows os acentos saem trocados quando a saída é
    redirecionada para um arquivo ou outro programa.
    """
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8", errors="replace")


def criar_parser() -> argparse.ArgumentParser:
    """Monta o leitor de argumentos com os comandos `recomendar`, `validar`, `servir` e `testar-api`."""
    parser = argparse.ArgumentParser(
        prog="talos",
        description="Talos: carteiras de investimento por meta (simulação).",
    )
    parser.add_argument("--versao", action="version", version=f"talos {__version__}")
    comandos = parser.add_subparsers(dest="comando", required=True)
    for nome, ajuda in (
        ("recomendar", "gera uma carteira para cada meta do cliente"),
        ("validar", "compara a carteira com referências simples"),
    ):
        sub = comandos.add_parser(nome, help=ajuda)
        sub.add_argument("--cliente", required=True, type=Path, help="arquivo JSON do cliente")
        sub.add_argument("--prateleira", required=True, type=Path, help="arquivo CSV de produtos")
        sub.add_argument("--premissas", type=Path, help=f"arquivo de premissas (padrão: {NOME_PREMISSAS} na pasta da prateleira)")
        sub.add_argument("--curva", type=Path, help=f"curva de DI (padrão: {NOME_CURVA} na pasta da prateleira)")
        sub.add_argument("--semente", type=int, help="semente dos cenários (padrão: a das premissas)")
        sub.add_argument("--cenarios", type=int, help="quantidade de cenários (padrão: a das premissas)")
        sub.add_argument("--reamostragem", action="store_true", help="usa a reamostragem de Michaud (mais lento)")
        sub.add_argument("--saidas", type=Path, default=PASTA_PADRAO, help="pasta do registro de auditoria (padrão: saidas)")
        sub.add_argument("--acoes", type=Path, help=f"lista de ações da corretora (padrão: {NOME_ACOES} na pasta da prateleira, se existir)")
        sub.add_argument(
            "--estrategia", choices=ESTRATEGIAS, default=MENOR_RISCO,
            help="menor_risco: menor risco que atinge a meta (padrão); crescimento: mais crescimento dentro do perfil",
        )

    servir = comandos.add_parser("servir", help="liga a API HTTP para corretoras (só neste computador)")
    servir.add_argument("--host", default=HOST_PADRAO, help=f"endereço (padrão: {HOST_PADRAO}, só este computador)")
    servir.add_argument("--porta", type=int, default=PORTA_PADRAO, help=f"porta (padrão: {PORTA_PADRAO})")
    servir.add_argument("--dados", type=Path, default=PASTA_EXEMPLOS, help="pasta com prateleira, premissas e curva (padrão: exemplos)")
    servir.add_argument("--saidas", type=Path, default=PASTA_PADRAO, help="pasta do registro de auditoria (padrão: saidas)")

    teste = comandos.add_parser("testar-api", help="chama a API com o cliente de exemplo, como uma corretora faria")
    teste.add_argument("--url", default=f"http://{HOST_PADRAO}:{PORTA_PADRAO}", help="endereço da API")
    teste.add_argument("--cliente", type=Path, default=PASTA_EXEMPLOS / "cliente.json", help="JSON do cliente enviado")
    teste.add_argument("--cenarios", type=int, help="quantidade de cenários (padrão: a das premissas do servidor)")
    teste.add_argument("--estrategia", choices=ESTRATEGIAS, help="estratégia enviada à API (padrão: a do servidor)")
    return parser


def main(argumentos: Sequence[str] | None = None) -> int:
    """Executa a linha de comando e devolve o código de saída (0 = sucesso, 2 = erro nos dados)."""
    usar_utf8_no_terminal()
    args = criar_parser().parse_args(argumentos)
    try:
        if args.comando == "servir":
            return servir(args)
        if args.comando == "testar-api":
            print(testar_api(args.url, args.cliente, args.cenarios, args.estrategia))
        else:
            print(recomendar(args) if args.comando == "recomendar" else comparar(args))
    except urllib.error.URLError as erro:
        print(
            f"Não foi possível falar com a API em {args.url} ({erro.reason}). "
            "Ligue o serviço em outro terminal com: python -m talos servir",
            file=sys.stderr,
        )
        return CODIGO_ERRO_DE_DADOS
    except FileNotFoundError as erro:
        print(f"Arquivo não encontrado: {erro.filename}", file=sys.stderr)
        return CODIGO_ERRO_DE_DADOS
    except ErroDeDados as erro:
        print(f"Erro nos dados: {erro}", file=sys.stderr)
        return CODIGO_ERRO_DE_DADOS
    return 0


def servir(args: argparse.Namespace) -> int:
    """Liga a API e atende até o usuário apertar Ctrl+C."""
    motor = carregar_motor(args.dados, args.saidas)
    try:
        servidor = criar_servidor(motor, args.host, args.porta)
    except OSError as erro:
        print(f"Não foi possível usar a porta {args.porta}: {erro}. Tente outra com --porta.", file=sys.stderr)
        return CODIGO_ERRO_DE_DADOS
    host, porta = servidor.server_address[:2]
    print(f"Talos {__version__} atendendo em http://{host}:{porta}")
    print("Rotas: GET /saude · GET /prateleira · POST /recomendacoes · POST /validacoes")
    print("Teste em outro terminal com: python -m talos testar-api")
    print("Para desligar, aperte Ctrl+C.", flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServiço desligado.")
    finally:
        servidor.server_close()
    return 0


def recomendar(args: argparse.Namespace) -> str:
    """Lê as entradas, otimiza, explica, grava a auditoria e devolve o relatório em texto."""
    premissas, curva, produtos, recomendacao = _otimizar(args)
    explicacao = explicar(recomendacao, premissas)
    caminho = registrar(recomendacao, explicacao, produtos, curva, premissas, args.saidas)
    return formatar_relatorio(recomendacao, explicacao, _cabecalho(args, premissas, recomendacao), caminho)


def comparar(args: argparse.Namespace) -> str:
    """Otimiza e compara cada meta com 100% do CDI e com pesos iguais, nos mesmos cenários."""
    premissas, curva, produtos, recomendacao = _otimizar(args)
    comparacoes = validar(recomendacao, produtos, premissas, curva, args.cenarios)
    return formatar_validacao(recomendacao, comparacoes, _cabecalho(args, premissas, recomendacao))


def _otimizar(args: argparse.Namespace) -> tuple[Premissas, CurvaDI, list[Produto], Recomendacao]:
    """Carrega as entradas e otimiza todas as metas do cliente."""
    pasta = args.prateleira.parent
    premissas = carregar_premissas(args.premissas or pasta / NOME_PREMISSAS)
    curva = carregar_curva_di(args.curva or pasta / NOME_CURVA, premissas)
    produtos = carregar_prateleira(args.prateleira) + _carregar_acoes_se_houver(args.acoes, pasta / NOME_ACOES)
    cliente = carregar_cliente(args.cliente)
    premissas.perfil(cliente.perfil)
    if args.cenarios is not None and args.cenarios < 1:
        raise ErroDeDados("--cenarios precisa ser de pelo menos 1.")
    recomendacao = otimizar_cliente(
        cliente, produtos, premissas, curva,
        reamostragem=args.reamostragem, semente=args.semente, quantidade=args.cenarios,
        estrategia=args.estrategia,
    )
    return premissas, curva, produtos, recomendacao


def _carregar_acoes_se_houver(pedido: Path | None, padrao: Path) -> list[Produto]:
    """Lê a lista de ações pedida; sem pedido, usa a da pasta da prateleira se ela existir."""
    if pedido is not None:
        return carregar_acoes(pedido)
    return carregar_acoes(padrao) if padrao.exists() else []


def _cabecalho(args: argparse.Namespace, premissas: Premissas, recomendacao: Recomendacao) -> str:
    """Linha com a data das premissas, a quantidade de cenários e a semente."""
    quantidade = f"{args.cenarios or premissas.cenarios.quantidade:,}".replace(",", ".")
    return (
        f"Premissas de {premissas.data_referencia:%d/%m/%Y} · {quantidade} cenários · "
        f"semente {recomendacao.semente} · estratégia {recomendacao.estrategia.replace('_', ' ')}"
        + (" · com reamostragem" if recomendacao.reamostragem else "")
    )


# ---------------------------------------------------------------------------
# Relatório em texto
# ---------------------------------------------------------------------------

def formatar_relatorio(
    recomendacao: Recomendacao, explicacao: Explicacao, cabecalho: str, caminho_auditoria: Path
) -> str:
    """Monta o relatório completo: cabeçalho, avisos gerais, uma seção por meta e o aviso final."""
    linhas = [
        f"TALOS · Recomendação para {recomendacao.cliente.identificador} (perfil {recomendacao.perfil.nome})",
        cabecalho,
    ]
    for frase in explicacao.antes_de_tudo:
        linhas += [""] + _paragrafo(frase, "» ")
    total = len(recomendacao.resultados)
    for numero, (resultado, texto) in enumerate(zip(recomendacao.resultados, explicacao.metas), start=1):
        linhas += [""] + _secao_meta(resultado, texto, numero, total)
    linhas += ["", "=" * LARGURA, f"Registro de auditoria: {caminho_auditoria}", ""]
    linhas += _paragrafo(explicacao.aviso)
    return "\n".join(linhas)


def _secao_meta(resultado: ResultadoMeta, texto: ExplicacaoMeta, numero: int, total: int) -> list[str]:
    """Seção de uma meta: título, números principais, tabela da carteira e explicações."""
    meta = resultado.meta
    rotulo = ROTULOS_SITUACAO.get(resultado.situacao, ROTULO_SEM_SOLUCAO)
    titulo = f"Meta {numero} de {total}: {meta.nome}"
    linhas = [
        "=" * LARGURA,
        f"{titulo}{rotulo:>{LARGURA - len(titulo)}}",
        f"{formatar_reais(meta.valor_atual)} → {formatar_reais(meta.valor_alvo)} em {formatar_prazo(meta.prazo_meses)}"
        + (" · reserva de emergência" if meta.reserva_emergencia else ""),
        "-" * LARGURA,
    ]
    if resultado.alocacoes:
        linhas += _paragrafo(
            f"Precisa render {formatar_percentual(resultado.retorno_necessario_aa)} a.a. · "
            f"carteira rende {formatar_percentual(resultado.retorno_esperado_aa)} a.a. · "
            f"chance de atingir {formatar_probabilidade(resultado.probabilidade_sucesso)}"
        )
        if resultado.valor_final_por_percentil:
            valores = resultado.valor_final_por_percentil
            linhas += _paragrafo(
                f"Valor no fim do prazo: 5% piores cenários até {formatar_reais(valores[5])} · "
                f"cenário do meio {formatar_reais(valores[50])} · "
                f"5% melhores acima de {formatar_reais(valores[95])}"
            )
        linhas += [""] + _tabela_carteira(resultado)
    linhas.append("")
    for frase in texto.resumo:
        linhas += _paragrafo(frase, "• ")
    for nome, frases in texto.produtos.items():
        linhas += ["", f"  {nome}"]
        for frase in frases[1:]:
            linhas += _paragrafo(frase, "    - ")
    if texto.nao_usados:
        linhas += ["", "  Não entraram nesta meta:"]
        for frase in texto.nao_usados:
            linhas += _paragrafo(frase, "    · ")
    return linhas


def formatar_validacao(
    recomendacao: Recomendacao, comparacoes: Sequence[ComparacaoMeta], cabecalho: str
) -> str:
    """Relatório da validação: uma tabela por meta, a conclusão, um resumo e o aviso final."""
    linhas = [
        f"TALOS · Validação para {recomendacao.cliente.identificador} (perfil {recomendacao.perfil.nome})",
        cabecalho,
        "",
    ]
    linhas += _paragrafo(
        "Cada carteira é avaliada nos mesmos cenários. Referências: 100% do CDI (um CDB comum) e "
        "pesos iguais entre os produtos que servem para a meta. A coluna \"5% piores\" mostra quanto "
        "o valor aplicado varia, em média, nos 5% piores cenários até o fim do prazo."
    )
    total = len(comparacoes)
    for numero, (comparacao, resultado) in enumerate(zip(comparacoes, recomendacao.resultados), start=1):
        rotulo = ROTULOS_SITUACAO.get(comparacao.situacao, ROTULO_SEM_SOLUCAO)
        titulo = f"Meta {numero} de {total}: {comparacao.meta.nome}"
        linhas += ["", "=" * LARGURA, f"{titulo}{rotulo:>{LARGURA - len(titulo)}}"]
        linhas += _paragrafo(
            f"Precisa render {formatar_percentual(resultado.retorno_necessario_aa)} a.a. para ir de "
            f"{formatar_reais(comparacao.meta.valor_atual)} a {formatar_reais(comparacao.meta.valor_alvo)} "
            f"em {formatar_prazo(comparacao.meta.prazo_meses)}"
        )
        linhas += [
            "-" * LARGURA,
            f"  {'Carteira':<14}  {'Rende a.a.':>10}  {'5% piores':>10}  {'Chance de atingir':>18}",
        ]
        for linha in comparacao.linhas:
            linhas.append(
                f"  {linha.carteira:<14}  {formatar_percentual(linha.retorno_esperado_aa):>10}  "
                f"{percentual_com_sinal(linha.resultado_piores):>10}  "
                f"{formatar_probabilidade(linha.probabilidade_sucesso):>18}"
            )
        linhas += [""] + _paragrafo(conclusao(comparacao), "• ")
    linhas += ["", "=" * LARGURA] + _paragrafo(resumo_validacao(comparacoes)) + [""]
    linhas += _paragrafo(AVISO_SIMULACAO)
    return "\n".join(linhas)


def _tabela_carteira(resultado: ResultadoMeta) -> list[str]:
    """Tabela com produto, peso, valor aplicado e rendimento líquido esperado."""
    largura_nome = max(len("Produto"), *(len(a.produto.nome) for a in resultado.alocacoes))
    linhas = [f"  {'Produto':<{largura_nome}}  {'Peso':>6}  {'Valor aplicado':>16}  {'Rende a.a.':>10}"]
    for alocacao in resultado.alocacoes:
        linhas.append(
            f"  {alocacao.produto.nome:<{largura_nome}}  {formatar_percentual(alocacao.peso, 1):>6}  "
            f"{formatar_reais(alocacao.valor):>16}  {formatar_percentual(alocacao.retorno_liquido_esperado_aa):>10}"
        )
    return linhas


def _paragrafo(texto: str, marcador: str = "") -> list[str]:
    """Quebra um texto longo em linhas, com o marcador na primeira e recuo nas seguintes."""
    return textwrap.wrap(
        texto, width=LARGURA, initial_indent=marcador, subsequent_indent=" " * len(marcador)
    )


if __name__ == "__main__":
    raise SystemExit(main())
