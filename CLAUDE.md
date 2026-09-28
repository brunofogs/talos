# Talos — instruções para o Claude Code

Você vai construir, passo a passo, o motor de alocação **Talos**: um sistema que monta carteiras de investimento personalizadas para investidores brasileiros de baixa e média renda, a partir da prateleira de produtos de uma instituição parceira.

Responda sempre em **português**. Explique o que fez ao fim de cada etapa, em linguagem simples, porque o dono do projeto não é programador experiente.

---

## 1. Regra número um: o código precisa rodar em qualquer computador

Tudo deve funcionar igual em **Windows, macOS e Linux**, sem configuração especial.

- **Linguagem:** Python 3.10 ou mais novo.
- **Dependências mínimas**, só pacotes que instalam com `pip` em qualquer sistema, sem compilador: `numpy`, `scipy`, `pandas` e `pytest` (testes). Peça permissão antes de adicionar qualquer outra.
- **Otimização:** use `scipy.optimize.linprog` com `method="highs"`. Nada de solvers comerciais ou que exijam instalação separada.
- **Caminhos de arquivo:** sempre com `pathlib.Path`, nunca caminhos absolutos nem barras fixas (`/` ou `\`).
- **Texto:** abra e salve arquivos sempre com `encoding="utf-8"`.
- **Sem internet para rodar:** os dados de exemplo ficam dentro do projeto, em CSV e JSON. Nada de APIs pagas ou chaves.
- **Sem scripts de um sistema só:** nada de `.sh` ou `.bat` como único jeito de rodar. O ponto de entrada é `python -m talos`.
- **Reprodutível:** toda simulação aleatória recebe uma semente (`seed`) fixa e configurável.
- **Prova de universalidade:** crie um workflow do GitHub Actions que rode os testes em `ubuntu-latest`, `windows-latest` e `macos-latest`, com Python 3.10 e 3.12.

---

## 2. Estrutura do projeto

```
talos/
├── pyproject.toml          # dependências com faixas de versão
├── README.md               # como instalar e rodar em cada sistema
├── CLAUDE.md               # este arquivo
├── .github/workflows/testes.yml
├── exemplos/
│   ├── cliente.json        # cliente fictício com metas
│   ├── prateleira.csv      # produtos fictícios de uma instituição
│   ├── curva_di.csv        # vértices fictícios da curva de DI futuro
│   └── premissas.json      # CDI, IPCA, limites do FGC etc.
├── src/talos/
│   ├── __init__.py
│   ├── __main__.py         # linha de comando
│   ├── modelos.py          # dataclasses: Cliente, Meta, Produto, Premissas
│   ├── impostos.py         # IR regressivo, isenções, ações, come-cotas
│   ├── fgc.py              # limites de garantia
│   ├── mercado.py          # curva de juros e geração de cenários
│   ├── otimizador.py       # otimização por CVaR, meta a meta
│   ├── explicacao.py       # frases em português, geradas por regras
│   ├── auditoria.py        # registro de cada recomendação
│   └── validacao.py        # comparação com referências simples
├── tests/                  # um arquivo de teste por módulo
└── saidas/                 # recomendações geradas (ignorado pelo git)
```

---

## 3. O que cada módulo faz (versão 1)

**modelos.py.** Dataclasses com type hints:
- `Meta`: nome, valor atual, valor-alvo, prazo em meses, se é reserva de emergência.
- `Cliente`: perfil de suitability, metas, renda mensal, dívidas caras (sim/não).
- `Produto`: nome, emissor, conglomerado, indexador, taxa, prazo/carência, tributação, classe, risco (1 a 5), coberto pelo FGC, aplicação mínima.
- `Premissas`: carregadas de `premissas.json`.

**impostos.py:**
- Tabela regressiva de IR: 22,5% até 180 dias; 20% de 181 a 360; 17,5% de 361 a 720; 15% acima de 720.
- Produtos isentos (LCI, LCA, debêntures incentivadas): 0%.
- Fundos de ações e ETFs: 15% sobre o ganho.
- Come-cotas semestral (maio e novembro): 15% para fundos de longo prazo e 20% para curto prazo, antecipando o IR.
- Uma função que calcula o **retorno líquido anualizado** dado o retorno bruto, a tributação e o prazo.

**fgc.py.** Limite por conglomerado (R$ 250 mil) e teto global por CPF (R$ 1 milhão a cada 4 anos). Os dois valores vêm de `premissas.json` para poderem ser atualizados, e o README deve avisar que eles precisam ser conferidos antes do uso real.

**mercado.py:**
- Lê `curva_di.csv` e projeta o CDI médio para qualquer prazo, por interpolação.
- Calcula a inflação implícita a partir de premissas simples.
- Gera N cenários de retorno (padrão 5.000) para cada classe de ativo. Use distribuição t de Student para ter caudas gordas, correlações configuráveis e semente fixa.

**otimizador.py.** O coração do sistema. Para **cada meta separadamente**:
1. Filtre os produtos: prazo/carência até o fim da meta, risco permitido pelo perfil, prazo mínimo para renda variável (3 anos) e para risco 3 (2 anos).
2. Se a meta é reserva de emergência, use só produtos de liquidez diária e risco 1.
3. Calcule o retorno necessário: `(valor_alvo / valor_atual) ** (1 / anos) - 1`.
4. **Minimize o CVaR a 95%** dos cenários (formulação linear de Rockafellar-Uryasev), sujeito a:
   - retorno líquido esperado maior ou igual ao retorno necessário;
   - soma dos pesos igual a 1 e pesos não negativos;
   - limite por produto (padrão 40%, exceto Tesouro Selic);
   - FGC por conglomerado;
   - teto de renda variável por perfil;
   - aplicação mínima de cada produto.
5. Se não houver solução, maximize o retorno esperado com o CVaR limitado ao teto do perfil e informe que a meta está difícil de atingir.
6. Calcule a **probabilidade de atingir a meta** nos cenários.
7. Opção `reamostragem=True`: repita a otimização com estimativas perturbadas e tire a média (reamostragem de Michaud).

**explicacao.py.** Para cada produto escolhido, gere frases em português por **regras** (nunca por IA generativa). Por exemplo: isenção equivalente a um CDB de X% do CDI, limite do FGC atingido, vencimento antes da meta, chance de sucesso da meta. Se o cliente tem dívidas caras, a primeira frase recomenda quitá-las antes de investir.

**auditoria.py.** Salva em `saidas/` um JSON por recomendação, com: data e hora, versão do código, todas as entradas, premissas, semente, resultado e explicações.

**validacao.py.** Compara a carteira do Talos com duas referências (100% CDI e pesos iguais entre os produtos elegíveis) nos mesmos cenários, e mostra retorno, CVaR e chance de sucesso lado a lado.

**__main__.py.** Linha de comando:
```
python -m talos recomendar --cliente exemplos/cliente.json --prateleira exemplos/prateleira.csv
python -m talos validar --cliente exemplos/cliente.json --prateleira exemplos/prateleira.csv
```
A saída no terminal deve ser legível: tabela da carteira por meta, estatísticas e explicações.

---

## 4. Regras de qualidade

- Type hints e docstrings em português em todas as funções.
- **Toda função financeira tem teste** com casos calculados à mão (por exemplo: IR de 22,5% para 180 dias, isenção de LCA, limite do FGC sendo respeitado, meta impossível tratada sem erro).
- Nenhuma taxa ou número "mágico" dentro do código: tudo vem dos arquivos de exemplo ou de premissas.
- Os dados de exemplo são **fictícios** e isso fica escrito no README e no topo de cada arquivo de exemplo.
- Funções pequenas, nomes em português, código fácil de ler.

## 5. O que não fazer

- Não executar ordens nem conectar a corretoras.
- Não usar dados pessoais reais.
- Não gerar explicações com IA generativa.
- Não adicionar dependências sem pedir.
- Não prometer retornos: toda saída termina com o aviso de que é uma simulação e não uma recomendação de investimento.

---

## 6. Ordem de trabalho

Trabalhe **uma etapa por vez**. Ao fim de cada uma: rode os testes, mostre o resultado e explique em português simples o que foi feito. **Espere minha confirmação** antes de seguir.

1. **Esqueleto:** estrutura de pastas, `pyproject.toml`, README com instalação para Windows, macOS e Linux (usando `python -m venv`), workflow do GitHub Actions e um teste que passa.
2. **Dados de exemplo e modelos:** arquivos em `exemplos/` e `modelos.py`, com testes de leitura.
3. **Impostos e FGC**, com testes.
4. **Mercado:** curva de juros e cenários, com testes.
5. **Otimizador**, com testes.
6. **Explicações e auditoria.**
7. **Linha de comando** funcionando de ponta a ponta.
8. **Validação** contra as referências simples.

Comece pela etapa 1.
