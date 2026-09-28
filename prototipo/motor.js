// Motor do protótipo: roda o Python (Pyodide) e o pacote talos num Web Worker,
// para a página continuar respondendo enquanto as carteiras são calculadas.

const PASTA_DADOS = "/home/pyodide/exemplos";
let pyodide = null;

function avisar(tipo, dados) {
  self.postMessage({ tipo, ...dados });
}

async function iniciar() {
  const base = new URL("./", self.location.href).href;
  avisar("progresso", { etapa: "python", texto: "Carregando o Python do navegador" });
  importScripts(base + "pyodide/pyodide.js");
  pyodide = await loadPyodide({ indexURL: base + "pyodide/" });

  avisar("progresso", { etapa: "pacotes", texto: "Carregando numpy e scipy" });
  await pyodide.loadPackage(["numpy", "scipy"]);

  avisar("progresso", { etapa: "talos", texto: "Carregando o motor Talos" });
  const arquivos = await (await fetch(base + "arquivos.json")).json();
  for (const caminho of arquivos) {
    const texto = await (await fetch(base + caminho)).text();
    const destino = "/home/pyodide/" + caminho;
    const pasta = destino.slice(0, destino.lastIndexOf("/"));
    pyodide.FS.mkdirTree(pasta);
    pyodide.FS.writeFile(destino, texto, { encoding: "utf8" });
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')\nimport ponte");
  const prateleira = pyodide.runPython(`ponte.dados_da_prateleira("${PASTA_DADOS}")`);
  avisar("pronto", { prateleira: JSON.parse(prateleira) });
}

function simular(entrada) {
  pyodide.globals.set("entrada_json", JSON.stringify(entrada));
  const inicio = performance.now();
  const resposta = pyodide.runPython(`ponte.simular(entrada_json, "${PASTA_DADOS}")`);
  avisar("resultado", { resultado: JSON.parse(resposta), segundos: (performance.now() - inicio) / 1000 });
}

self.onmessage = (evento) => {
  try {
    if (evento.data.tipo === "simular") simular(evento.data.entrada);
  } catch (erro) {
    avisar("falha", { texto: String(erro) });
  }
};

iniciar().catch((erro) => avisar("falha", { texto: String(erro) }));
