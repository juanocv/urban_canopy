> 🇬🇧 **English:** [Read this page in English](../faq.md)

# FAQ

O README diz o que executar. Este documento diz por que o projeto se comporta
como se comporta, e cobre o que costuma dar errado numa máquina nova.

- [Instalação](#instalação)
- [Escolha do backend](#escolha-do-backend)
- [O indicador](#o-indicador)
- [Ground truth e avaliação](#ground-truth-e-avaliação)
- [Execução e saídas](#execução-e-saídas)
- [Desenvolvimento](#desenvolvimento)

---

## Instalação

### `pip install -e` no DeepLabV3Plus-Pytorch falha com "does not appear to be a Python project"

Porque ele não é um. O `DeepLabV3Plus-Pytorch` do VainF é código de pesquisa sem
`setup.py`, então o pip não tem o que instalar.

Você também não precisa cloná-lo. O pipeline importa exatamente uma coisa de lá
— o pacote autocontido `network` — e um script auxiliar busca isso num commit
fixado, sem git, em cerca de 65 KB:

```bash
python scripts/fetch-deeplab.py
tree-ai --image street.jpg --seg deeplab \
        --deeplab-repo ./DeepLabV3Plus-network --ckpt <pesos-cityscapes.pth>
```

`--deeplab-repo` também aceita um clone completo ou esparso, se você já tiver um.

### Preciso passar `--ckpt` e `--deeplab-repo` em toda chamada ao DeepLab?

Não. Ambos são propriedades da máquina, não de uma execução, então defina-os uma
vez no `.env`:

```ini
UC_DEEPLAB_CKPT=C:/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
UC_DEEPLAB_REPO=./DeepLabV3Plus-network
```

```bash
tree-ai --image street.jpg --seg deeplab        # ambos resolvidos pelo .env
```

As flags ainda vencem quando passadas, para uma sobrescrita pontual.

### Detectron2 falha com `ModuleNotFoundError: No module named 'pkg_resources'`

O Detectron2 importa `pkg_resources`, que o setuptools removeu na versão 81:

```bash
python -m pip install "setuptools<81"
```

Isso quebra de forma idêntica no Linux e no WSL — não é um problema de Windows.

### Devo migrar para o WSL para o Detectron2 funcionar?

Em geral não, se a build no Windows já funciona. A parte genuinamente dolorosa é
a mesma em todo lugar: o Detectron2 compila uma extensão `_C` contra a build
exata do torch presente no momento da compilação, então trocar o torch força uma
recompilação.

O [`detectron2-windows.md`](detectron2-windows.md) cobre a decisão inteira, o
acoplamento torch/`_C` e os casos em que o WSL realmente compensa.

### `--device cuda` falha mesmo com o `nvidia-smi` funcionando

Verifique a build do torch **dentro do ambiente de onde você está executando**.
Um venv não herda o torch de outro venv, então uma string de versão terminada em
`+cpu` significa que CUDA está indisponível ali, independentemente do que a GPU
consiga fazer:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Se aparecer `+cpu`, reinstale o torch a partir de
[pytorch.org](https://pytorch.org/get-started/locally/) com a build CUDA do seu
driver, ou execute com `--device cpu`.

### Quanto é baixado no primeiro uso?

OneFormer e Mask2Former instalam com o extra `ml` e não precisam de mais nada;
seus pesos caem em `HF_HOME` no primeiro uso:

| Checkpoint | Tamanho aproximado |
|---|---|
| OneFormer ADE20K Swin-L (padrão) | 1,7 GB |
| Mask2Former ADE20K Swin-L (padrão) | 850 MB |
| Variantes `swin-tiny` | bem menos |

O Detectron2 também baixa seus pesos COCO-panoptic no primeiro uso, mas antes
disso compila do código-fonte — `build-essential python3-dev` no Ubuntu, Visual
Studio Build Tools no Windows.

### Por que existe o `--trust-checkpoint`?

O DeepLab carrega checkpoints weights-only por padrão. Os checkpoints originais
do VainF precisam do pickle do Python, que pode executar código durante o
carregamento, então exigem um `--trust-checkpoint` explícito (ou
`UC_TRUST_CHECKPOINT=1` no `.env`). Use apenas para um arquivo em que você
confia. Execuções bem-sucedidas do DeepLab registram o SHA-256 do checkpoint no
manifesto.

---

## Escolha do backend

### Qual backend devo usar?

OneFormer, a menos que você tenha motivo para não usá-lo. Medido contra o ground
truth manual nos sete quadros anotados de exemplo:

| Backend | IoU | Precisão | Revocação | MAE de cobertura | Viés |
|---|---|---|---|---|---|
| OneFormer | 0,880 | 0,935 | 0,937 | 0,96 pp | +0,02 pp |
| Mask2Former | 0,833 | 0,914 | 0,904 | 1,46 pp | −0,16 pp |
| Detectron2 | 0,714 | 0,721 | 0,986 | 5,16 pp | +5,16 pp |
| DeepLab | — | — | — | — | — |

Sete quadros são uma amostra pequena, e a leitura honesta é uma afirmação sobre a
direção e a ordem de grandeza do erro, não um ranking entre backends cujos erros
se sobrepõem. O que ela mostra com clareza: o viés do Detectron2 é igual ao seu
MAE, ou seja, ele erra sempre para o mesmo lado — revocação 0,986 contra precisão
0,721 significa que ele pinta cobertura arbórea sobre coisas que o anotador não
rotulou. Isso é um deslocamento calibrável, não ruído.

### Por que o DeepLab não reporta cobertura arbórea nenhuma?

Porque o Cityscapes não tem classe de árvore. Sua classe `vegetation` funde
árvores com arbustos, então não há razão arbórea honesta a reportar, e o pipeline
diz `tree_source="unavailable"` em vez de passar o número de vegetação como se
fosse de árvores.

`--allow-vegetation-proxy` sobrescreve isso quando você quiser; os resultados
passam então a carregar `tree_from_vegetation_proxy` e
`tree_source="vegetation_proxy"`, de modo que a substituição fica visível em todo
arquivo derivado.

### Por que o Mask2Former aparece com três espaços de classes?

Ele é o único backend publicado para vários deles, o que permite manter a
arquitetura fixa e variar o conjunto de rótulos — separando "o modelo discorda"
de "o dataset não tem essa classe". Em um quadro de exemplo:

```text
oneformer     tree 31.97%   vegetation 42.68%     (ADE20K)
mask2former   tree 32.69%   vegetation 42.88%     (ADE20K)
detectron2    tree 36.21%   vegetation 46.00%     (COCO-panoptic)
deeplab       tree    n/a   vegetation 34.54%     (Cityscapes — sem classe de árvore)

mask2former --seg-model facebook/mask2former-swin-tiny-cityscapes-semantic
              tree    n/a   vegetation 36.24%     (mesmo modelo, sem classe de árvore)
```

A última linha é o ponto: a mesma arquitetura não reporta razão arbórea alguma
quando apontada para um espaço de classes incapaz de expressá-la.

---

## O indicador

### Por que o projeto mede área e nunca conta árvores?

Porque nenhum modelo disponível para esses espaços de classes consegue contá-las.
Isso é um achado, não uma simplificação:

- O COCO-80 tem apenas `potted plant`.
- O `tree-merged` do COCO-panoptic é uma classe *stuff* — todas as árvores de um
  quadro formam uma única região por construção.
- As 1203 categorias do LVIS v1 contêm apenas `Christmas_tree`.
- O Cityscapes-instance tem oito classes de pessoas e veículos.
- O conjunto de instâncias do ADE20K (100 things) tem `palm` e `flower`, mas não
  `tree`.

Todo modelo de segmentação de instâncias de árvore que dá para baixar —
detectree2, DeepForest, `restor/tcd-mask-rcnn-r50` — é treinado em imagens
**aéreas verticais**, onde as copas são manchas separadas. Do nível da rua elas
se sobrepõem e se ocluem, e os trabalhos recentes que atacam isso não liberam
pesos.

Ou seja, uma métrica por instância só poderia ter sido calculada contra um modelo
que não existe. O suporte a árvores individuais foi removido em vez de ficar como
um caminho de código inalcançável.

### Por que uma razão contínua em vez de faixas "pouca / média / muita vegetação"?

Faixas escondem a escolha do limiar dentro do resultado. Dois estudos que citam
"vegetação média" não podem ser comparados a menos que ambos publiquem seus
pontos de corte, e um valor perto de uma fronteira muda de categoria com ruído de
medição. A razão contínua é a saída; quem precisar de faixas aplica os próprios
limiares sobre ela e os declara.

### Qual é o denominador?

Todos os pixels da imagem. Os quadros são usados como entregues — sem recortar
céu, via ou capô do veículo — porque um denominador que varia por imagem torna
duas medições incomparáveis.

### Por que o mesmo local é reportado quatro vezes?

Porque um heading isolado é propriedade da fotografia, não do lugar. Na varredura
de quatro headings em `samples/images/`, a cobertura rotulada manualmente de um
mesmo ponto vai de 1,2% a 29,3% — uma variação de 28 pp que não é erro do modelo.

É por isso que execuções multi-vista reportam mediana e IQR sobre um conjunto
determinístico de headings, e por isso os headings vêm da configuração e nunca da
saída da segmentação: escolher a vista pelo quão bem o modelo a segmenta
enviesaria a medição por construção.

---

## Ground truth e avaliação

### Por que anotar um polígono por árvore se a métrica é por pixel?

Por duas razões. É o que o Roboflow produz naturalmente, e o ground truth em
pixels é simplesmente a união desses polígonos — desenhar um segundo ground
truth, em nível de região, sobre os mesmos pixels produziria duas versões que
discordariam entre si.

Manter as instâncias não custa nada e deixa em aberto um trabalho por árvore no
futuro.

### Por que um quadro sem árvores importa?

É o único tipo de quadro no qual um falso positivo é mensurável. Todas as outras
imagens podem punir um modelo por deixar de detectar cobertura; só um quadro sem
árvores pode puni-lo por inventar cobertura.

O Roboflow não exporta nada para um quadro assim, então ele precisa ser
reinserido à mão como uma entrada de imagem com zero anotações. Mantenha essa
lista em algum lugar visível no código de avaliação, e não escondida num arquivo
de dados: um caso negativo que some em silêncio é indistinguível de um que nunca
foi rotulado.

### Por que fundir as exportações de anotação antes de avaliar?

O Roboflow exporta um arquivo COCO por trabalho de rotulação. Avaliá-los um a um
dá um relatório por imagem, a *n* = 1 cada, e esses não podem ser promediados:
uma IoU micro-averaged é agrupada sobre pixels, não sobre pontuações por imagem.
Fundir antes dá um resultado no tamanho real da amostra.

### Por que dois níveis de avaliação em vez de um número só?

Eles respondem perguntas diferentes, e discordam de um jeito informativo. Uma
máscara deslocada lateralmente pontua mal em IoU enquanto concorda quase
exatamente na cobertura:

- **Nível de pixel** — IoU, Dice/F1, precisão, revocação. A máscara está no lugar
  certo?
- **Nível do indicador** — MAE, RMSE, viés em pontos percentuais. O número que o
  estudo vai publicar está certo?

Reportar só um esconde metade da história. Convenções completas, regras de
casamento e tratamento de casos vazios: [`evaluation.md`](evaluation.md).

---

## Execução e saídas

### Nada foi escrito depois da minha execução

Nada é escrito a menos que uma flag de saída peça. `--save-artifacts` escreve o
pacote inteiro; `--csv`, `--json` e as flags de imagem pedem cada uma um pedaço.
`--csv` sozinha escreve as linhas e nenhuma imagem, que costuma ser o que um lote
grande quer, e qualquer uma delas aceita um caminho explícito
(`--csv resultados.csv`).

### Por que as execuções se acumulam em vez de sobrescrever?

Porque comparar backends é o objetivo. Analisar uma imagem com o OneFormer e
depois com o Detectron2 deixa os dois resultados lado a lado, cada um em seu
próprio diretório com timestamp sob `--outdir`. Dê nome à execução você mesmo com
`--run-name`.

### Um lote grande de imagens locais mantém tudo em memória?

Não. Lotes são consumidos como iterador, o RGB fica desabilitado a menos que
artefatos de imagem sejam pedidos e, quando são, cada vista é escrita
imediatamente e sua alocação de RGB é liberada antes que o próximo resultado se
acumule.

### O que acontece quando alguns headings falham numa execução multi-vista?

Multi-vista exige ao menos um heading utilizável por padrão;
`--min-successful-views N` define uma regra de estudo mais estrita. Headings que
falharam retornam com seu estágio (`fetch` ou `analysis`) e o tipo de erro, em vez
de desaparecerem nos logs.

### Outras flags que vale conhecer

- `--no-refine` passa a máscara bruta do segmentador adiante — a baseline de
  comparação que todo experimento de refinamento deveria reportar.
- `--view-mode offsets|equiangular|fixed`, com `--offsets`, `--n-views` ou
  `--headings`, controla o plano multi-vista de forma determinística.
- `--deterministic` adicionalmente solicita algoritmos determinísticos de
  Torch/CUDA. É mais estrito que `--seed`, mas o manifesto deliberadamente não
  afirma identidade bit a bit entre hardwares ou versões de biblioteca
  diferentes.

### Por que a Web API se recusa a subir?

Configuração de backend inválida ou incompleta aborta a inicialização antes que o
servidor se declare pronto — um servidor que responde `/ping` mas não consegue
segmentar é pior do que um que nunca subiu. Confira os valores `UC_*` contra o
`.env.example`.

Note que a API não tem autenticação e chama uma API paga do Google a cada
requisição. Mantenha-a atrás de um proxy ou presa ao localhost.

---

## Desenvolvimento

### O que os gates de qualidade cobram de fato?

- **pytest** — 80% de cobertura agregada de branches, mais um piso de 60% por
  módulo através do `scripts/check_coverage.py`, para que um pacote bem coberto
  não consiga esconder um módulo sem teste.
- **Hypothesis** — testes de propriedade sobre RLE, máscara/cobertura, agregação
  e invariantes geográficas.
- **Ruff** — bugbear, ordenação de imports, modernização, simplificação e regras
  específicas de NumPy, além de erros fatais e nomes indefinidos.
- **Pyright** — os contratos científicos públicos, de dependências leves.

A CI testa Python 3.10 e 3.13. Um workflow semanal instala separadamente o
conjunto mínimo declarado de dependências e os lançamentos mais recentes
compatíveis, de modo que os limites inferiores e as atualizações upstream sejam
afirmações executáveis, e não metadados sem teste.

### Por que a suíte de testes padrão é offline e somente CPU?

Para que um clone possa ser verificado sem GPU, sem chave de API e sem um
download de 1,7 GB. O `pyproject.toml` desmarca os marcadores `gpu` e `network`
por padrão; rode-os deliberadamente com `pytest -m gpu` / `pytest -m network`, e
coloque novos testes pesados sob um dos marcadores.
