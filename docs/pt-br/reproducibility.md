> 🇬🇧 **English:** [Read this page in English](../reproducibility.md)

# Notas de Reprodutibilidade

## Ativos de modelo e caches

| Backend | Origem dos pesos | Local do cache |
|---|---|---|
| OneFormer | `shi-labs/oneformer_ade20k_swin_large` via HuggingFace | `HF_HOME` (~1,7 GB) |
| Mask2Former | `facebook/mask2former-swin-large-ade-semantic` via HuggingFace | `HF_HOME` (~850 MB) |
| Detectron2 panoptic | model zoo `COCO-PanopticSegmentation/panoptic_fpn_R_50_3x` | cache do `FVCORE_CACHE`/torch |
| DeepLab | checkpoint Cityscapes do `DeepLabV3Plus-Pytorch` do VainF (download manual) | onde você guardar o `--ckpt` |

Defina `HF_HOME` e `TORCH_HOME` no `.env` se `~/.cache` não for o lugar dos
downloads de modelo. Eles ficam em cache; só a primeira execução paga.

## Configuração por backend

**OneFormer** precisa apenas do extra `ml` (`transformers`).

**Mask2Former** também precisa apenas do extra `ml`. Ele é deliberadamente *não*
instalado a partir do
[facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former):
aquele repositório é construído sobre o Detectron2 e compila ops CUDA customizadas
(MultiScaleDeformableAttention) do código-fonte, o que no Windows significa toda
a novela do toolchain MSVC de novo. A porta em `transformers` é a mesma
arquitetura carregando os mesmos pesos publicados, sem etapa de build.

O valor dele aqui é publicar pesos para vários datasets, de modo que o espaço de
classes é propriedade do checkpoint e não do backend:

```bash
tree-ai --seg mask2former                                  # ADE20K, tem classe de árvore
tree-ai --seg mask2former --seg-model facebook/mask2former-swin-large-coco-panoptic
tree-ai --seg mask2former --seg-model facebook/mask2former-swin-tiny-cityscapes-semantic
```

`--seg-model` também funciona para `--seg oneformer`. O token de dataset no nome
(`ade`, `coco`, `cityscapes`) seleciona a taxonomia, e o token de tarefa
(`semantic`, `panoptic`) seleciona o pós-processamento — um checkpoint
Mask2Former é treinado para uma tarefa, então o nome decide, e não uma flag. Um
checkpoint que não nomeia nenhum dataset reconhecido, como os do Mapillary
Vistas, é recusado em vez de adivinhado; passe `--taxonomy` para declarar o
mapeamento você mesmo.

Use checkpoints `swin-tiny` para experimentar: mesma interface, uma fração do
download.

**Detectron2** compila do código-fonte:

```bash
# Linux
sudo apt install build-essential python3-dev
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"
```

No Windows, siga as instruções upstream (Visual Studio Build Tools é
obrigatório). O backend roda o modelo COCO-panoptic do zoo e não tem caminho para
pesos customizados: nenhum espaço de classes publicado do Detectron2 carrega
árvore como classe *thing*, então não há nada que uma config de instância
fine-tuned pudesse acrescentar aqui.

Veja [Detectron2 no Windows](detectron2-windows.md) para a falha de
`pkg_resources`, o acoplamento torch/`_C` e uma avaliação de se o WSL vale a
pena.

**DeepLab** precisa do repositório do VainF mais um checkpoint Cityscapes. O
repositório é código de pesquisa, **não um pacote instalável** — ele não traz
`setup.py` nem `pyproject.toml`, então `pip install -e` nele falha com:

```
ERROR: ... does not appear to be a Python project:
neither 'setup.py' nor 'pyproject.toml' found.
```

Aponte o `--deeplab-repo` para um checkout em vez disso. Nada é instalado, e o
checkout não é modificado.

Você não precisa do repositório inteiro. O pipeline importa exatamente uma coisa
de lá — `network.modeling` — e esse pacote é autocontido: depende de torch e
numpy, e de mais nada no repositório. Todo o resto de lá é andaime de treino
(`datasets/`, `metrics/`, `main.py`) e 2,1 MB de imagens de demonstração.

**Recomendado — buscar só o necessário (sem git, commit fixado):**

```bash
python scripts/fetch-deeplab.py            # -> ./DeepLabV3Plus-network, ~65 KB

tree-ai --image street.jpg --seg deeplab \
        --deeplab-repo ./DeepLabV3Plus-network \
        --ckpt best_deeplabv3plus_mobilenet_cityscapes_os16.pth
```

O script baixa um commit fixado como tarball e extrai `network/` mais o `LICENSE`
upstream, de modo que duas máquinas obtêm código de modelo byte a byte idêntico.

**Com git, se você preferir um checkout de verdade** — esparso e raso, 262 KB em
vez de 11 MB:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/VainF/DeepLabV3Plus-Pytorch
cd DeepLabV3Plus-Pytorch && git sparse-checkout set network
```

**Clone completo** (`git clone https://github.com/VainF/DeepLabV3Plus-Pytorch`)
também funciona e é a escolha certa apenas se você pretende treinar ou avaliar
com os próprios scripts do upstream.

| Abordagem | Baixado | Em disco | Precisa de git | Fixado |
|---|---|---|---|---|
| `scripts/fetch-deeplab.py` | 2,2 MB | **65 KB** | não | sim |
| Clone esparso + raso | 262 KB | 262 KB | sim (2.25+) | não |
| Clone completo | ~11 MB | 11 MB | sim | não |

Os três produzem resultados idênticos; a escolha é só sobre pegada em disco e se
você quer o histórico git do upstream.

Os checkpoints vêm do [README
upstream](https://github.com/VainF/DeepLabV3Plus-Pytorch#results); o loader
infere a arquitetura pelo nome do arquivo e recusa um checkpoint que não caiba no
backbone escolhido, de modo que um checkpoint mobilenet não pode ser carregado em
silêncio num resnet101.

O carregamento de checkpoint é explicitamente `weights_only=True`,
independentemente da versão do Torch instalada. Um checkpoint que exige o pickle
do Python é rejeitado, porque o pickle pode executar código arbitrário durante o
carregamento.

**Os checkpoints upstream linkados acima são exatamente esse tipo de arquivo**,
então o caminho documentado do DeepLab exige o opt-in em toda execução:

```
ValueError: This checkpoint requires Python pickle, which can execute code while
loading. Use a weights-only checkpoint, or pass --trust-checkpoint ...
```

Eles são publicados pelo repositório em torno do qual este backend foi
construído, então confiar neles é uma decisão razoável — mas deve ser uma
decisão, tomada uma vez e registrada, e não um padrão silencioso:

```bash
tree-ai --image street.jpg --seg deeplab --ckpt best_deeplabv3plus_mobilenet_cityscapes_os16.pth \
        --trust-checkpoint
```

Defina isso uma vez para a máquina em vez de repetir a flag, junto com os demais
padrões do DeepLab abaixo:

```ini
# .env
UC_TRUST_CHECKPOINT=1
```

De um jeito ou de outro, a execução loga um aviso nomeando o arquivo em que
confiou. Verifique o arquivo antes se ele não veio do release upstream — o
SHA-256 registrado no manifesto é o que torna isso verificável depois.

A opção equivalente na biblioteca é `allow_pickle=True`; seu padrão é `False`.
Para execuções bem-sucedidas do DeepLab, o manifesto também registra o digest
SHA-256 do checkpoint sem expor seu caminho local específico da máquina, de modo
que os pesos exatos podem ser verificados independentemente do nome do arquivo.

### Padrões permanentes

O checkpoint e o checkout ficam no mesmo caminho por semanas enquanto todas as
outras flags mudam de execução para execução, então são configuração e não
argumentos. Defina-os uma vez — no ambiente ou no `.env` — e as flags viram
opcionais:

| Variável | Substitui | Recorre a |
|---|---|---|
| `UC_DEEPLAB_CKPT` | `--ckpt` | — (obrigatório de um jeito ou de outro) |
| `UC_DEEPLAB_REPO` | `--deeplab-repo` | o que quer que já importe como `network` |
| `UC_DEEPLAB_MODEL` | `--deeplab-model` | arquitetura inferida do nome do arquivo do checkpoint |

```ini
# .env
UC_DEEPLAB_CKPT=C:/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
UC_DEEPLAB_REPO=./DeepLabV3Plus-network
```

```bash
tree-ai --image street.jpg --seg deeplab                       # usa os dois
tree-ai --image street.jpg --seg deeplab --ckpt outro.pth      # a flag sobrescreve
```

A precedência é flag, depois variável, depois nada. Um arquivo ausente é
reportado contra quem quer que o tenha fornecido, então `does not exist` nomeia
`--ckpt` ou `UC_DEEPLAB_CKPT` em vez de deixar você adivinhar qual estava em
vigor. Valores em branco (`UC_DEEPLAB_CKPT=`, como vem no `.env.example`) contam
como não definidos.

Para uso em biblioteca ou notebook, passe os mesmos caminhos diretamente:

```python
build_segmenter("deeplab", ckpt_path=ckpt, repo_path="./DeepLabV3Plus-network")
```

Um arquivo `.pth` em site-packages é uma terceira opção, se você preferir que o
checkout seja importável por tudo no venv:

```bash
python -c "import sysconfig, pathlib; \
  pathlib.Path(sysconfig.get_paths()['purelib'], 'deeplab.pth') \
  .write_text(str(pathlib.Path('DeepLabV3Plus-network').resolve()))"
```

Lembre-se do que este backend consegue e do que não consegue reportar: o
Cityscapes não tem classe de árvore, então a cobertura arbórea volta como
`unavailable` e sinalizada. Só `--allow-vegetation-proxy` produz um número, com
origem `vegetation_proxy`:

```text
TREE COVERAGE n/a  (source=unavailable)      # padrão: honesto
  vegetation coverage: 34.54%
  flags: tree_coverage_unavailable

TREE COVERAGE 34.49%  (source=vegetation_proxy)   # com o proxy habilitado
  flags: tree_from_vegetation_proxy
```

## Determinismo

- Planos de heading são funções puras da configuração (`core/viewplan.py`).
- `--seed` semeia Python, NumPy e torch. Ele **não** atribui `PYTHONHASHSEED`: o
  Python lê essa variável antes da inicialização do interpretador, então uma
  atribuição em runtime seria enganosa e inócua para o processo atual.
- `--deterministic` chama `torch.use_deterministic_algorithms(True)`, desabilita o
  benchmarking do cuDNN, habilita comportamento determinístico do cuDNN e
  configura o workspace do cuBLAS antes da inicialização do modelo/CUDA. Uma
  operação sem implementação determinística pode então falhar ruidosamente.
- O manifesto separa `rng_seeded` de `deterministic_algorithms_requested`,
  registra as flags efetivas de Torch/cuDNN/CUDA e sempre declara
  `bitwise_determinism_guaranteed=false`: versões, drivers e hardware ainda podem
  mudar resultados de ponto flutuante.
- Quadros do Street View são cacheados pelo conjunto completo de parâmetros, e o
  id do panorama + a data de captura são registrados por vista: o Google
  refotografa ruas, então duas execuções com meses de diferença podem
  legitimamente divergir — o id do panorama é o que diz se deveriam.
- Entradas do cache são decodificadas antes do reuso. Downloads são decodificados
  antes de uma substituição atômica, de modo que uma escrita corrompida ou
  interrompida nunca é publicada como quadro válido em cache.
- O Google pode servir imagens diferentes para as mesmas coordenadas ao longo do
  tempo. Para um estudo congelado, arquive os quadros baixados (o diretório de
  cache) junto com o arquivo de predições.

## Validação e instalações limpas

A configuração de runtime rejeita coordenadas não finitas, parâmetros de captura
e limiares fora de faixa, dimensões de imagem malformadas ou excessivas, modos
inválidos e kernels de morfologia negativos ou exagerados. Os mesmos validadores
sem dependências sustentam as dataclasses, o parsing da CLI e os schemas da API.

O job normal de CI instala apenas `dev,api` e verifica que os módulos adaptadores
importam sem dependências de ML. Um job separado, `ml-import-smoke`, instala o
extra `ml`. As wheels construídas excluem `urban_canopy.tests` e são inspecionadas
quanto a esse contrato durante a CI.

## Uso da API do Google

Requisições de imagem são cobradas; requisições de metadados não. O pipeline chama
metadados uma vez por local, não por heading. O cache em disco faz com que
execuções repetidas do mesmo plano batam no Google apenas uma vez.
