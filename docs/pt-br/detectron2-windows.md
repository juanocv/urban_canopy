> 🇬🇧 **English:** [Read this page in English](../detectron2-windows.md)

# Detectron2 no Windows

O Detectron2 é o único backend que o upstream não suporta oficialmente no Windows
("we do not provide official support for Windows"). Esse aviso é real, mas mais
estreito do que parece, e vale separar os problemas que são de fato sobre Windows
daqueles que apenas acontecem lá.

## A falha de `pkg_resources`

```
File "...\detectron2\model_zoo\model_zoo.py", line 4, in <module>
    import pkg_resources
ModuleNotFoundError: No module named 'pkg_resources'
```

**Causa.** O `pkg_resources` vem como parte do `setuptools`, e o **setuptools o
removeu na versão 81**. O Detectron2 0.6 é anterior a essa remoção e ainda o
importa, num único lugar, para localizar os arquivos de config do model zoo
empacotados dentro da instalação.

**Correção.** Restaure-o no ambiente:

```bash
python -m pip install "setuptools<81"
```

**Isso não é um problema de Windows.** O mesmo lançamento do setuptools quebra o
mesmo import do Detectron2 de forma idêntica no Linux, macOS e WSL. Nada em
migrar de sistema operacional teria evitado isso, e vale ser explícito a respeito
porque o traceback chega com cara de build nativa quebrada.

Duas notas relacionadas:

- Python 3.12 e mais novos não criam mais virtualenvs com `setuptools`
  pré-instalado, então um venv novo pode não ter `pkg_resources` mesmo quando o
  Python do sistema tem. A falha parece então específica do ambiente, e não é.
- O `urban_canopy` não depende de `setuptools` e deliberadamente não o fixa: a
  restrição pertence ao Detectron2, não a este projeto, e fixá-la no
  `pyproject.toml` imporia uma ferramenta de build obsoleta a todo mundo que
  instalar apenas o caminho do OneFormer. Em vez disso,
  `build_segmenter("detectron2")` detecta exatamente essa falha e imprime o
  comando acima.

## A extensão compilada está presa a uma versão do torch

Uma instalação funcional contém uma extensão nativa nomeada pelo interpretador e
pela plataforma exatos contra os quais foi construída:

```
.venv\Lib\site-packages\detectron2\_C.cp313-win_amd64.pyd
```

Esse arquivo é linkado contra a ABI da libtorch do release do torch presente no
momento do build. **Trocar o torch depois exige recompilar o Detectron2.** Isso
importa mais ao migrar de uma build de CPU para uma de CUDA: instalar um torch
CUDA num ambiente cujo `_C` foi compilado contra um torch de CPU é a causa
habitual de um erro de import ou de um crash duro que aparece "do nada" depois.

Verifique o que você tem:

```bash
python -c "import torch, detectron2; from detectron2 import _C; print(torch.__version__, _C.__file__)"
```

Uma incompatibilidade aparece como um erro de import que não nomeia nada útil:

```
ImportError: DLL load failed while importing _C: The specified procedure could not be found.
```

## Migrando um venv de torch CPU para torch CUDA

Exemplo prático, e a ordem importa: torch primeiro, depois recompilar o
Detectron2 contra ele.

**1. Escolha uma build CUDA que sua GPU ainda suporte.** Este é o passo que
decide todo o resto. Placas Maxwell (GTX 9xx, compute capability 5.2) são
suportadas pelo CUDA 11.8 e abandonadas pelo CUDA 12.8, e o PyTorch parou de
distribuir wheels cu118 depois da 2.6. Verifique o que uma build candidata
suporta:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
python -c "import torch; print(torch.cuda.get_device_capability(0))"
```

Uma capability `(5, 2)` é coberta por `sm_50` na lista de arquiteturas — binários
CUDA são compatíveis dentro de uma família de arquitetura, então código `sm_50`
roda em `sm_52`.

**2. Instale torch e torchvision como um par casado**, somente dentro do venv:

```bash
.venv/Scripts/python.exe -m pip install "torch==2.6.0+cu118" "torchvision==0.21.0+cu118" \
  --index-url https://download.pytorch.org/whl/cu118
```

**3. Recompile o Detectron2 contra o novo torch**, a partir do mesmo commit, num
shell em que o MSVC esteja no path:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_VISIBLE_DEVICES=-1
set DISTUTILS_USE_SDK=1
set MAX_JOBS=2
.venv\Scripts\python.exe -m pip install --no-build-isolation --force-reinstall --no-deps ^
  "git+https://github.com/facebookresearch/detectron2.git@<commit>"
```

`--no-build-isolation` é obrigatório: o `setup.py` do Detectron2 importa o torch
que você acabou de instalar para decidir o que compilar, e um ambiente de build
isolado não o teria. `MAX_JOBS` limita a memória — uma build MSVC paralela sem
limite é causa comum do compilador ser morto no meio do processo.

### Por que `CUDA_VISIBLE_DEVICES=-1` durante o build

Isso faz o Detectron2 compilar suas ops para CPU (`CppExtension`) em vez de CUDA.
Aqui isso é deliberado, por duas razões:

- O `setup.py` seleciona uma build CUDA quando `torch.cuda.is_available()` e um
  toolkit CUDA são ambos encontrados. O toolkit que ele encontra é o que estiver
  instalado — **12.8** nesta máquina — enquanto o torch foi construído contra a
  **11.8**. Compilar uma extensão com uma versão maior de CUDA diferente da do
  torch não produz um binário funcional.
- O CUDA 12.8 não consegue gerar código para `sm_52` de qualquer forma, então uma
  build CUDA "bem-sucedida" não rodaria na GPU.

Definir a variável como `-1` faz `torch.cuda.is_available()` retornar falso
apenas para o processo de build, o que basta para selecionar o caminho de CPU.
Limpar `CUDA_HOME`/`CUDA_PATH` **não** funciona: o torch recorre a varrer o
diretório padrão de instalação do toolkit.

**O que isso custa.** A rede continua rodando na GPU — isso afeta apenas os
kernels customizados do próprio Detectron2 (caixas rotacionadas, convolução
deformável, avaliação COCO acelerada). O caminho panóptico que este projeto usa
nunca os chama: NMS e ROIAlign vêm do torchvision. Se um modelo futuro precisar de
um deles, ele falha com um claro "not compiled with GPU support" em vez do erro
críptico de DLL acima. Obter ops CUDA completas significaria instalar o toolkit
CUDA 11.8 ao lado do 12.8 e recompilar sem `CUDA_VISIBLE_DEVICES=-1`.

**Resultado medido** numa GTX 970, `panoptic_fpn_R_50_3x`, quadro de 600x400,
somente segmentação, após aquecimento:

| Dispositivo | Inferência mediana |
|---|---|
| CPU | 2,09 s |
| CUDA | 0,23 s |

Cerca de 9x, e saída de cobertura idêntica nos dois — que é a verificação que
importa: o dispositivo não pode mudar o número.

## O WSL vale a pena?

Não automaticamente, e em geral não depois que o Windows já está funcionando. A
troca honesta:

**Razões para ficar no Windows**

- Compilar o `_C` com MSVC é o passo genuinamente difícil, e é um custo único.
  Uma vez que esse `.pyd` existe e importa, a experiência do dia a dia é idêntica
  à do Linux. Migrar joga fora trabalho já pago.
- O projeto vive num drive Windows. A partir do WSL2 esse drive é alcançável em
  `/mnt/...` por uma camada de tradução cuja I/O de arquivos pequenos é lenta — e
  este pipeline lê datasets de imagem arquivo por arquivo. Evitar isso significa
  copiar o repositório para dentro do sistema de arquivos do WSL, o que deixa
  duas cópias de trabalho para manter sincronizadas.
- Caches de modelo (~1,7 GB só para o OneFormer, mais os pesos do zoo) seriam
  baixados e armazenados duas vezes, a menos que `HF_HOME`/`TORCH_HOME` sejam
  apontados atravessando a fronteira, o que reintroduz o caminho lento.
- Tudo, exceto o Detectron2 — OneFormer, DeepLab, a CLI, a API, a suíte de testes
  — não tem ressalva alguma no Windows.

**Razões pelas quais o WSL genuinamente ajuda**

- **Você precisa mudar a versão do torch.** Recompilar o Detectron2 é
  substancialmente mais fácil no Linux, e existem wheels pré-construídas para as
  combinações comuns de torch/CUDA. No Windows toda mudança de torch significa
  mais uma build MSVC. (Essa recompilação já foi feita aqui com sucesso — veja a
  seção acima — então o custo é conhecido, e não hipotético: um `pip install` com
  as variáveis de ambiente certas e alguns minutos de compilação.)
- **Você quer paridade com a CI ou com o alvo de deploy**, ambos Linux aqui.
- **A build MSVC ainda não deu certo.** Se você ainda está brigando com o
  compilador, o WSL é o caminho mais curto — é esse o cenário de que o aviso do
  upstream realmente trata.

**Recomendação.** Se `import detectron2` e `from detectron2 import _C` funcionam
no Windows, fique por lá e fixe `setuptools<81`. Recorra ao WSL quando uma
mudança de torch forçar uma recompilação, ou se um upgrade futuro de
Detectron2/Python não compilar — não por causa do erro de `pkg_resources`, que o
WSL não resolve.

## Conferindo o ambiente

O `tree-ai-diagnostics` reporta o interpretador, a build do torch, a
disponibilidade de CUDA e se cada backend importa. As duas coisas que vale
confirmar especificamente para o Detectron2:

```bash
python -c "import pkg_resources; print('pkg_resources ok')"
python -c "from detectron2 import model_zoo; print('model_zoo ok')"
```

Um torch somente-CPU reporta `+cpu` na sua string de versão, e `--device cuda`
vai falhar ali independentemente do que a GPU consiga fazer — vale conferir por
ambiente, já que um venv não herda a build de torch de outro na mesma máquina.

## Isolamento de ambiente

Mudar o torch num venv não pode afetar outro projeto na mesma máquina, desde que
o venv tenha sido criado sem os system site-packages:

```bash
grep include-system-site-packages .venv/pyvenv.cfg   # precisa dizer false
python -c "import sys; print(sys.prefix != sys.base_prefix)"   # precisa dizer True
```

Com isso, o único jeito de perturbar um projeto vizinho é rodar o `pip` contra o
interpretador errado, então enderece o venv explicitamente
(`.venv/Scripts/python.exe -m pip ...`) em vez de confiar em qual está ativo. O
driver da NVIDIA é compartilhado e não é tocado por nada disso; o CUDA em si
chega dentro da wheel do torch, e é por isso que dois venvs numa mesma máquina
podem conter versões diferentes de CUDA sem conflito.

Tire um retrato antes de uma migração de torch, para que o ambiente possa ser
restaurado:

```bash
.venv/Scripts/python.exe -m pip freeze > venv-freeze-backup.txt
```
