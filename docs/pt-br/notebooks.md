> 🇬🇧 **English:** [Read this page in English](../../notebooks/README.md)

# Notebooks de exemplo

Exemplos completos para quem quer entender o que o pipeline produz antes de
plugá-lo em qualquer coisa. Os dois rodam sobre as imagens em `samples/images/`,
então **nenhuma chave da API do Google é necessária**, e os dois caem para CPU
quando não há GPU.

| Notebook | O que ele cobre |
|---|---|
| [`01_getting_started.ipynb`](../../notebooks/01_getting_started.ipynb) | Uma imagem de ponta a ponta: construir um backend, ler o indicador de cobertura, a separação árvore/vegetação, inspecionar máscaras brutas vs refinadas e a trava de crescimento do refinamento |
| [`02_multiview_and_evaluation.ipynb`](../../notebooks/02_multiview_and_evaluation.ipynb) | Agregar quatro headings de um mesmo local, por que mediana e IQR são reportadas, e os dois níveis de avaliação de ponta a ponta |

## Como executá-los

```bash
python -m pip install -e ".[ml,notebooks]"
jupyter lab notebooks/
```

A primeira célula de cada notebook resolve a raiz do repositório, seja ele
iniciado a partir de `notebooks/` ou da raiz do projeto.

Os dois usam `BACKEND = "oneformer"` por padrão, o que baixa ~1,7 GB de pesos no
primeiro uso e os mantém em cache. Mude essa variável para `"detectron2"` se você
já tiver uma instalação compilada do Detectron2 (veja
[`detectron2-windows.md`](detectron2-windows.md)).

## As saídas são versionadas

Os notebooks são guardados com suas saídas, de modo que as figuras e os números
são legíveis no GitHub sem executar nada. Eles foram executados contra as imagens
de exemplo em CUDA; reexecutá-los com outro backend ou dispositivo mudará os
números, o que é esperado. Todos os sete quadros de exemplo estão anotados
manualmente, então todo número de avaliação é pontuado contra ground truth.

Para reexecutar depois de editar:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
