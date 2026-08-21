> 🇬🇧 **English:** [Read this page in English](../evaluation.md)

# Metodologia de Avaliação

A avaliação roda offline a partir de dois arquivos:

```bash
# 1. inferência: escreve artifacts_out/<timestamp>_<backend>/predictions.json
tree-ai --image street.jpg --predictions-json

# 2. avaliação, offline e repetível
tree-ai evaluate --predictions artifacts_out/<run>/predictions.json \
                 --annotations annotations.json
```

O arquivo de predições embute o manifesto da execução (modelo, versões,
taxonomia, configuração de refinamento, semente), de modo que todo número
reportado é rastreável. O schema de intercâmbio atual é
`urban_canopy/predictions/3`. Arquivos da versão 1 precisam ser regerados porque
podem ter usado um denominador recortado, e os da versão 2 porque carregavam
predições por instância que esta build não avalia mais — veja abaixo.

O casamento é feito pelo nome-base original da imagem. Para exportações do
Roboflow isso significa `images[].extra.name`, preferido sempre que presente
porque o Roboflow substitui o `file_name` por um nome com hash específico da
exportação; caso contrário usa-se o nome-base do `file_name` do COCO. Duas
imagens resolvendo para o mesmo nome é um erro fatal, e não uma sobrescrita
silenciosa, e o `tree-ai validate-dataset` reporta isso antes que qualquer
avaliação seja executada. Imagens presentes em apenas um dos lados são listadas
no relatório, nunca descartadas em silêncio.

**Extensões.** Ferramentas de anotação recodificam: um quadro JPEG rotulado no
Roboflow volta como `frame.png`. Nomes que ficaram sem par depois da passada
exata são, portanto, casados de novo ignorando a extensão, e todo par desse tipo
é logado e listado no relatório sob `settings.joined_across_extensions` — juntar
dois arquivos de nomes diferentes é um julgamento que o relatório precisa
mostrar, não esconder.

Casamentos exatos são feitos primeiro, então um conjunto que genuinamente contém
`frame.jpg` e `frame.png` casa cada um com a sua própria anotação e o fallback
nunca os vê. Quando o fallback seria ambíguo — dois restos de cada lado
compartilhando um nome —, a avaliação para em vez de escolher, porque parear o
errado pontua uma imagem contra o ground truth de outra e ainda assim imprime um
número plausível. A comparação diferencia maiúsculas de minúsculas, já que
`Frame.jpg` e `frame.jpg` são arquivos diferentes no Linux.

Dois níveis independentes são calculados. Eles respondem perguntas diferentes e
nunca são fundidos numa única pontuação.

**Não existe nível por instância.** O projeto chegou a planejar um, e ele foi
removido com base em evidência: nenhum checkpoint publicado para qualquer espaço
de classes aqui carrega árvore como classe *thing* (o COCO-80 só tem `potted
plant`; o `tree-merged` do COCO-panoptic é stuff; as 1203 categorias do LVIS v1
contêm apenas `Christmas_tree`; o conjunto de 100 things do ADE20K tem `palm` mas
não `tree`), e todo modelo de instância de árvore que dá para baixar é treinado
em imagens aéreas verticais, não em nível de rua. Uma revocação ou AP50 para
árvores individuais só poderia ter sido calculada contra um modelo que não
existe. O ground truth continua sendo anotado uma árvore por vez — essas
instâncias são unidas na máscara semântica contra a qual os dois níveis pontuam.

## Nível 1 — Segmentação semântica (pixels)

Comparação binária árvore-vs-resto entre a máscara refinada predita e a união das
instâncias anotadas, sobre a imagem completa. Os pixels de atribuição e marca
d'água do Street View permanecem intocados tanto na inferência quanto na
avaliação do ground truth.

Imagens cujo backend não consegue expressar uma classe de árvore carregam
`mask_status="unavailable"`. Elas são listadas sob `semantic_skipped_images` e
nunca convertidas numa predição toda de fundo. `mask_status="omitted"` significa,
analogamente, que a exportação da máscara foi desabilitada de propósito.

Reportados por imagem e agrupados:

- **IoU** = VP / (VP + FP + FN)
- **Dice / F1** = 2·VP / (2·VP + FP + FN)
- **precisão** = VP / (VP + FP)
- **revocação** = VP / (VP + FN)

Convenções: métricas agrupadas ("micro") somam as contagens da matriz de confusão
sobre o conjunto inteiro primeiro — são os números de manchete. Médias macro
também são reportadas, com a contagem de imagens que contribuíram. Uma imagem em
que nem a predição nem o ground truth têm qualquer pixel de árvore tem IoU por
imagem *indefinida* (representada como `NaN` durante o cálculo e exportada como
`null` no JSON, contada em `n_images_without_trees_in_both`), em vez de um 1,0
lisonjeiro ou um 0,0 punitivo. Todas as exportações JSON são estritas: nem `NaN`
nem `Infinity` são emitidos.

## Nível 2 — O indicador de cobertura

Comparação direta do número publicado, `tree_coverage_pred` vs
`tree_coverage_gt`, ambos em porcentagem, com `tree_coverage_gt` calculado a
partir da união das anotações sobre a mesma imagem completa da predição.

- **MAE** em pontos percentuais (manchete)
- **RMSE** em pontos percentuais
- **viés** (erro médio com sinal — super/subestimação sistemática)
- erro absoluto máximo, média dos dois lados

O **r de Pearson** é reportado apenas como diagnóstico complementar. Um modelo
que prediz exatamente o dobro da cobertura real tem r = 1,0 e está errado por um
fator de dois; correlação nunca substitui as métricas de erro. O r é omitido
quando um dos lados não tem variância.

Os dois níveis são deliberadamente separados: uma máscara deslocada lateralmente
pode ter IoU ruim e concordância perfeita de cobertura. Os dois fatos importam e
os dois são reportados.

## Divisão experimental

- Backends pré-treinados usados zero-shot com configurações padrão: o conjunto
  rotulado inteiro pode servir como teste.
- No momento em que qualquer parâmetro for ajustado olhando resultados — tamanhos
  de refinamento, limiares de score ou edições na taxonomia —, o conjunto precisa
  ser dividido: um subconjunto de **calibração/validação** para o ajuste e um de
  **teste retido**, tocado uma única vez, no fim. Registre a divisão como listas
  de arquivos ao lado das anotações.
- Qualquer fine-tuning exige a disciplina completa de treino/validação/teste, sem
  nenhum trecho de rua compartilhado entre as divisões (quadros do mesmo local
  são quase duplicatas).
- Sementes, nomes de modelo e configuração vão para o manifesto automaticamente;
  mantenha as listas da divisão e a versão da exportação de anotações no mesmo
  commit dos números reportados.

## Auditoria qualitativa

`--save-artifacts` escreve, por vista: o quadro RGB, a máscara bruta, a máscara
refinada, o overlay de árvore e um JSON de métricas. O relatório de avaliação
carrega linhas por imagem; ordená-las por IoU ou por erro absoluto de cobertura e
abrir as pastas de artefatos correspondentes é o fluxo de trabalho pretendido
para coletar casos de sucesso e de falha.
