> 🇬🇧 **English:** [Read this page in English](../annotation_protocol.md)

# Protocolo de Anotação

O ground truth é produzido manualmente no Roboflow e exportado como **COCO
Instance Segmentation**, uma anotação por árvore individual.

Nada pontua árvores individuais — o pipeline mede área, não contagem — mas a
forma por árvore é mantida porque é o que o Roboflow produz, e porque o ground
truth em nível de pixel é a **união** dessas anotações. Desenhar uma única
máscara fundida seria um segundo ground truth, desenhado à parte, que poderia
discordar deste.

Segmentações em polígono, RLE não comprimido e RLE COCO comprimido são todas
aceitas.

O Roboflow substitui o `file_name` por um nome com hash específico da exportação
e guarda o original em `images[].extra.name`. O avaliador casa pelo nome original
quando ele está presente e mantém o nome com hash para proveniência, de modo que
predições produzidas a partir das imagens de origem continuam casando após uma
reexportação.

Este documento é o contrato de rotulação: todo anotador o segue, e toda métrica
em [`evaluation.md`](evaluation.md) é definida contra ele.

## 1. O que conta como árvore

Uma **árvore** é uma planta lenhosa perene com tronco distinguível (visível ou
plausivelmente ocluído) sustentando uma copa elevada. Palmeiras contam como
árvores.

Explicitamente **não** são árvores:

| Categoria | Exemplos | Anotar? |
|---|---|---|
| Arbustos | cercas vivas, arbustos ornamentais, arbustos com menos de ~2 m sem tronco claro | Não |
| Grama | gramados, canteiros, faixas de grama | Não |
| Trepadeiras / plantas em vaso | trepadeiras em muros, floreiras, canteiros de flores | Não |
| Troncos mortos sem copa | postes nus de árvores removidas | Não |
| Árvores em propriedade privada visíveis da rua | copas de quintal por cima de um muro | **Sim** — o indicador é a copa *visível*, qualquer que seja a posse da terra |
| Árvores em covas / floreiras | arborização viária em covas de calçada | Sim |

Quando o tronco está escondido mas o tamanho, a altura e a textura da copa fazem
de "árvore" a única leitura razoável, anote. Quando for genuinamente ambíguo
entre árvore e arbusto, **não** anote, e registre a imagem na lista de
ambiguidades — subinclusão consistente é mensurável, oscilação não é.

## 2. O que a máscara inclui

A máscara cobre **copa e tronco juntos** — todo pixel que visualmente pertence à
árvore: folhagem, galhos, tronco visível. Ela exclui:

- céu visível **através** da copa onde os vãos reais são discerníveis no zoom de
  rotulação. Vãos internos pequenos (< ~100 px na resolução nativa) podem ser
  fechados pelo polígono; isso combina com o teto de preenchimento de buracos
  pequenos do pipeline, de modo que ground truth e predição erram na mesma
  direção;
- suportes, estacas, protetores, sinalização presa à árvore;
- sombras no chão;
- folhas caídas / detritos no solo.

## 3. Oclusão

Anote **apenas pixels visíveis**. Uma copa dividida por um poste ou por um ônibus
em vários fragmentos visíveis é **uma única anotação** (polígonos COCO permitem
várias partes por instância), desde que os fragmentos claramente pertençam a uma
mesma árvore. Nunca adivinhe pixels atrás de um oclusor.

Se as copas de duas árvores se sobrepõem, divida na melhor fronteira visual;
quando nenhuma fronteira for discernível, atribua os pixels ambíguos à árvore
mais próxima (mais baixa na imagem). A união de pixels — que é o que a cobertura
e as métricas semânticas usam — não é afetada por onde essa fronteira interna
cai.

## 4. Árvores parcialmente fora do quadro

Anote a parte visível, por menor que seja, se ela passar do piso de visibilidade
abaixo. Uma copa entrando pela borda superior da imagem ainda é copa visível.

## 5. Visibilidade mínima

Ignore vegetação que seja:

- menor que **~0,1% da área da imagem** (≈ 400 px num quadro de 640×640); ou
- tão distante/borrada que não dê para decidir entre árvore e outra vegetação com
  zoom de 2×.

Esses pisos existem para que o ground truth não dependa da paciência do anotador.
Eles fazem parte da definição de `tree_coverage_gt`: pixels preditos sobre
vegetação abaixo do piso contarão como falsos positivos, que é a leitura
conservadora pretendida.

## 6. Imagens sem árvores

Mantenha-as, com zero anotações. Imagens negativas são obrigatórias — são o único
jeito de medir o comportamento de falso positivo, e o `tree-ai validate-dataset`
reporta quantas o conjunto contém.

## 7. Categorias

Exporte com uma única categoria chamada `tree`. Se o workspace também rotula
`shrub`/`grass` para outros fins, mantenha-os como categorias separadas; o loader
(`urban_canopy/evaluation/coco.py`) só trata como instâncias de árvore as
categorias chamadas `tree`/`arvore`/`árvore` (configurável).

Regiões `iscrowd` não são usadas: um maciço de árvores denso demais para separar
deve ser anotado como uma instância por copa *distinguível*, ou ignorado e
registrado se nenhuma for distinguível. O validador sinaliza qualquer anotação
`iscrowd` que encontrar, e a avaliação recusa o dataset até que a região seja
resolvida ou removida.

## 8. Processo

- Rotule na resolução nativa da imagem; a exportação precisa manter a mesma
  largura e altura que o pipeline analisou, ou a avaliação recusará o par.
- Um anotador por imagem mais uma passada de revisão em ~20% é o mínimo; registre
  as discordâncias na lista de ambiguidades.
- Versione toda exportação (`annotations_vN.json`) e nunca edite uma exportação
  contra a qual já se avaliou.
