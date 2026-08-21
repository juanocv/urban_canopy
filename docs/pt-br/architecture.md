> 🇬🇧 **English:** [Read this page in English](../architecture.md)

# Arquitetura

O Urban Canopy herda a arquitetura modular do
[`sidewalk_analysis`](https://github.com/juanocv/sidewalk_analysis) e remove tudo
que é específico da medição métrica de calçadas. Este documento registra a
estrutura e o mapeamento exato a partir do projeto de origem.

## Pipeline

```
aquisição (Street View / local)
        │  ImageRequest, cache, metadados do panorama
        ▼
plano de vistas (única / multi-vista)    ← determinístico, guiado por configuração
        │  headings, pitch, fov
        ▼
backend de segmentação          ← OneFormer | Mask2Former | Detectron2 | DeepLab
        │  SegmentationOutput: máscaras de grupo por taxonomia, notas
        ▼
resolução da máscara de árvore           ← classe de árvore, ou proxy de vegetação explícito
        ▼
refinamento conservador                  ← opcional; com trava de crescimento
        ▼
indicadores de cobertura                 ← pixels de árvore / todos os pixels da imagem
        ▼
agregação (multi-vista)                  ← mediana/IQR/p25/p75 sobre as razões de cobertura
        ▼
exportações                              ← artefatos, JSON de métricas, CSV, predições
```

A avaliação é um estágio separado e offline que junta um arquivo de predições com
uma exportação COCO de ground truth (veja [`evaluation.md`](evaluation.md)).

## Módulos

| Módulo | Responsabilidade |
|---|---|
| `core/pipeline.py` | Orquestração; segmentador e cliente de Street View injetados por dependência |
| `core/viewplan.py` | Planos determinísticos de heading (fixed / offsets / equiangular) |
| `core/config.py` | `CanopyConfig`, sementes, manifesto da execução |
| `validation.py` | Limites sem dependências, compartilhados por dataclasses, CLI e API |
| `core/results.py` | Resultados de vista, falhas estruturadas de heading, linhas de CSV |
| `io/streetview.py` | Cliente GSV, cache, geocodificação, `ImageRequest`, metadados do panorama |
| `io/image_io.py` | Decodificação para RGB e overlays |
| `io/artifacts.py` | Artefatos de auditoria por vista, com escrita atômica verificada |
| `io/atomic.py`, `io/json_io.py` | Substituição atômica de arquivos e conversão estrita para JSON |
| `io/geo.py` | Auxiliares geográficos puros |
| `models/taxonomy.py` | Mapeamento espaço de classes → grupo; árvore e vegetação mantidas separadas |
| `models/base.py` | O contrato `SegmentationOutput` e sua validação |
| `models/{oneformer,mask2former,detectron2,deeplab}.py` | Adaptadores de backend |
| `models/factory.py` | Construção preguiçosa de backend; imports de ML opcionais acontecem na construção |
| `processing/coverage.py` | O indicador; semântica de proxy/indisponível |
| `processing/refinement.py` | Limpeza conservadora da copa com trava de crescimento |
| `processing/aggregate.py` | Estatísticas multi-vista robustas |
| `evaluation/*` | Carregamento COCO, RLE, os dois níveis de métrica, runner, intercâmbio |
| `cli/`, `webapi.py` | Interfaces |

## Mapeamento a partir do sidewalk_analysis

### Reaproveitado (inalterado ou levemente adaptado)

| Componente | Destino |
|---|---|
| `StreetViewClient`, `ImageRequest`, geocodificação, cache joblib | **Reaproveitado**; acrescentado o registro de id/data do panorama e requisição hasheável |
| `io/geo.py` | **Reaproveitado** literalmente |
| Decodificação de imagem | **Adaptado**; `decode_rgb`, `from_bgr_array` e `ensure_rgb_u8` mantêm explícitas as entradas codificadas, BGR e RGB, enquanto os quadros do Street View permanecem intactos |
| `log.py` (logging texto/JSON) | **Reaproveitado** (`SWAI_*` → `UC_*`) |
| `diagnostics.py` | **Reaproveitado**, enxugado para as dependências relevantes |
| Padrão de settings (pydantic-settings, `.env`) | **Reaproveitado**, incluindo a correção de regressão do `extra="ignore"` e seu teste |
| Exports preguiçosos de pacote (PEP 562) | **Reaproveitado** |
| Factory com imports preguiçosos + dicas de instalação | **Reaproveitado** |
| Gerenciamento de dispositivo (`--device auto/cpu/cuda`, falha cedo) | **Reaproveitado** |
| Loader de checkpoint do DeepLab (inferência de arquitetura, guarda de casamento de tensores) | **Reaproveitado** |
| Estrutura da CLI, correções de console no Windows | **Reaproveitado**, mais os subcomandos |
| Estrutura da Web API (lifespan, registry, semáforo, CORS) | **Reaproveitado**, rechaveado na configuração de copa |
| Filosofia de testes offline/somente CPU, marcadores `gpu`/`network` | **Reaproveitado** |
| Workflow de CI, scripts de check/setup | **Reaproveitado** |

### Adaptado (mesma ideia, domínio novo)

| Componente | Mudança |
|---|---|
| Protocolo `Segmenter` (tupla de 4) | → dataclass `SegmentationOutput`: máscaras de grupo guiadas por taxonomia, notas de proveniência |
| Wrappers de backend | Alvo desacoplado de `sidewalk`; auditoria de espaço de classes por backend; **nenhum refinamento dentro dos adaptadores** (o projeto de origem chamava `shave_above_top_envelope` lá; o refinamento agora é um estágio explícito do pipeline) |
| Sinônimos de rótulo do `AliasSegmenter` | → `Taxonomy` (dados, serializável, sobrescrevível por estudo) |
| Agregação multi-vista (mediana de largura) | → estatísticas robustas sobre razões de cobertura |
| Artefatos de debug | → diretórios estruturados de artefatos por vista |

### Removido (não levado para o caminho de inferência)

Segmentação e refinamento de calçada (`refine_sidewalk_mask`, RANSAC de linha de
meio-fio, preenchimento de ponte, `shave_above_top_envelope`); extração de
obstáculos e bases de contato; MiDaS; ZoeDepth; todo o caminho de profundidade
(`DepthScale`, `to_metric_depth`, escalas de fallback); modelo de câmera e
conversão pixel→metro; estimativa de largura (`WidthResult`, `compute_width`);
gabaritos (`ClearanceResult`, `compute_clearances`); métricas e classificações de
acessibilidade da NBR 9050; fusão de máscaras em ensemble (fundir máscaras de
espaços de classes diferentes não faz sentido para uma medição cujo valor *é* a
área da máscara — a comparação entre backends acontece na avaliação); e a busca
do centro da rua guiada por máscara (`_find_street_center`).

Essa última remoção é uma questão de correção, não de simplificação: escolher
headings segmentando quadros de sondagem faria a amostra depender do modelo que
está sendo medido. Os planos de heading aqui são determinísticos e cegos às
imagens.

## Contratos que vale conhecer

- **`tree_source`** em todo resultado: `tree_class`, `vegetation_proxy`
  (explicitamente solicitado, sinalizado) ou `unavailable` (nunca zero em
  silêncio).
- **O espaço de classes segue o checkpoint**, não o backend, para OneFormer e
  Mask2Former — ambos publicam pesos para vários datasets. `infer_class_space()`
  lê o token do dataset no nome do modelo e seleciona a taxonomia a partir dele,
  de modo que um checkpoint Cityscapes recebe uma taxonomia sem grupo de árvore e
  reporta cobertura como indisponível. Um nome que não casa com nenhum dataset
  conhecido é recusado em vez de assumir um padrão: aplicar uma taxonomia ADE20K
  a classes desconhecidas rotularia errado cada pixel, em silêncio.
- **Consistência de taxonomia**: todo adaptador rejeita uma taxonomia de outro
  espaço de classes antes de importar ou baixar seu modelo. Aliases usam a mesma
  normalização dos rótulos preditos; nomes de grupo duplicados e aliases ambíguos
  são rejeitados a menos que `alias_priority` resolva o conflito.
- **Área, nunca contagem.** Não há saída por instância: nenhum checkpoint
  publicado para esses espaços de classes tem árvore como classe *thing*, e todo
  modelo de instância de árvore que dá para baixar é de imagem aérea vertical. O
  ground truth continua anotado por árvore e unido na máscara semântica.
- **Denominador de quadro completo**: os quadros do Street View, incluindo os
  pixels de atribuição e marca d'água, permanecem intactos. A cobertura é sempre
  dividida por `H * W`.
- **Status da máscara na predição**: `available`, `unavailable` (o espaço de
  classes não tem classe de árvore) ou `omitted` (exportação de máscara
  desabilitada). Só máscaras disponíveis entram nas métricas semânticas.
- **Intercâmbio de predições**: RLE COCO não comprimido, legível com ou sem
  pycocotools, manifesto embutido.
- **Mínimo multi-vista**: um plano exige `min_successful_views >= 1`. Cada heading
  que falha registra estágio, tipo de exceção e mensagem; cair abaixo do mínimo
  levanta `MultiViewAnalysisError` em vez de retornar uma execução vazia.
- **JSON estrito**: métricas numéricas indefinidas permanecem indefinidas em
  memória e são exportadas como `null` no JSON; `NaN` e `Infinity` nunca são
  escritos.
- **Saídas atômicas**: quadros de cache, JSON, CSV e artefatos de imagem são
  escritos num arquivo temporário irmão e substituem o alvo atomicamente apenas
  após sucesso.
- **Retenção limitada de RGB**: `keep_rgb` é falso por padrão. Lotes da CLI
  produzem um resultado por vez e liberam o RGB depois dos artefatos por vista; a
  API o retém apenas para requisições `/analyse/single` que pedem overlays.
- **Níveis de reprodutibilidade**: o seeding do RNG e os algoritmos
  determinísticos do Torch são campos separados do manifesto. `PYTHONHASHSEED` é
  observado, nunca atribuído após a inicialização, e a identidade bit a bit entre
  stacks não é garantida — explicitamente.
