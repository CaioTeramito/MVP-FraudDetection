# Data

Este diretório separa os dados por finalidade:

- `data/raw/`: datasets de entrada originais.
- `data/processed/`: artefatos tabulares derivados do pipeline.
- `data/external/`: dados externos auxiliares, caso existam no futuro.

Observação importante:
Os CSVs referenciados no notebook histórico `Final.ipynb` nao estao presentes no workspace em 29/08/2026. Por isso, o pipeline reproduzível do MVP foi estruturado para funcionar assim que os arquivos forem recolocados nesses diretórios, mas nenhum resultado novo sobre os datasets reais foi inventado.
