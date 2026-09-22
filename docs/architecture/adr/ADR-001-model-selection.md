# ADR-001 - Estrategia de selecao do modelo

## Status

Aceito

## Contexto

O MVP precisa servir previsoes com um artefato unico, mas o projeto tambem precisa comparar varios candidatos experimentalmente.

## Decisao

Comparar os candidatos (`Logistic Regression`, `Random Forest`, `XGBoost`) em `results/metrics/model_comparison.csv` e escolher o artefato operacional com base em uma metrica configuravel (`model_selection_metric`).

## Consequencias

- O sistema continua extensivel para novos modelos.
- A escolha do modelo operacional fica rastreavel e reproduzivel.
- O usuario pode alterar o criterio sem reescrever a arquitetura.
