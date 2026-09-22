# ADR-003 - Estrategia de threshold

## Status

Aceito

## Contexto

O foco cientifico do projeto e estudar o impacto do threshold na relacao entre deteccao de fraudes e falsos positivos.

## Decisao

Avaliar thresholds em grade configuravel e registrar automaticamente os melhores pontos segundo:

- `max_f1`
- `max_f2`
- `max_recall_with_min_precision`
- `max_precision_with_min_recall`

## Consequencias

- O threshold operacional deixa de ser fixo em `0.5`.
- O trade-off fica visivel e justificavel.
- Restricoes de negocio passam a ser configuraveis.
