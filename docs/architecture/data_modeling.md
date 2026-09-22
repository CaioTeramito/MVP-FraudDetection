# Modelagem de Dados

## Decisao atual

O MVP nao introduz banco de dados transacional nem persistencia operacional adicional nesta fase.

## Justificativa

- O foco imediato e reproducao experimental, inferencia e analise de threshold.
- Adicionar banco sem necessidade aumentaria complexidade arquitetural sem beneficio claro para o MVP.

## Persistencia existente

- `models/fraud_model_artifact.joblib`: modelo treinado, preprocessamento, metricas, thresholds e metadata.
- `results/`: tabelas e figuras experimentais derivadas do treinamento.

## Modelo conceitual futuro

Se houver necessidade de persistir auditoria e historico, as entidades candidatas sao:

- `Transaction`
- `Prediction`
- `Model`
- `ThresholdConfiguration`
- `EvaluationResult`

## Observacao

Este documento registra explicitamente que a ausencia de banco de dados e intencional nesta etapa do MVP.
