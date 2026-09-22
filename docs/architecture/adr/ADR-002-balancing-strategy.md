# ADR-002 - Estrategia de balanceamento

## Status

Aceito

## Contexto

O notebook original ja utilizava `SMOTE`, mas era necessario evitar vazamento de dados.

## Decisao

Aplicar `SMOTE` somente dentro do pipeline de treino e validacao cruzada com `imblearn.Pipeline`, apos o `train_test_split`.

## Consequencias

- Reduz risco de data leakage.
- Mantem compatibilidade conceitual com o experimento original.
- Facilita reproducao metodologica em TCC/artigo.
