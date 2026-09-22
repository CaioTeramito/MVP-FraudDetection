# Protocolo Experimental

## Objetivo

Executar os experimentos de forma reproduzivel, sem depender exclusivamente de notebooks.

## Pre-condicoes

1. Instalar dependencias com `pip install -r requirements.txt`.
2. Adicionar o dataset real em `data/raw/` ou informar um caminho explicito.
3. Configurar variaveis de ambiente a partir de `.env.example`, se necessario.

## Passo a passo

1. Treinamento:

```bash
python scripts/train.py --dataset data/raw/<dataset>.csv --dataset-name <nome_logico>
```

2. Inspecao do artefato:

```bash
python scripts/evaluate.py --artifact models/fraud_model_artifact.joblib
```

3. Reanalise de thresholds:

```bash
python scripts/threshold_analysis.py --artifact models/fraud_model_artifact.joblib
```

4. Predicao local:

```bash
python scripts/predict.py --artifact models/fraud_model_artifact.joblib --input sample_transaction.json
```

5. API:

```bash
uvicorn app.main:app --reload
```

6. Frontend:

```bash
streamlit run streamlit_app.py
```

## Controles metodologicos

- `train_test_split` com estratificacao.
- `SMOTE` aplicado apenas no treino via `imblearn.Pipeline`.
- Threshold otimizado apenas sobre o conjunto holdout final salvo no artefato.
- Os notebooks permanecem exploratorios e nao substituem os scripts reproduziveis.

## Saidas esperadas

- `models/fraud_model_artifact.joblib`
- `results/metrics/model_comparison.csv`
- `results/threshold/<modelo>_thresholds.csv`
- Figuras em `results/figures/`

## Registro de suposicoes

- Se o dataset nao estiver presente, o experimento deve ser interrompido sem inferir resultados.
- Os valores de `threshold_low` e `threshold_high` no MVP sao parametros demonstrativos de negocio, nao conclusoes cientificas.
