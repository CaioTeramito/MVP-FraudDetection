# API

Endpoints principais do MVP:

- `GET /api/v1/health`
- `POST /api/v1/predict`
- `GET /api/v1/model/info`
- `GET /api/v1/thresholds`
- `GET /api/v1/metrics`
- `POST /api/v1/threshold/evaluate`

Observações:

- `POST /predict` recebe um objeto `features` com os nomes reais das colunas presentes no artefato do modelo treinado.
- A OpenAPI fica disponível em `/api/v1/docs`.
- O endpoint de threshold reavalia o conjunto holdout salvo no artefato, sem treinar novamente o modelo.
