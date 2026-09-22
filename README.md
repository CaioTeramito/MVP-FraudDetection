# Fraud Detection MVP

Projeto de deteccao de fraude financeira com foco cientifico no impacto do ajuste do threshold de decisao sobre o equilibrio entre deteccao de fraudes e falsos positivos.

## Problema

Problemas de fraude financeira costumam ser altamente desbalanceados. Nesses cenarios, usar apenas `accuracy` pode mascarar falhas graves de deteccao, porque o modelo pode acertar quase todas as transacoes legitimas e ainda assim deixar passar boa parte das fraudes.

## Objetivo

Transformar o experimento existente em um MVP completo que:

- compare modelos de Machine Learning;
- serialize um artefato de inferencia;
- exponha API e dashboard;
- permita estudar o efeito do threshold sobre `Precision`, `Recall`, `F1`, `F2`, `FPR`, `FNR` e `PR-AUC`;
- deixe a base pronta para evolucao em TCC/artigo cientifico.

## Hipotese

Ajustar o threshold de decisao pode melhorar o equilibrio entre `Recall` e `Precision` em relacao ao threshold padrao de `0.5`.

## Estado atual do repositório

O repositório original continha apenas o notebook [Final.ipynb](</c:/Users/caiok/OneDrive/Documents/Python/MVP - Fraude/Final.ipynb>), preservado como material exploratorio. Em 29/08/2026, os datasets referenciados nele nao estao presentes no workspace, portanto nenhum resultado novo sobre os dados reais foi inventado.

## Arquitetura

- `app/`: FastAPI, schemas e servicos de aplicacao.
- `ml/`: carga de dados, preprocessamento, modelos, avaliacao, threshold e inferencia.
- `scripts/`: treino, avaliacao, threshold analysis e predicao local.
- `streamlit_app.py`: interface demonstrativa.
- `docs/`: arquitetura, pesquisa, ADRs e diagramas Mermaid.
- `results/`: saidas experimentais reproduziveis.

Detalhes em [architecture.md](docs/architecture/architecture.md) e [data_modeling.md](docs/architecture/data_modeling.md).

## Tecnologias

- Python 3.11
- FastAPI
- Streamlit
- pandas
- scikit-learn
- imbalanced-learn
- XGBoost
- SHAP
- matplotlib
- seaborn
- pytest

## Estrutura do projeto

```text
app/
ml/
data/
docs/
models/
results/
scripts/
tests/
streamlit_app.py
Final.ipynb
```

## Como instalar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e ajuste se necessario.

## Como executar treinamento

```bash
python scripts/train.py --dataset data/raw/<dataset>.csv --dataset-name <nome_logico>
```

Parametros importantes:

- `--model-selection-metric`
- `--threshold-selection-strategy`

## Como executar experimentos

```bash
python scripts/evaluate.py --artifact models/fraud_model_artifact.joblib
python scripts/threshold_analysis.py --artifact models/fraud_model_artifact.joblib
```

## Como executar API

```bash
uvicorn app.main:app --reload
```

Documentacao OpenAPI: `http://localhost:8000/api/v1/docs`

## Como executar frontend

```bash
streamlit run streamlit_app.py
```

## Como executar testes

```bash
pytest
```

## Como analisar thresholds

O projeto avalia thresholds em grade configuravel e salva:

- tabela de thresholds em `results/threshold/`
- comparacao de modelos em `results/metrics/`
- graficos em `results/figures/`

Critérios suportados:

- `max_f1`
- `max_f2`
- `max_recall_with_min_precision`
- `max_precision_with_min_recall`

## Exemplo de uso da API

```json
POST /api/v1/predict
{
  "features": {
    "feature_1": 0.12,
    "feature_2": 1
  }
}
```

Resposta:

```json
{
  "fraud_probability": 0.87,
  "threshold": 0.30,
  "prediction": 1,
  "risk": "HIGH",
  "decision": "BLOCK",
  "model_name": "xgboost",
  "model_version": "0.1.0",
  "top_contributors": []
}
```

Os nomes reais das features sao derivados do dataset treinado e validados pelo artefato. O JSON acima e apenas ilustrativo de formato.

## Diagramas

Os diagramas Mermaid criados estao em `docs/diagrams/`:

- contexto
- caso de uso
- componentes
- sequencia
- classes
- atividades
- estados
- implantacao
- pipeline de ML
- experimento cientifico

## Resultados

- Resultados historicos do experimento original permanecem no notebook `Final.ipynb`.
- Resultados novos e reproduziveis devem ser gerados pelos scripts quando os datasets reais forem recolocados no workspace.
- Os artefatos atualmente presentes em `models/` e `results/` foram gerados em 29/08/2026 com um dataset sintetico de smoke test para validar o software, nao para sustentar conclusoes cientificas.
- Nao foram adicionados numeros experimentais inventados a este repositório.

## Limitacoes

- Datasets reais ausentes no workspace em 29/08/2026.
- O dashboard depende de um artefato previamente treinado.
- A explicabilidade com SHAP pode variar conforme o modelo serializado e a disponibilidade do pacote.

## Trabalhos futuros

- autenticacao e autorizacao;
- model registry;
- auditoria de predicoes;
- monitoramento de data drift e model drift;
- pipelines CI/CD;
- deployment em cloud;
- retreinamento automatizado.

## Relacao com TCC / artigo

O projeto foi estruturado para separar claramente:

- experimentacao exploratoria em notebook;
- experimentos reproduziveis por script;
- artefato operacional para inferencia;
- documentacao metodologica e arquitetural.

Documentos principais:

- [research_question.md](docs/research/research_question.md)
- [experimental_protocol.md](docs/research/experimental_protocol.md)
- [architecture.md](docs/architecture/architecture.md)
