# Requisitos do MVP de Detecção de Fraude

## Requisitos Funcionais

- RF01 — O sistema deve permitir submeter uma transação para análise.
- RF02 — O sistema deve validar os dados de entrada antes da predição.
- RF03 — O sistema deve executar o modelo de Machine Learning sobre a transação.
- RF04 — O sistema deve retornar a probabilidade de fraude.
- RF05 — O sistema deve classificar o nível de risco da transação.
- RF06 — O sistema deve gerar uma decisão entre APPROVE, REVIEW e BLOCK.
- RF07 — O sistema deve informar o threshold utilizado na decisão.
- RF08 — O sistema deve permitir consultar os thresholds avaliados.
- RF09 — O sistema deve permitir reavaliar thresholds sobre o holdout salvo no artefato.
- RF10 — O sistema deve suportar estratégias de seleção de threshold: max_f1, max_f2, max_recall_with_min_precision e max_precision_with_min_recall.
- RF11 — O sistema deve disponibilizar métricas de avaliação do modelo.
- RF12 — O sistema deve disponibilizar informações e metadados do modelo.
- RF13 — O dashboard deve permitir visualizar os resultados da predição e da análise de threshold.
- RF14 — O projeto deve permitir executar o treinamento por script reproduzível.
- RF15 — O projeto deve permitir comparar os modelos Logistic Regression, Random Forest e XGBoost.
- RF16 — O sistema deve serializar o modelo, preprocessador, métricas, thresholds e metadados em um artefato.
- RF17 — A API deve disponibilizar um endpoint de health check.

## Requisitos Não Funcionais

- RNF02 — Os dados de entrada devem ser validados antes da inferência.
- RNF03 — O sistema deve manter separação entre experimentação, treinamento, inferência, API e apresentação.
- RNF04 — O treinamento deve ser reproduzível por meio de scripts e parâmetros configuráveis.
- RNF05 — O sistema deve evitar data leakage, aplicando SMOTE somente no pipeline de treinamento.
- RNF06 — O sistema deve preservar a separação entre treinamento e holdout final.
- RNF07 — A mesma entrada, com a mesma versão do modelo e configuração, deve produzir resultado consistente.
- RNF08 — O sistema deve permitir execução local e containerizada.
- RNF09 — O código deve ser modular e manutenível.
- RNF10 — O sistema deve registrar erros e eventos relevantes para diagnóstico.
- RNF11 — Credenciais e configurações sensíveis não devem ser armazenadas diretamente no código-fonte.
- RNF12 — O sistema deve permitir evolução futura para autenticação, model registry, monitoramento de drift, auditoria, CI/CD e cloud.

## Requisitos Experimentais

- RE01 — O sistema deve avaliar o impacto do threshold sobre Precision e Recall.
- RE02 — O sistema deve permitir avaliar uma grade configurável de thresholds.
- RE03 — O sistema deve calcular F1, F2, Precision, Recall, FPR, FNR e PR-AUC quando aplicável.
- RE04 — O sistema deve permitir selecionar thresholds segundo critérios objetivos.
- RE05 — Os resultados experimentais devem ser persistidos em arquivos de resultados.
- RE06 — A inferência operacional deve ser separada do processo de treinamento e experimentação.
