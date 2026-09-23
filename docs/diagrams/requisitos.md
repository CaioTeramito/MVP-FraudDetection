# Requisitos do MVP de Detecção de Fraude

## Requisitos Funcionais

### Essenciais

| ID | Requisito |
|---|---|
| RF01 | O sistema deve permitir submeter uma transação para análise. |
| RF02 | O sistema deve validar os dados de entrada antes da predição. |
| RF03 | O sistema deve executar o modelo de Machine Learning sobre a transação. |
| RF04 | O sistema deve retornar a probabilidade de fraude. |
| RF05 | O sistema deve classificar o nível de risco da transação. |
| RF06 | O sistema deve gerar uma decisão entre APPROVE, REVIEW e BLOCK. |
| RF07 | O sistema deve disponibilizar métricas de avaliação do modelo. |
| RF08 | O sistema deve serializar o modelo, preprocessador, métricas, thresholds e metadados em um artefato. |

### Importantes

| ID | Requisito |
|---|---|
| RF09 | O sistema deve informar o threshold utilizado na decisão. |
| RF10 | O sistema deve permitir consultar os thresholds avaliados. |
| RF11 | O sistema deve permitir reavaliar thresholds sobre o holdout salvo no artefato. |
| RF12 | O sistema deve suportar estratégias de seleção de threshold: max_f1, max_f2, max_recall_with_min_precision e max_precision_with_min_recall. |
| RF13 | O sistema deve disponibilizar informações e metadados do modelo. |
| RF14 | O dashboard deve permitir visualizar os resultados da predição e da análise de threshold. |
| RF15 | O projeto deve permitir comparar os modelos Logistic Regression, Random Forest e XGBoost. |

### Desejáveis

| ID | Requisito |
|---|---|
| RF16 | O projeto deve permitir executar predição local por script. |
| RF17 | A API deve disponibilizar um endpoint de health check. |

---

## Requisitos Não Funcionais

### Essenciais

| ID | Requisito |
|---|---|
| RNF01 | A API deve utilizar HTTP/REST. |
| RNF02 | O sistema deve manter separação entre experimentação, treinamento, inferência, API e apresentação. |
| RNF03 | O treinamento deve ser reproduzível por meio de scripts e parâmetros configuráveis. |
| RNF04 | O sistema deve evitar data leakage, aplicando SMOTE somente no pipeline de treinamento. |
| RNF05 | O sistema deve preservar a separação entre treinamento e holdout final. |
| RNF06 | A mesma entrada, com a mesma versão do modelo e configuração, deve produzir resultado consistente. |

### Importantes

| ID | Requisito |
|---|---|
| RNF07 | O sistema deve permitir execução local e containerizada. |
| RNF08 | O código deve ser modular e manutenível. |
| RNF09 | O sistema deve registrar erros e eventos relevantes para diagnóstico. |
| RNF10 | Credenciais e configurações sensíveis não devem ser armazenadas diretamente no código-fonte. |

### Desejáveis

| ID | Requisito |
|---|---|
| RNF11 | O sistema deve permitir evolução futura para autenticação, model registry, monitoramento de drift, auditoria, CI/CD e cloud. |

---

## Requisitos Experimentais

### Essenciais

| ID | Requisito |
|---|---|
| RE01 | O sistema deve avaliar o impacto do threshold sobre Precision e Recall. |
| RE02 | O sistema deve permitir avaliar uma grade configurável de thresholds. |
| RE03 | O sistema deve calcular F1, F2, Precision, Recall, FPR, FNR e PR-AUC quando aplicável. |
| RE04 | O sistema deve permitir selecionar thresholds segundo critérios objetivos. |
| RE05 | A inferência operacional deve ser separada do processo de treinamento e experimentação. |

### Importantes

| ID | Requisito |
|---|---|
| RE06 | Os resultados experimentais devem ser persistidos em arquivos de resultados. |