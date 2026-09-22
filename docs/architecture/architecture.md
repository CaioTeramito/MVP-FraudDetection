# Arquitetura

## Visao geral

O projeto foi reorganizado a partir do notebook `Final.ipynb` para separar claramente experimentacao, treinamento reproduzivel, inferencia, API e dashboard.

## Componentes

- `ml/`: pipeline de dados, preprocessamento, modelos, avaliacao, threshold e inferencia.
- `app/`: camada de aplicacao com schemas, servicos e API FastAPI.
- `scripts/`: comandos reproduziveis para treino, avaliacao, analise de threshold e predicao local.
- `streamlit_app.py`: dashboard simples de demonstracao.
- `docs/`: documentacao tecnica, cientifica e diagramas.
- `results/`: metricas, tabelas de threshold, figuras e relatorios experimentais.

## Responsabilidades

- `ml.data`: carregar dataset, padronizar target e validar entrada.
- `ml.preprocessing`: inferir tipos e montar `ColumnTransformer`.
- `ml.models`: encapsular os modelos candidatos sob uma estrategia comum.
- `ml.training`: treinar, comparar modelos, executar analise de threshold e serializar artefatos.
- `ml.inference`: carregar artefato e executar predicao.
- `app.services`: aplicar regras de negocio e expor servicos reutilizaveis.
- `app.api`: expor endpoints de saude, predicao, threshold e metadata do modelo.

## Fluxo de dados

1. O dataset e carregado e a coluna alvo e padronizada.
2. O conjunto e dividido em treino e teste final.
3. O preprocessamento e ajustado apenas no treino.
4. `SMOTE` e aplicado apenas dentro do pipeline de treinamento.
5. Os candidatos sao comparados com validacao cruzada.
6. O modelo serializado do MVP e escolhido segundo uma metrica configuravel.
7. O threshold operacional e escolhido segundo uma estrategia configuravel.
8. O artefato final alimenta a API e o dashboard sem retreinamento por requisicao.

## Decisoes arquiteturais

- FastAPI foi adotada para fornecer OpenAPI, validacao e simplicidade de integracao.
- Streamlit foi adotado para demonstracao rapida do impacto do threshold.
- O notebook original foi preservado como registro exploratorio, mas a execucao final migrou para scripts.
- O artefato do modelo salva preprocessador, estimador, metricas e saidas do holdout para reavaliacao de threshold.

## Limitacoes

- Os datasets historicos nao estao versionados no repositório em 29/08/2026.
- Sem os dados reais, nao foi possivel reproduzir os experimentos finais do notebook dentro deste workspace.
- O endpoint `/threshold/evaluate` trabalha sobre o holdout salvo no artefato e nao substitui novos experimentos com outro dataset.

## Evolucoes futuras

- autenticacao e autorizacao;
- model registry;
- monitoramento de drift;
- trilha de auditoria;
- CI/CD;
- deployment cloud;
- monitoramento operacional e observabilidade.
