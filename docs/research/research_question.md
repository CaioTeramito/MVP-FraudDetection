# Questao de Pesquisa

## Problema

Deteccao de fraude financeira em bases altamente desbalanceadas, nas quais a classe positiva representa uma parcela muito pequena das transacoes.

## Pergunta de pesquisa

Como o ajuste do threshold de decisao altera o equilibrio entre deteccao de fraudes e falsos positivos quando comparado ao threshold padrao de `0.5`?

## Hipotese

Ajustar o threshold de decisao de modelos de Machine Learning pode melhorar o equilibrio entre Recall e Precision na deteccao de transacoes fraudulentas, quando comparado ao threshold padrao de `0.5`.

## Objetivos

- Construir um pipeline reproduzivel para treinamento, avaliacao e inferencia.
- Comparar `Logistic Regression`, `Random Forest` e `XGBoost`.
- Medir o impacto do threshold sobre `Precision`, `Recall`, `F1`, `F2`, `FPR`, `FNR` e `PR-AUC`.
- Disponibilizar um MVP com API e dashboard para demonstracao.

## Variaveis independentes

- Algoritmo de classificacao.
- Threshold de decisao.
- Restricoes de negocio: `minimum_precision` e `minimum_recall`.

## Variaveis dependentes

- `Precision`
- `Recall`
- `F1-score`
- `F2-score`
- `ROC-AUC`
- `PR-AUC`
- `False Positive Rate`
- `False Negative Rate`
- Quantidade de falsos positivos e falsos negativos

## Metricas

Accuracy e mantida apenas como referencia secundaria. Em problemas extremos de desbalanceamento, ela pode permanecer alta mesmo quando o modelo falha em detectar a maior parte das fraudes.

## Metodologia

1. Carregar dataset tabular e padronizar a coluna alvo para `target`.
2. Separar treino e teste com estratificacao.
3. Ajustar preprocessamento apenas no treino.
4. Aplicar `SMOTE` apenas dentro do pipeline de treinamento e validacao cruzada.
5. Treinar os modelos candidatos.
6. Comparar os candidatos com metricas apropriadas.
7. Realizar analise sistematica de thresholds no conjunto holdout final.
8. Serializar o modelo e a configuracao selecionados para inferencia.

## Experimentos

- Comparacao entre modelos com `PR-AUC`, `F2`, `Recall` e `Precision`.
- Analise de threshold em grade `0.05` a `0.95`.
- Busca por:
  - maior `F1`;
  - maior `F2`;
  - maior `Recall` com `Precision >= X`;
  - maior `Precision` com `Recall >= Y`.

## Limitacoes

- Em 29/08/2026, os datasets referenciados pelo notebook historico nao estao presentes no repositório.
- Sem os CSVs reais no workspace, o pipeline pode ser validado estruturalmente, mas nao e possivel gerar novos resultados experimentais sobre os dados originais.
- A interpretabilidade com SHAP depende da disponibilidade do pacote e da compatibilidade do estimador treinado.
