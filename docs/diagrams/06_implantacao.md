# Diagrama de Implantação

```mermaid
flowchart TD
    C[Cliente / Navegador]
    FE[Streamlit Dashboard]
    API[FastAPI]
    ML[Camada de Inferência ML]
    ART[(fraud_model_artifact.joblib)]
    TRAIN[Ambiente de Treinamento / Avaliação]
    DATA[(Dataset)]
    RES[(resultados e thresholds)]

    C --> FE
    C --> API
    FE --> API
    API --> ML
    ML --> ART
    TRAIN --> DATA
    TRAIN --> ART
    TRAIN --> RES
    FE --> ART
    API --> ART
```

A implantação segue a estratégia do projeto: frontend Streamlit e API FastAPI consumindo um artefato serializado previamente treinado. A etapa de treinamento e reavaliação de thresholds acontece em ambiente separado e gera o artefato e os resultados experimentais, sem alterar o comportamento da inferência em tempo real.
