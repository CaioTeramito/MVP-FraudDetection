# Diagrama de Caso de Uso

```mermaid
flowchart LR
    A[Analista / Usuário]
    subgraph S[Sistema de Detecção de Fraude]
      UC1((Submeter transação))
      UC2((Consultar resultado da predição))
      UC3((Consultar thresholds avaliados))
      UC4((Reavaliar thresholds sobre o holdout))
      UC5((Consultar métricas do modelo))
      UC6((Consultar informações do modelo))
      UC7((Consultar status do sistema))
    end
    A --> UC1
    A --> UC2
    A --> UC3
    A --> UC4
    A --> UC5
    A --> UC6
    A --> UC7
    UC1 --> UC2
    UC3 --> UC4
```

> O MVP não implementa edição persistente do threshold operacional em tempo real. A funcionalidade disponível é a consulta e a reavaliação de thresholds com base no artefato serializado.
