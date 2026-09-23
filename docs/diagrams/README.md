# Diagramas UML e Requisitos — MVP de Detecção de Fraude

Pacote preparado a partir da estrutura e documentação do projeto.

## Diagramas

1. Caso de uso
2. Atividade — predição
3. Sequência — predição
4. Classes
5. Componentes
6. Implantação
7. Estados — predição
8. Experimento científico

Os arquivos `.puml` são destinados à renderização em ferramentas compatíveis com PlantUML. Os arquivos `.md` contêm versões Mermaid para documentação em Markdown/GitHub.

## Requisitos

`09_requisitos.md` contém requisitos funcionais, não funcionais e experimentais.

## Observação metodológica

Os diagramas foram ajustados para refletir o comportamento real do MVP: a análise e a reavaliação de thresholds ocorrem em ambiente de experimentação/seriação do artefato, enquanto a inferência operacional usa o threshold salvo no artefato serializado. A alteração persistente do threshold operacional em tempo real não é uma funcionalidade implementada no código atual.
