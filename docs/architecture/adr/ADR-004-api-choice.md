# ADR-004 - Escolha da API

## Status

Aceito

## Contexto

Era necessario expor predicao, metricas e configuracoes de threshold em uma interface simples e documentada.

## Decisao

Adotar FastAPI como camada HTTP do MVP.

## Consequencias

- OpenAPI e validacao de entrada vem nativamente.
- Facilita testes de integracao.
- Mantem baixo acoplamento com a camada de ML.
