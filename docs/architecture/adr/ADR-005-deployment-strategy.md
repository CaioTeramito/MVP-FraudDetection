# ADR-005 - Estrategia de deployment

## Status

Aceito

## Contexto

O projeto precisa ser demonstravel localmente e preparado para evolucao futura.

## Decisao

Fornecer `Dockerfile` unico e `docker-compose.yml` com dois servicos opcionais: API e frontend.

## Consequencias

- Simplifica demonstracoes.
- Evita infraestrutura desnecessaria para o MVP.
- Mantem um caminho simples para conteinerizacao futura.
