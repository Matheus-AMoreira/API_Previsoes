## Sobre
  Api desenvolvida com flask para utilização do modelo de machine learn baseado em regressão linear criado com PyTorch. O front-end que consome a api se encontra [aqui](https://github.com/Matheus-AMoreira/gerenciador-estoque.git)

# Compose
  Repositório com o arquivo docker-compose para executar todo o projeto no docker está [aqui](https://github.com/Matheus-AMoreira/projeto-projecao) 

# ENV exemplo
```
PG_CONTAINER=db-postgres # Container name in case of compose or the url to the database
PG_USER=postgres
PG_HOST=db-postgres
PG_DATABASE=estoque
PG_PASSWORD=postgres
PG_PORT=5432

DATABASE_URL: postgresql://${PG_USER}:${PG_PASSWORD}@${PG_CONTAINER}:${PG_PORT}/${PG_DATABASE}
```
