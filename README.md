# Do experimento à produção — v4

Quinto e último estágio implementado do projeto demonstrativo do minicurso.

## Cenário

Na `v3-api`, o modelo selecionado no MLflow Model Registry passou a ser consumido por uma API FastAPI.

A aplicação já podia receber telemetria por HTTP e devolver uma predição:

```text
Cliente
   │
   ▼
FastAPI
   │
   ▼
Modelo
```

Mas ela ainda dependia diretamente do ambiente local:

- versão do Python;
- bibliotecas instaladas;
- sistema operacional;
- configuração do processo;
- disponibilidade do modelo.

A pergunta da `v4-docker` é:

> **“Como empacotar aplicação, runtime, dependências e uma versão conhecida do modelo em um artefato portátil?”**

A resposta desta versão é Docker.

## Evolução completa

```text
v0 — experimento
Notebook
    ↓
“O modelo funciona?”

v1 — engenharia
Projeto Python + testes
    ↓
“Consigo manter e reproduzir?”

v2 — MLOps
MLflow Tracking + Model Registry
    ↓
“Consigo rastrear e gerenciar meus modelos?”

v3 — serviço
FastAPI + Pydantic
    ↓
“Outro sistema consegue consumir o modelo?”

v4 — empacotamento
Docker
    ↓
“Consigo levar a aplicação para outro ambiente?”
```

## Arquitetura da release

O alias `champion` é utilizado para selecionar o modelo que deve entrar na release.

Antes do build, o alias é resolvido para uma versão concreta:

```text
MLflow Model Registry
        │
        ▼
     @champion
        │
        ▼
     Version 1
        │
        ▼
      release/
      ├── model/
      └── model-manifest.json
```

A imagem Docker recebe essa versão específica do modelo.

```text
┌─────────────────────────────────────┐
│ satellite-anomaly-api:v1            │
│                                     │
│ Python 3.12                         │
│ FastAPI + Uvicorn                   │
│ MLflow                              │
│ scikit-learn + skops                │
│ satellite_ml                        │
│                                     │
│ release/                            │
│ ├── model/                          │
│ │   └── Model Version 1             │
│ └── model-manifest.json             │
└─────────────────────────────────────┘
```

## Por que resolver o alias antes do build?

O alias:

```text
champion
```

pode ser alterado no Model Registry.

Já uma versão:

```text
Version 1
```

é uma referência concreta.

Por isso, o processo de release segue:

```text
champion
    ↓
resolve alias
    ↓
Version 1
    ↓
exporta versão
    ↓
docker build
```

Assim, uma mesma imagem Docker não muda de modelo apenas porque o alias do Registry foi atualizado posteriormente.

## Gerando a release

O módulo:

```text
src/satellite_ml/export_model.py
```

resolve o alias `champion`, identifica a versão correspondente e exporta os artefatos para:

```text
release/model/
```

Também é criado:

```text
release/model-manifest.json
```

Exemplo:

```json
{
  "model_name": "satellite-anomaly-classifier",
  "source_alias": "champion",
  "model_version": "1",
  "source_run_id": "fd26fd166edb4015bbcee44c186f537b",
  "model_uri": "models:/satellite-anomaly-classifier/1"
}
```

Para gerar a release:

```bash
python -m satellite_ml.export_model
```

O MLflow Tracking Server precisa estar disponível durante essa etapa, pois é nele que o alias é resolvido e o modelo é obtido.

## Git e artefatos de release

O diretório:

```text
release/
```

é gerado automaticamente e não é versionado no Git.

```text
.gitignore
    ↓
release/ não é versionado
```

Entretanto, ele faz parte do contexto de build do Docker:

```text
.dockerignore
    ↓
release/ não é ignorado
```

Assim, o código do processo é versionado, enquanto o artefato de release é reconstruído a partir do Model Registry.

## Dependências de runtime

A imagem usa:

```text
requirements-runtime.txt
```

Esse arquivo contém apenas as dependências necessárias para executar a aplicação e carregar o modelo.

Ele é diferente do ambiente completo de desenvolvimento:

```text
requirements.lock.txt
    ↓
ambiente utilizado para desenvolvimento

requirements-runtime.txt
    ↓
ambiente necessário para executar o serviço
```

## Dockerfile

O `Dockerfile` empacota:

```text
Python
+ dependências
+ código da aplicação
+ API FastAPI
+ Model Version 1
```

A API é iniciada pelo Uvicorn dentro do container:

```text
0.0.0.0:8000
```

O container também possui um health check baseado no endpoint:

```text
GET /health
```

## Construindo a imagem

Primeiro, gere a release:

```bash
python -m satellite_ml.export_model
```

Depois:

```bash
docker build   -t satellite-anomaly-api:v1   .
```

Confira:

```bash
docker images satellite-anomaly-api
```

## Executando

Neste ambiente demonstrativo, a porta `8000` do host já estava ocupada por outro serviço.

Por isso, utilizamos:

```text
host 8001 → container 8000
```

Execute:

```bash
docker run --rm   --name satellite-anomaly-api   -p 8001:8000   satellite-anomaly-api:v1
```

## Health check

```bash
curl http://127.0.0.1:8001/health
```

Resposta esperada:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

O próprio Docker também verifica o estado do serviço.

```bash
docker ps
```

O container deve aparecer como:

```text
healthy
```

## Predição

```bash
curl -X POST   http://127.0.0.1:8001/predict   -H "Content-Type: application/json"   -d '{
    "battery_voltage": 28.1,
    "battery_current": 1.7,
    "battery_temperature": 24.8,
    "solar_panel_current": 4.9,
    "bus_voltage": 28.0,
    "attitude_error": 0.02,
    "eclipse": 0
  }'
```

Exemplo de resposta:

```json
{
  "anomaly": false,
  "prediction": 0,
  "model": "satellite-anomaly-classifier",
  "version": "1",
  "source_alias": "champion"
}
```

## Runtime independente do Registry

O MLflow Model Registry participa da criação da release:

```text
Tracking
   ↓
Registry
   ↓
champion
   ↓
Version 1
   ↓
release
```

Depois do build, o container utiliza o modelo incluído na própria imagem:

```text
Docker container
   │
   ├── FastAPI
   ├── código
   ├── dependências
   └── Model Version 1
```

A aplicação não precisa consultar o Registry para cada predição.

## Testes

Os testes das versões anteriores continuam disponíveis:

```bash
pytest -q
```

Nesta versão:

```text
6 passed
```

Eles verificam:

- carregamento e separação dos dados;
- inferência;
- health check da API;
- predição via API;
- validação de payload.

## O que construímos

Ao final da parte prática, o projeto passou por toda esta evolução:

```text
Notebook
   ↓
Projeto Python estruturado
   ↓
Testes
   ↓
Experiment Tracking
   ↓
Model Registry
   ↓
FastAPI
   ↓
Docker
```

O resultado é uma aplicação de Machine Learning:

- versionada;
- testável;
- com experimentos rastreáveis;
- com modelo promovido de forma explícita;
- acessível por API;
- empacotada em uma imagem Docker;
- com uma versão conhecida do modelo;
- portátil entre ambientes compatíveis com containers.

> **Neste ponto, o artefato está pronto para deploy.**

## O que ainda falta para produção?

Estar pronto para deploy não significa que todos os requisitos de produção estejam resolvidos.

Uma arquitetura real pode precisar ainda de:

- infraestrutura de cloud;
- banco de dados e armazenamento de objetos;
- Data Lake ou Data Warehouse;
- orquestração de pipelines;
- CI/CD;
- autenticação e gerenciamento de segredos;
- observabilidade;
- escalabilidade;
- alta disponibilidade;
- monitoramento de dados e modelos.

Esses componentes não são implementados neste repositório.

A partir daqui, o minicurso deixa de **construir** e passa a **arquitetar**:

> **Até aqui: construímos. Daqui em diante: arquitetamos.**