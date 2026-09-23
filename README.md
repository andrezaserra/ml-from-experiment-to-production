# Do experimento à produção — v3

Quarto estágio do projeto demonstrativo do minicurso.

## Cenário

Nas versões anteriores, o projeto evoluiu de um experimento em notebook para um projeto Python estruturado, testável e com gerenciamento de experimentos e modelos por meio do MLflow.

Na `v2-mlflow-registry`, já conseguimos responder:

> **“Qual modelo deve ser utilizado pela aplicação?”**

O Model Registry passou a disponibilizar o modelo:

```text
satellite-anomaly-classifier@champion
```

Mas ainda restava uma pergunta:

> **“Como outro sistema consegue utilizar esse modelo?”**

Na `v3-api`, o modelo passa a ser exposto por meio de uma API HTTP construída com FastAPI.

## Estrutura

```text
ml-from-experiment-to-production/
├── data/
│   └── telemetry.csv
├── models/
│   ├── metrics.json
│   └── model.pkl
├── notebooks/
│   └── 00_experiment.ipynb
├── src/
│   └── satellite_ml/
│       ├── __init__.py
│       ├── api.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── inference.py
│       ├── register.py
│       ├── schemas.py
│       ├── tracking.py
│       └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_inference.py
├── environment.txt
├── pyproject.toml
├── requirements.lock.txt
├── requirements.txt
└── README.md
```

## O que mudou em relação à v2?

Na `v2`, o modelo selecionado podia ser carregado diretamente pelo código Python:

```text
models:/satellite-anomaly-classifier@champion
```

Na `v3`, adicionamos uma camada de aplicação:

```text
Cliente
   │
   │ HTTP + JSON
   ▼
FastAPI
   │
   ├── validação com Pydantic
   ├── GET /health
   └── POST /predict
            │
            ▼
      predict_one()
            │
            ▼
      MLflow Registry
            │
            ▼
         @champion
```

O cliente não precisa conhecer MLflow, scikit-learn ou a implementação interna do modelo.

## Contrato da API

Os contratos de entrada e saída ficam definidos em:

```text
src/satellite_ml/schemas.py
```

A entrada de uma predição segue o schema `TelemetryRequest`:

```json
{
  "battery_voltage": 28.1,
  "battery_current": 1.7,
  "battery_temperature": 24.8,
  "solar_panel_current": 4.9,
  "bus_voltage": 28.0,
  "attitude_error": 0.02,
  "eclipse": 0
}
```

O Pydantic valida automaticamente o formato recebido.

Por exemplo, o campo:

```text
eclipse
```

aceita apenas valores entre `0` e `1`.

Campos desconhecidos também são rejeitados.

## Endpoints

### `GET /health`

Verifica se a aplicação iniciou corretamente e se o modelo foi carregado.

Exemplo de resposta:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Recebe uma amostra de telemetria e executa a inferência.

Exemplo de resposta:

```json
{
  "anomaly": false,
  "prediction": 0,
  "model": "satellite-anomaly-classifier",
  "alias": "champion"
}
```

## Carregamento do modelo

O modelo é carregado uma única vez durante a inicialização da aplicação.

```text
API inicia
   │
   ▼
carrega @champion
   │
   ▼
aguarda requisições
```

Isso evita buscar e carregar novamente o modelo a cada chamada ao endpoint `/predict`.

A URI utilizada é:

```text
models:/satellite-anomaly-classifier@champion
```

## Executando localmente

Ative o ambiente virtual:

```bash
source .venv/bin/activate
```

O MLflow Tracking Server precisa estar disponível:

```bash
mlflow server
```

Por padrão:

```text
http://127.0.0.1:5000
```

Em outro terminal, inicie a API:

```bash
python -m uvicorn satellite_ml.api:app \
  --host 127.0.0.1 \
  --port 8001 \
  --reload
```

Neste ambiente demonstrativo usamos a porta `8001` para evitar conflito com outros serviços locais.

## Testando a API

Health check:

```bash
curl http://127.0.0.1:8001/health
```

Predição:

```bash
curl -X POST \
  http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "battery_voltage": 28.1,
    "battery_current": 1.7,
    "battery_temperature": 24.8,
    "solar_panel_current": 4.9,
    "bus_voltage": 28.0,
    "attitude_error": 0.02,
    "eclipse": 0
  }'
```

## Documentação automática

O FastAPI gera automaticamente uma interface interativa baseada no contrato da aplicação:

```text
http://127.0.0.1:8001/docs
```

Nela é possível visualizar e executar:

```text
GET  /health
POST /predict
```

## Testes automatizados

Os testes da API ficam em:

```text
tests/test_api.py
```

Eles validam:

- health check;
- predição com payload válido;
- rejeição de payload inválido.

Para manter os testes independentes da infraestrutura, o MLflow Registry real é substituído por um modelo simples durante os testes.

```text
execução real
FastAPI → MLflow Registry → @champion

testes
FastAPI → DummyModel
```

Execute:

```bash
pytest -q
```

Nesta versão:

```text
6 passed
```

## API não é o modelo

A API representa uma nova responsabilidade no sistema.

```text
Modelo
    ↓
faz inferência

API
    ↓
define como outros sistemas acessam essa inferência
```

Isso permite separar o código de Machine Learning do contrato utilizado pelos consumidores da aplicação.

## Por que esta versão ainda é incompleta?

Agora outro sistema consegue consumir o modelo por HTTP.

Mas a aplicação ainda depende diretamente do ambiente em que foi configurada:

- Python;
- bibliotecas instaladas;
- versões das dependências;
- sistema operacional;
- configuração do processo;
- acesso ao MLflow.

Surge então uma nova pergunta:

> **“Como empacotar essa aplicação e suas dependências para executá-la de forma consistente em outro ambiente?”**

Na próxima versão, essa pergunta será respondida com **Docker**.

O objetivo da `v4-docker` será transformar a aplicação em um artefato portátil e pronto para deploy.