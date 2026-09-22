# Do experimento à produção — v1

Segundo estágio do projeto demonstrativo do minicurso.

## Cenário

O projeto usa uma telemetria **100% sintética** para demonstrar um problema simples de classificação de anomalias. Nenhum valor deve ser interpretado como limite operacional de um satélite real.

Na versão inicial (`v0-experiment`), praticamente toda a lógica de Machine Learning estava concentrada em um notebook:

```text
data/telemetry.csv
        ↓
notebooks/00_experiment.ipynb
        ↓
models/model.pkl
```

A pergunta daquela versão era:

> **“Consigo treinar um modelo que funcione?”**

Na `v1-engineering`, o mesmo experimento começa a ser tratado como um projeto de software.

A nova pergunta é:

> **“Consigo organizar, reutilizar, testar e reproduzir melhor esse projeto?”**

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
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── inference.py
│       └── train.py
├── tests/
│   ├── test_data.py
│   └── test_inference.py
├── generate_data.py
├── environment.txt
├── pyproject.toml
├── requirements.lock.txt
├── requirements.txt
└── README.md
```

## O que mudou em relação à v0?

O notebook continua disponível para exploração e experimentação, mas a lógica reutilizável passa a ser organizada em módulos Python.

```text
config.py
    ↓
configurações compartilhadas

data.py
    ↓
leitura e preparação dos dados

train.py
    ↓
treinamento

evaluate.py
    ↓
avaliação

inference.py
    ↓
inferência reutilizável

tests/
    ↓
validação automatizada
```

O treinamento deixa de depender da execução manual das células do notebook e pode ser executado diretamente:

```bash
python -m satellite_ml.train
```

A inferência também passa a existir como uma função reutilizável:

```python
from satellite_ml.inference import predict_one

telemetry = {
    "battery_voltage": 28.1,
    "battery_current": 1.7,
    "battery_temperature": 24.8,
    "solar_panel_current": 4.9,
    "bus_voltage": 28.0,
    "attitude_error": 0.02,
    "eclipse": 0,
}

prediction = predict_one(telemetry)
```

Essa separação será importante nas próximas versões, quando a mesma lógica de inferência passar a ser utilizada por uma API.

## Resultado de referência

No ambiente de referência da `v1-engineering`, o modelo produz aproximadamente:

- **F1:** 0.942
- **Precision:** 0.985
- **Recall:** 0.902

O desempenho alto é intencional: o objetivo não é construir um benchmark científico de detecção de anomalias, mas ter um experimento simples e estável sobre o qual adicionaremos as camadas de engenharia.

## Reprodutibilidade

A `v1` também começa a tornar o ambiente de execução mais explícito.

O `pyproject.toml` declara as dependências do projeto:

```text
Quais bibliotecas o projeto precisa?
```

O `requirements.lock.txt` registra as versões exatas instaladas no ambiente de referência:

```text
Quais versões estavam instaladas quando o projeto foi executado?
```

O `environment.txt` registra as principais versões utilizadas nesta etapa.

Durante a evolução entre `v0` e `v1`, o mesmo código de Machine Learning foi executado em ambientes com versões diferentes das bibliotecas e apresentou uma pequena variação nas métricas.

Isso ilustra uma ideia importante:

> **Reprodutibilidade envolve código, dados e ambiente.**

Controlar apenas uma seed ou `random_state` não é suficiente para garantir resultados idênticos entre ambientes diferentes.

## Testes

A `v1` também introduz testes automatizados.

Atualmente são verificados:

- presença das colunas esperadas no dataset;
- criação dos conjuntos de treino e teste;
- retorno válido da função de inferência.

Para executar:

```bash
pytest -v
```

Os testes não verificam se o modelo é cientificamente adequado. Eles verificam comportamentos esperados do software.

## Executando

Crie um ambiente Python e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Treine o modelo:

```bash
python -m satellite_ml.train
```

Execute os testes:

```bash
pytest -v
```

Para regenerar os dados:

```bash
python generate_data.py
```

## Voltando para a versão anterior

O experimento inicial está preservado pela tag:

```bash
git checkout v0-experiment
```

Para retornar à linha principal do projeto:

```bash
git switch main
```

## Por que esta versão ainda é incompleta?

Agora o projeto está melhor organizado, possui módulos reutilizáveis, testes e informações sobre o ambiente de execução.

Mas novas perguntas aparecem:

1. Como registrar automaticamente cada treinamento realizado?
2. Quais hiperparâmetros produziram determinado resultado?
3. Como comparar diferentes execuções do experimento?
4. Como associar métricas, parâmetros e artefatos ao modelo produzido?
5. Como saber qual modelo deve seguir para a próxima etapa?

Na próxima versão, essas perguntas motivarão a introdução do **MLflow** para rastreamento de experimentos e modelos.