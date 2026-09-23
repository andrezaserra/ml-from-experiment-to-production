# Do experimento à produção — v2

Terceiro estágio do projeto demonstrativo do minicurso.

## Cenário

O projeto usa uma telemetria **100% sintética** para demonstrar um problema simples de classificação de anomalias. Nenhum valor deve ser interpretado como limite operacional de um satélite real.

Na `v1-engineering`, o experimento foi reorganizado como um pequeno projeto Python, com módulos reutilizáveis, testes e ambiente de execução mais explícito.

A pergunta daquela versão era:

> **“Consigo organizar, reutilizar, testar e reproduzir melhor esse projeto?”**

Na `v2-mlflow`, surge um novo problema:

> **“Como rastrear sistematicamente cada treinamento, seus parâmetros, métricas, artefatos e modelos?”**

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
│       ├── tracking.py
│       └── train.py
├── tests/
│   ├── test_data.py
│   └── test_inference.py
├── environment.txt
├── pyproject.toml
├── requirements.lock.txt
├── requirements.txt
└── README.md
```

Os arquivos locais do MLflow, como `mlflow.db` e `mlartifacts/`, não são versionados no Git.

## O que mudou em relação à v1?

Na `v1`, cada treinamento produzia arquivos locais:

```text
treinamento
    ↓
model.pkl
metrics.json
```

Isso funciona, mas não preserva de forma estruturada o contexto de cada execução.

Na `v2`, cada treinamento cria um **run** no MLflow:

```text
treinamento
    ↓
MLflow Run
    ├── parâmetros
    ├── métricas
    ├── tags
    ├── artefatos
    └── modelo
```

Cada execução passa a ter identidade própria e pode ser consultada posteriormente.

## Conceitos principais

### Experiment

Agrupa diferentes execuções relacionadas ao mesmo problema.

Neste projeto:

```text
satellite-anomaly-detection
```

### Run

Representa uma execução individual do treinamento.

Exemplos:

```text
rf-small
rf-baseline
rf-large
validation-v2
```

### Parameters

Descrevem como o modelo foi configurado antes do treinamento.

Exemplos:

```text
n_estimators
max_depth
class_weight
random_state
```

### Metrics

Representam os resultados medidos após o treinamento.

Neste projeto:

```text
f1
precision
recall
```

### Artifacts

São arquivos ou objetos produzidos por uma execução.

Exemplos:

```text
metrics.json
modelo treinado
assinatura do modelo
exemplo de entrada
```

## Tracking

A configuração do MLflow foi separada em:

```text
src/satellite_ml/tracking.py
```

Esse módulo define onde o tracking é realizado e qual experimento será utilizado.

Por padrão, o projeto espera um MLflow Tracking Server em:

```text
http://127.0.0.1:5000
```

A URI pode ser alterada por meio da variável de ambiente:

```bash
export MLFLOW_TRACKING_URI=http://outro-servidor:5000
```

## Executando o MLflow

Ative o ambiente virtual:

```bash
source .venv/bin/activate
```

Inicie o servidor:

```bash
mlflow server
```

A interface estará disponível em:

```text
http://127.0.0.1:5000
```

## Executando experimentos

Um treinamento com os parâmetros padrão:

```bash
python -m satellite_ml.train \
  --run-name baseline-v2
```

Um modelo menor:

```bash
python -m satellite_ml.train \
  --n-estimators 100 \
  --max-depth 6 \
  --run-name rf-small
```

Um modelo intermediário:

```bash
python -m satellite_ml.train \
  --n-estimators 200 \
  --max-depth 10 \
  --run-name rf-baseline
```

Um modelo maior:

```bash
python -m satellite_ml.train \
  --n-estimators 300 \
  --max-depth 14 \
  --run-name rf-large
```

Cada comando gera um novo run no experimento `satellite-anomaly-detection`.

## Resultado de referência

Um exemplo de execução da `v2`:

```text
Run:        validation-v2
F1:         0.9382
Precision:  0.9773
Recall:     0.9021
```

Os valores podem variar de acordo com os hiperparâmetros utilizados.

O objetivo não é escolher o melhor modelo, mas demonstrar como o MLflow permite comparar diferentes execuções de forma estruturada.

## Um run também pode falhar

Durante o desenvolvimento da `v2`, uma primeira tentativa de registrar o modelo falhou durante a serialização.

Esse run permaneceu registrado no MLflow com status de falha.

Isso ilustra uma característica importante:

> **Experiment tracking registra o processo experimental, e não apenas os resultados bem-sucedidos.**

Runs com falha também ajudam a preservar o histórico do desenvolvimento.

## Tracking não é deployment

Na `v1`, o arquivo:

```text
models/model.pkl
```

representava o modelo local de referência.

Na `v2`, novos experimentos são registrados no MLflow, mas não substituem automaticamente esse modelo.

```text
treinamento experimental
        ↓
      MLflow
        ↓
novo modelo registrado

models/model.pkl
        ↓
modelo de referência permanece inalterado
```

Isso reforça uma distinção importante:

```text
Training
    ↓
gera um modelo

Tracking
    ↓
registra o que aconteceu

Promotion / Deployment
    ↓
decide qual modelo será utilizado
```

Executar um novo experimento não significa automaticamente colocar aquele modelo em produção.

## Testes

Os testes introduzidos na `v1` continuam válidos:

```bash
pytest -v
```

Atualmente são verificados:

- presença das colunas esperadas no dataset;
- criação dos conjuntos de treino e teste;
- retorno válido da função de inferência.

## Reprodutibilidade

A `v2` atualiza também o ambiente de referência.

O `pyproject.toml` passa a incluir o MLflow como dependência do projeto.

O `requirements.lock.txt` registra as versões exatas instaladas no ambiente utilizado para esta etapa.

O `environment.txt` registra as principais versões, incluindo o MLflow.

## Por que esta versão ainda é incompleta?

Agora conseguimos rastrear experimentos, comparar configurações e preservar métricas, artefatos e modelos associados a cada execução.

Mas ainda existem novas perguntas:

1. Como outro sistema pode enviar dados e obter uma predição?
2. Como expor a inferência por uma interface padronizada?
3. Como validar os dados recebidos antes de enviá-los ao modelo?
4. Como desacoplar o código cliente da implementação interna do Machine Learning?

Na próxima versão, essas perguntas motivarão a criação de uma **API com FastAPI**.