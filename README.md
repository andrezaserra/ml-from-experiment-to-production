# Do experimento à produção — v0

Primeiro estágio do projeto demonstrativo do minicurso.

## Cenário

O projeto usa uma telemetria **100% sintética** para demonstrar um problema
simples de classificação de anomalias. Nenhum valor deve ser interpretado como
limite operacional de um satélite real.

Neste estágio temos apenas:

```text
data/telemetry.csv
        ↓
notebooks/00_experiment.ipynb
        ↓
models/model.pkl
```

A pergunta desta versão é:

> **“Consigo treinar um modelo que funcione?”**

## Estrutura

```text
ml_satellite_minicourse_v0/
├── data/
│   └── telemetry.csv
├── models/
│   ├── metrics.json
│   └── model.pkl
├── notebooks/
│   └── 00_experiment.ipynb
├── generate_data.py
├── requirements.txt
└── README.md
```

## Resultado de referência

Com `random_state=42`, o modelo incluído nesta versão produz aproximadamente:

- **F1:** 0.945
- **Precision:** 1.000
- **Recall:** 0.895

O desempenho alto é intencional: o objetivo não é construir um benchmark
científico de detecção de anomalias, mas ter um experimento simples e estável
sobre o qual adicionaremos as camadas de engenharia.

## Executando

Crie um ambiente Python e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
jupyter notebook
```

Abra `notebooks/00_experiment.ipynb`.

Para regenerar os dados:

```bash
python generate_data.py
```

## Por que esta versão é propositalmente incompleta?

Ela representa o tipo de artefato que frequentemente encerra um experimento:
um notebook, um dataset e um arquivo de modelo.

Nas próximas versões vamos responder, uma a uma, perguntas como:

1. Qual versão do código produziu o resultado?
2. Como rastrear parâmetros, métricas e modelos?
3. Como expor a inferência para outro sistema?
4. Como reproduzir o ambiente?
5. Onde armazenar dados e estado?
6. Como orquestrar o pipeline?
7. Como testar e entregar alterações automaticamente?
8. Como executar e escalar isso em cloud?
