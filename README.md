# machine-learning-101

> Colección de proyectos introductorios de Machine Learning en Python. Actualmente incluye predicción de supervivencia del Titanic (Kaggle) y conversión Celsius ↔ Fahrenheit con regresión lineal.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)

## Tabla de Contenidos

- [Características](#características)
- [Stack](#stack)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Tests](#tests)
- [Configuración](#configuración)
- [CI](#ci)
- [Datos](#datos)
- [Limitaciones / Roadmap](#limitaciones--roadmap)
- [Licencia](#licencia)

## Características

- Predicción de supervivencia en Titanic con scikit-learn (clasificación binaria)
- Análisis exploratorio de datos (EDA) con pandas, seaborn y matplotlib
- Conversión Celsius ↔ Fahrenheit mediante regresión lineal (`conversion/`)
- Pipeline de preprocesamiento: limpieza de features, encoding, train/test split
- Modelos exportables y reutilizables
- Dataset de Titanic incluido (descargado de Kaggle)

## Stack

- **Python 3.11+**
- **pandas** — manipulación de datos
- **numpy** — cómputo numérico
- **scikit-learn** — modelos ML (regresión, clasificación)
- **seaborn / matplotlib** — visualización

## Arquitectura

```
machine-learning-101/
├── .env.example
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── app.py                   # EDA y preprocesamiento Titanic
├── conversion/
│   └── app.py               # Regresión lineal Celsius ↔ Fahrenheit
├── titanic/
│   ├── train.csv            # Dataset de entrenamiento (Kaggle)
│   ├── test.csv             # Dataset de prueba
│   └── gender_submission.csv
└── tests/
    └── test_smoke.py
```

## Requisitos

- Python 3.11 o superior

## Instalación

```bash
git clone <repo-url>
cd machine-learning-101
pip install -r requirements.txt
```

## Uso

### Exploración de datos Titanic

```bash
python app.py
# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns)
# ...
```

### Conversión de temperatura

```bash
cd conversion
python app.py
# Precisión del modelo: 1.0
# 30 grados Celsius son 86.0 grados Fahrenheit.
```

## Tests

```bash
pip install pytest ruff
pytest -q
```

## Configuración

No requiere variables de entorno. Ver `.env.example`:

```
# No environment variables required for this project.
```

## CI

GitHub Actions — Push/PR:

- `ruff check`
- `pytest`

## Datos

El dataset de Titanic proviene de [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic). Archivos en `titanic/`:

| Archivo               | Descripción                     |
| --------------------- | ------------------------------- |
| `train.csv`           | Datos de entrenamiento (891 obs) |
| `test.csv`            | Datos de prueba (418 obs)       |
| `gender_submission.csv` | Ejemplo de submission          |

Los CSVs están en `.gitignore` — descargar manualmente si se clona fresco.

## Limitaciones / Roadmap

- Solo un proyecto principal (Titanic) implementado
- `app.py` realiza solo EDA básico — sin entrenamiento de modelo
- Sin notebooks Jupyter (todo en scripts .py)
- A futuro: más datasets, pipelines completos, deployment

## Licencia

MIT — ver [LICENSE](LICENSE).
