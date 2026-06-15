# 🔭 photoz-ml — Photometric Redshift Estimation with Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Domain](https://img.shields.io/badge/domain-astrophysics%20%2F%20data%20science-blueviolet)

> 🚧 **This project is actively under development.** It is part of my personal portfolio while transitioning into a Data Science / ML role. New models (XGBoost and others), better evaluation, and refactored code are being added progressively. Feedback and suggestions are welcome!

---

## 📑 Table of Contents

- [Overview](#overview)
- [Why this project matters for Data Science](#why-this-project-matters-for-data-science)
- [Project Status & Roadmap](#project-status--roadmap)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Notebooks](#notebooks)
- [Current Results](#current-results)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Versión en Español](#-photoz-ml--estimación-de-redshift-fotométrico-con-machine-learning)

---

## Overview

**Photometric redshift (photo-z) estimation** is a classic high-dimensional **regression problem** in astrophysics: given a handful of brightness measurements of a galaxy through different color filters, predict its *redshift* — a value directly related to its distance and look-back time in the universe.

This project builds an end-to-end pipeline — from raw survey data to trained regression models — using real data from the **Vera C. Rubin Observatory's Data Preview 1 (DP1)**, the precursor dataset to the upcoming LSST survey.

The pipeline currently covers:

1. **Data exploration** of a real astronomical catalog (~84,000 objects, dozens of raw columns).
2. **Feature engineering**: converting raw flux measurements into magnitudes, correcting for interstellar dust extinction, and deriving color indices.
3. **Dimensionality reduction** (PCA) and feature scaling.
4. **Model benchmarking**: training and comparing several regression algorithms with domain-standard evaluation metrics.

## Why this project matters for Data Science

Although the domain is astrophysics, the underlying problem maps directly onto common industry challenges:

- **Tabular regression on noisy, real-world data** — missing values, outliers, and skewed distributions are handled explicitly.
- **Feature engineering from raw measurements** — turning low-level sensor data (fluxes) into physically meaningful features (magnitudes, colors).
- **Correct ML practice** — careful train/test separation, scaling and PCA fit only on training data, avoiding data leakage.
- **Model comparison with robust metrics** — beyond plain error scores, using *bias*, *scatter*, and *outlier rate*, which are conceptually similar to the kind of robust error analysis used in risk modeling and forecasting.
- **Reproducible, packaged code** — reusable functions live in an installable Python package (`photoz_utils`), not just notebook cells.

## Project Status & Roadmap

This is a **living project**. Current focus and next steps:

**Done**
- [x] Exploratory data analysis of the Rubin DP1 photometric catalog
- [x] Magnitude conversion from 5 different flux types (PSF, cModel, Sérsic, GAaP 1.0″, GAaP 3.0″)
- [x] Galactic extinction (reddening) correction per object and per band
- [x] Color index construction (u−g, g−r, r−i, i−z, z−y) and an additional "total magnitude" feature
- [x] Feature scaling and PCA-based dimensionality reduction
- [x] Baseline comparison of 5 regression models (KNN, HistGradientBoosting, ExtraTrees, AdaBoost, Linear Regression)

**In progress / Planned**
- [ ] Add **XGBoost** as an additional regressor and compare against current baselines
- [ ] Hyperparameter tuning (grid/random/Bayesian search) for the best-performing models
- [ ] More rigorous cross-validation and metric reporting
- [ ] Feature importance / explainability analysis (e.g. SHAP)
- [ ] Extend the comparison to the other flux types (cModel, Sérsic, GAaP)
- [ ] Possible neural network (PyTorch) baseline
- [ ] Final write-up summarizing model selection and error analysis

If you're a recruiter or fellow practitioner browsing this repo: the **modeling and evaluation framework is in place**, and the project is being iterated on to add stronger models and a more thorough analysis.

## Repository Structure

```
photoz-ml/
├── data/                          # Not included (see Data section below)
├── images/                         # Figures used in notebooks / README
├── notebooks/
│   ├── 01_data_preprocessing.ipynb        # Catalog exploration & feature engineering
│   └── 02_photometric_redshift_estimators.ipynb  # Model training & evaluation
├── src/
│   └── photoz_utils/
│       ├── __init__.py
│       ├── data_preprocessing.py   # Flux→magnitude, reddening correction, colors, feature builders
│       ├── plotting_functions.py   # Histograms, color-color diagrams, true-vs-predicted plots
│       └── utility_functions.py    # Model training/evaluation helpers (run_regression, metrics)
├── environment.yml                 # Conda environment definition
├── pyproject.toml                  # Editable install configuration
└── README.md
```

*(Structure shown reflects the current layout — it will evolve as the project grows.)*

## Data

This project uses data from the **Vera C. Rubin Observatory Data Preview 1 (DP1)**, a large-scale astronomical catalog with multi-band photometric observations (~84,000 objects in the `ugrizy` optical bands), and a matched subset with spectroscopic redshifts (~9,700 objects) used for supervised training (70/30 train/test split).

> ⚠️ **Raw data is not redistributed in this repository** due to Rubin Observatory data-access policies. The notebooks document exactly how the data is loaded and processed so the pipeline can be reproduced by anyone with access to the DP1 catalog (publicly described at the [DP1 schema documentation](https://sdm-schemas.lsst.io/dp1.html#Object)).

## Notebooks

### `01_data_preprocessing.ipynb` — Catalog Exploration & Feature Engineering
- Loads and explores the raw DP1 object catalog.
- Converts five different flux measurement types into magnitudes.
- Applies a per-object, per-band correction for interstellar dust (Galactic reddening).
- Derives 5 color indices per flux type and visualizes their distributions (histograms, color-color diagrams).
- Builds a combined feature set (6 magnitudes + 5 colors = 11 features per flux type) used as model input.

### `02_photometric_redshift_estimators.ipynb` — Model Training & Evaluation
- Loads a labeled (spec-z matched) subset and builds the same feature set as notebook 01.
- Scales features and reduces dimensionality with PCA (fit on training data only).
- Trains and evaluates several regression models:
  - K-Nearest Neighbors
  - HistGradientBoostingRegressor
  - Extra Trees Regressor
  - AdaBoost Regressor
  - Linear Regression
  - *(XGBoost — in progress)*
- Visualizes predicted vs. true redshift and reports standard photo-z metrics: **bias (Δz)**, **scatter (σz)**, and **outlier rate**.

## Current Results

Baseline comparison (default hyperparameters, PCA-reduced feature space, no tuning yet):

| Model | Bias (Δz) | Scatter (σz) | Outlier rate |
|---|---|---|---|
| HistGradientBoostingRegressor | 0.0001 | 0.0552 | 15.9% |
| K-Nearest Neighbors | -0.0017 | 0.0382 | 18.9% |
| Extra Trees Regressor | 0.0017 | 0.0372 | 21.2% |
| Linear Regression | 0.0187 | 0.2391 | 2.8% |
| AdaBoost Regressor | 0.4151 | 0.3553 | 3.6% |

*Lower bias and scatter are better, but they need to be read together with the outlier rate: Linear Regression and AdaBoost show a low outlier rate mainly because their predictions are compressed into a narrow range — not because they recover redshift accurately. The tree-based and neighbor-based models currently provide the most balanced trade-off. This is exactly the kind of nuance the upcoming tuning and XGBoost comparison aims to address.*

**Example output** (true vs. predicted redshift, HistGradientBoostingRegressor):

![True vs predicted redshift](images/sample_result_hbr_pca.png)

*(Replace with your own up-to-date plot from notebook 02 as the project evolves.)*

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — preprocessing, PCA, regression models, evaluation
- **NumPy** — numerical computing
- **Matplotlib** — visualization
- **tables_io** — reading astronomical HDF5 catalogs
- **Jupyter Notebooks** — exploration and analysis
- **Conda** + editable `pip install` — reproducible environment and package layout

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/joseljimenez/photoz-ml.git
cd photoz-ml

# Create and activate the conda environment
conda env create -f environment.yml
conda activate photoz-ml

# Install the local package in editable mode
pip install -e .

# Launch the notebooks
jupyter lab
```

> 📌 Data access: see the [Data](#data) section — you'll need access to the Rubin DP1 catalog to run the notebooks end to end.

## Acknowledgments

- Data: [Vera C. Rubin Observatory](https://rubinobservatory.org/) — Data Preview 1 (DP1).
- Built as part of my self-directed transition into Machine Learning / Data Science, applying tools and concepts also explored through community photo-z workshops and resources.

## Contact

**Jose Luis Jimenez Apolinar** — Physics MSc candidate transitioning into Data Science / ML

- LinkedIn: [https://www.linkedin.com/in/jos%C3%A9-luis-jim%C3%A9nez-apolinar-953b38244/]
- Email: [joseluisjimenez026@gmail.com]
- Portfolio / other projects: [https://github.com/joseljimenez]

---

# 🔭 photoz-ml — Estimación de Redshift Fotométrico con Machine Learning

> 🚧 **Este proyecto está en desarrollo activo.** Forma parte de mi portafolio personal mientras transiciono hacia un rol de Data Science / ML. Se están agregando progresivamente nuevos modelos (XGBoost, entre otros), mejor evaluación y código refactorizado. ¡Sugerencias y comentarios son bienvenidos!

## Resumen

La **estimación de redshift fotométrico (photo-z)** es un problema clásico de **regresión en alta dimensión** dentro de la astrofísica: a partir de unas pocas mediciones de brillo de una galaxia en distintos filtros de color, se busca predecir su *redshift*, un valor directamente relacionado con su distancia y con qué tan "atrás en el tiempo" se observa.

Este proyecto construye un pipeline completo — desde los datos crudos del survey hasta modelos de regresión entrenados — usando datos reales del **Vera C. Rubin Observatory Data Preview 1 (DP1)**, el dataset precursor del futuro survey LSST.

El pipeline cubre actualmente:

1. **Exploración de datos** de un catálogo astronómico real (~84,000 objetos, decenas de columnas).
2. **Ingeniería de características**: conversión de flujos a magnitudes, corrección por extinción interestelar, y construcción de colores.
3. **Reducción de dimensionalidad** (PCA) y escalamiento de variables.
4. **Comparación de modelos**: entrenamiento y evaluación de varios algoritmos de regresión con métricas estándar del área.

## ¿Por qué este proyecto es relevante para Data Science?

Aunque el dominio es astrofísico, el problema subyacente se traduce directamente en retos comunes de la industria:

- **Regresión sobre datos tabulares ruidosos y reales** — valores faltantes, outliers y distribuciones sesgadas se manejan explícitamente.
- **Ingeniería de características** a partir de mediciones crudas (flujos → magnitudes → colores).
- **Buenas prácticas de ML** — separación correcta train/test, escalamiento y PCA ajustados solo con datos de entrenamiento, evitando *data leakage*.
- **Comparación de modelos con métricas robustas** — además del error simple, se usan *bias*, *scatter* y *outlier rate*, conceptualmente similares al análisis robusto de errores usado en modelos de riesgo y pronóstico.
- **Código reproducible y empaquetado** — las funciones reutilizables viven en un paquete Python instalable (`photoz_utils`), no solo en celdas de notebook.

## Estado del Proyecto y Próximos Pasos

Este es un **proyecto vivo**. Estado actual y siguientes pasos:

**Hecho**
- [x] Análisis exploratorio del catálogo fotométrico Rubin DP1
- [x] Conversión de magnitudes a partir de 5 tipos de flujo (PSF, cModel, Sérsic, GAaP 1.0″, GAaP 3.0″)
- [x] Corrección por extinción galáctica (reddening) por objeto y por banda
- [x] Construcción de índices de color (u−g, g−r, r−i, i−z, z−y) y una característica adicional de "magnitud total"
- [x] Escalamiento de variables y reducción de dimensionalidad con PCA
- [x] Comparación base de 5 modelos de regresión (KNN, HistGradientBoosting, ExtraTrees, AdaBoost, Regresión Lineal)

**En progreso / Planeado**
- [ ] Agregar **XGBoost** como modelo adicional y compararlo contra los modelos base
- [ ] Optimización de hiperparámetros (grid/random/Bayesian search) para los mejores modelos
- [ ] Validación cruzada más rigurosa y reporte de métricas
- [ ] Análisis de importancia de variables / explicabilidad (p. ej. SHAP)
- [ ] Extender la comparación a los demás tipos de flujo (cModel, Sérsic, GAaP)
- [ ] Posible modelo base con redes neuronales (PyTorch)
- [ ] Reporte final con selección de modelos y análisis de errores

Si eres reclutador o colega revisando este repo: el **marco de modelado y evaluación ya está implementado**, y el proyecto se sigue iterando para incorporar modelos más fuertes y un análisis más profundo.

## Datos

Este proyecto usa datos del **Vera C. Rubin Observatory Data Preview 1 (DP1)**, un catálogo a gran escala con observaciones fotométricas multi-banda (~84,000 objetos en las bandas `ugrizy`), y un subconjunto cruzado con redshifts espectroscópicos (~9,700 objetos) usado para el entrenamiento supervisado (70/30 train/test).

> ⚠️ **Los datos crudos no se redistribuyen en este repositorio** debido a las políticas de acceso del Rubin Observatory. Los notebooks documentan exactamente cómo se cargan y procesan los datos, para que el pipeline pueda reproducirse por cualquier persona con acceso al catálogo DP1 (descrito públicamente en la [documentación del esquema DP1](https://sdm-schemas.lsst.io/dp1.html#Object)).

## Notebooks

### `01_data_preprocessing.ipynb` — Exploración del catálogo e ingeniería de características
- Carga y explora el catálogo crudo de objetos DP1.
- Convierte cinco tipos distintos de medición de flujo a magnitudes.
- Aplica una corrección por polvo interestelar (reddening galáctico) por objeto y por banda.
- Construye 5 índices de color por tipo de flujo y visualiza sus distribuciones (histogramas, diagramas color-color).
- Construye un conjunto de características combinado (6 magnitudes + 5 colores = 11 features por tipo de flujo) usado como entrada de los modelos.

### `02_photometric_redshift_estimators.ipynb` — Entrenamiento y evaluación de modelos
- Carga un subconjunto etiquetado (cruzado con redshift espectroscópico) y construye el mismo conjunto de características del notebook 01.
- Escala las características y reduce la dimensionalidad con PCA (ajustado solo con datos de entrenamiento).
- Entrena y evalúa varios modelos de regresión:
  - K-Nearest Neighbors
  - HistGradientBoostingRegressor
  - Extra Trees Regressor
  - AdaBoost Regressor
  - Regresión Lineal
  - *(XGBoost — en progreso)*
- Visualiza el redshift predicho vs. verdadero y reporta las métricas estándar de photo-z: **bias (Δz)**, **scatter (σz)** y **outlier rate**.

## Resultados Actuales

Comparación base (hiperparámetros por defecto, espacio de características reducido con PCA, sin ajuste aún):

| Modelo | Bias (Δz) | Scatter (σz) | Outlier rate |
|---|---|---|---|
| HistGradientBoostingRegressor | 0.0001 | 0.0552 | 15.9% |
| K-Nearest Neighbors | -0.0017 | 0.0382 | 18.9% |
| Extra Trees Regressor | 0.0017 | 0.0372 | 21.2% |
| Regresión Lineal | 0.0187 | 0.2391 | 2.8% |
| AdaBoost Regressor | 0.4151 | 0.3553 | 3.6% |

*Un bias y scatter más bajos son mejores, pero deben leerse junto con el outlier rate: la Regresión Lineal y AdaBoost muestran un outlier rate bajo principalmente porque sus predicciones se comprimen en un rango estrecho — no porque recuperen bien el redshift. Los modelos basados en árboles y vecinos cercanos ofrecen actualmente el mejor balance. Esta es justo la clase de matiz que el próximo ajuste de hiperparámetros y la comparación con XGBoost buscan abordar.*

## Stack Tecnológico

- **Python 3.10+**
- **scikit-learn** — preprocesamiento, PCA, modelos de regresión, evaluación
- **NumPy** — cómputo numérico
- **Matplotlib** — visualización
- **tables_io** — lectura de catálogos astronómicos en HDF5
- **Jupyter Notebooks** — exploración y análisis
- **Conda** + instalación editable con `pip` — entorno y paquete reproducibles

## Configuración e Instalación

```bash
# Clonar el repositorio
git clone https://github.com/<tu-usuario>/photoz-ml.git
cd photoz-ml

# Crear y activar el entorno conda
conda env create -f environment.yml
conda activate photoz-ml

# Instalar el paquete local en modo editable
pip install -e .

# Abrir los notebooks
jupyter lab
```

> 📌 Acceso a datos: ver la sección [Datos](#datos) — necesitarás acceso al catálogo Rubin DP1 para correr los notebooks de principio a fin.

## Agradecimientos

- Datos: [Vera C. Rubin Observatory](https://rubinobservatory.org/) — Data Preview 1 (DP1).
- Desarrollado como parte de mi transición autodirigida hacia Machine Learning / Data Science, aplicando herramientas y conceptos explorados también a través de talleres y recursos comunitarios sobre photo-z.

## Contacto

**Jose Luis Jimenez Apolinar** — Candidato a Maestría en Física en transición hacia Data Science / ML

- LinkedIn: [https://www.linkedin.com/in/jos%C3%A9-luis-jim%C3%A9nez-apolinar-953b38244/]
- Email: [joseluisjimenez026@gmail.com]
- Portafolio / otros proyectos: [https://github.com/joseljimenez]

