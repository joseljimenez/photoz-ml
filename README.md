# Photometric Redshift Estimation with Machine Learning

## Description

This project focuses on developing and evaluating **Machine Learning** models to predict the **photometric redshift** (*photo-z*) of galaxies using simulated photometric data.

The dataset is designed to resemble the type of observations expected from the **Vera C. Rubin Observatory**, particularly the **DP1** catalog.

The main goal is to explore different machine learning regression algorithms for estimating galaxy redshifts using photometric information.

---

## Data and Feature Engineering

In the `01_data_preprocessing.ipynb` notebook, an exploratory data analysis of the dataset is performed, followed by **feature engineering**.

Starting from the photometric fluxes, the following features are constructed:

* **1 total magnitude**, obtained from the total flux of each object.
* **5 colors**, defined as the difference between adjacent photometric magnitudes.

Schematically:

```text
Fluxes
  ↓
Magnitudes
  ↓
Total magnitude + 5 colors
  ↓
Machine Learning features
```

These features are then used as inputs for the regression models.

---

## Machine Learning Models

In the `02_photometric_redshift_estimators.ipynb` notebook, different **supervised learning** regression models are trained to estimate the photometric redshift.

The models are evaluated using the following metrics:

* **Bias:** the mean of the prediction errors.
* **Dispersion:** the standard deviation of the prediction errors.
* **Outlier rate:** the fraction of predictions classified as outliers according to the criterion used in the project.

The initial results are:

| Model                           | Mean Error | Std. Error | Outlier Rate |
| ------------------------------- | ---------: | ---------: | -----------: |
| `KNeighborsRegressor`           |   -0.00017 |     0.0382 |       0.1886 |
| `HistGradientBoostingRegressor` |     0.0001 |     0.0552 |       0.1587 |
| `ExtraTreesRegressor`           |     0.0017 |     0.0372 |       0.2117 |

These results correspond to the initial model configurations, before performing systematic hyperparameter optimization.

---

## Work in Progress

The `03_model_selection_and_nn.ipynb` notebook is currently under development.

The main objectives of this stage are:

* Perform hyperparameter optimization for the previously evaluated models.
* Compare different configurations for each regression algorithm.
* Implement a **neural network as a regression model**.
* Optimize the neural network hyperparameters.
* Compare the performance of the optimized models using the same evaluation metrics.

