# undac-thesis-smart-math-tutor

Juego educativo adaptativo de matematicas para primaria, impulsado por Machine Learning.

Proyecto de tesis para construir un tutor inteligente de matematicas usando el dataset **KDD Cup 2010 (Algebra I)**. Incluye limpieza de datos, ingenieria de caracteristicas base, entrenamiento de multiples modelos, seleccion automatica del mejor modelo y API de prediccion con FastAPI.

## Estructura del proyecto

```text
configs/
data/
  external/   # colocar aqui los archivos KDD (train/test/master)
  raw/
  processed/
models/
notebooks/
reports/
  figures/
  metrics/
scripts/
src/
  api/
  data/
  evaluation/
  features/
  models/
  training/
  utils/
main.py
```

## Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Donde colocar los datos

Coloca los archivos de KDD Cup 2010 Algebra en:

- `data/external/algebra_2005_2006_train.txt`
- `data/external/algebra_2005_2006_test.txt`
- `data/external/algebra_2005_2006_master.txt` (recomendado para entrenamiento supervisado)

El pipeline usa primero `*train*.txt` y `*test*.txt` cuando existen.
Si no los encuentra, usa `*master*.txt` como respaldo.

## Ejecucion rapida

1. Limpiar y unificar dataset:

```bash
python main.py clean
```

2. Entrenar modelos y seleccionar el mejor:

```bash
python main.py train
```

Puedes cambiar la metrica de seleccion:

```bash
python main.py train --metric f1
```

3. Levantar API de prediccion:

```bash
python main.py serve
```

Documentacion Swagger disponible en: `http://127.0.0.1:8000/docs`

## Flujo adaptativo (Streamlit)

Entrenar flujo adaptativo con features de dificultad:

```bash
py main_train.py
```

Ejecutar app interactiva:

```bash
py -m streamlit run main_app.py
```

Prueba rapida de inferencia:

```bash
py test_model.py
```

## Modelos entrenados

Se entrenan automaticamente:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM
- Gaussian Naive Bayes

Se guardan en `models/model_{name}.pkl` y metadata en `models/meta.pkl`.

## Nota sobre XGBoost (opcional)

Este scaffold usa Gradient Boosting por defecto para evitar dependencia extra. Si deseas XGBoost, instala manualmente:

```bash
pip install xgboost
```

## Citacion del dataset

Usa esta referencia en la tesis:

```
KDD Cup 2010 Educational Data Mining Challenge.
Algebra I 2005-2006 dataset.
Organized by PSLC DataShop / KDD Cup 2010.
```

Sitio historico de referencia: <https://pslcdatashop.web.cmu.edu/KDDCup/>
