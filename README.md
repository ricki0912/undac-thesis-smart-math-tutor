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

Banco de preguntas del juego:

- `data/raw/question_bank.csv` con columnas:
  - `level`
  - `text`
  - `answer`
  - `hint`

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

3. Levantar backend local:

```bash
python main.py serve
```

Interfaz local HTML + JavaScript (Bootstrap local): `http://127.0.0.1:8000/`

Documentacion Swagger disponible en: `http://127.0.0.1:8000/docs`

## Interfaz Web Local (sin nube)

La carpeta `web/` contiene:

- `web/index.html`: interfaz de login, modo jugador y modo administrador.
- `web/app.js`: logica frontend, consumo de endpoints locales.
- `web/styles.css`: estilos del juego.
- `web/vendor/bootstrap.min.css` y `web/vendor/bootstrap.bundle.min.js`: Bootstrap local (offline).

Todo corre en tu maquina y llama al backend Python local (FastAPI), sin servicios cloud.

## Flujo adaptativo (Streamlit, opcional/legacy)

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

## Diccionario de datos del modelo

### Entrada base (dataset limpio)

- `student_id`: identificador del estudiante (trazabilidad e historial).
- `step_name`: texto de la ecuacion/paso matematico.
- `incorrects`: numero de errores cometidos.
- `hints`: numero de pistas usadas.
- `correct_first_attempt`: indicador binario (1 si acerto al primer intento, 0 si no).
- `step_duration_sec`: tiempo de resolucion del intento en segundos.
- `timestamp`: marca temporal del evento (si existe en origen).
- `source_split`: origen del dato (`train`, `test`, `master`, `user_gameplay`).

### Features de entrada al modelo (actuales)

- `incorrects`
- `hints`
- `step_duration_sec`
- `correct_first_attempt`
- `error_rate`
- `time_efficiency`
- `difficulty_score`
- `student_attempt_count_prev`
- `student_avg_incorrects_prev`
- `student_avg_time_prev`
- `student_accuracy_prev`
- `step_success_rate_prev`
- `step_len`
- `step_num_ops`
- `step_has_parentheses`
- `step_num_digits`
- `step_num_variables`
- `step_abs_constant_sum`

### Salida del modelo

- `difficulty_level`: clase numerica (`0`, `1`, `2`).
- `difficulty_label`: etiqueta interpretada (`baja`, `media`, `alta`).
- `probability`: confianza del modelo para la prediccion.
- `model_inputs`: snapshot de las features usadas en la inferencia para auditoria.

## Preprocesamiento aplicado

El flujo de preprocesamiento y construccion de features es:

1. Carga de datos:
- Se prioriza `data/external/*train*.txt` y `*test*.txt`.
- Si no existen, se usa `*master*.txt`.
- Se agregan tambien logs de juego en `data/raw/gameplay_logs.csv` para reentrenamiento continuo.

2. Limpieza y estandarizacion:
- Normalizacion de columnas base (`student_id`, `step_name`, `incorrects`, `hints`, `correct_first_attempt`, `step_duration_sec`).
- Coercion numerica segura (`errors='coerce'`) y relleno de faltantes.
- Recorte de rangos invalidos:
  - `incorrects >= 0`
  - `hints >= 0`
  - `step_duration_sec >= 0`
  - `correct_first_attempt` en `{0,1}`.

3. Orden temporal para evitar fuga de informacion:
- Se crea `event_order` usando `timestamp` cuando esta disponible.
- Si no hay `timestamp`, se usa el orden secuencial del dataset.

4. Feature engineering base:
- `error_rate = incorrects / (incorrects + hints + 1)`
- `time_efficiency = correct_first_attempt / (step_duration_sec + 1)`
- `difficulty_score` como combinacion normalizada de errores, pistas, tiempo y acierto.

5. Features historicas por estudiante (solo pasado):
- `student_attempt_count_prev`
- `student_avg_incorrects_prev`
- `student_avg_time_prev`
- `student_accuracy_prev`

6. Features historicas por tipo de paso:
- `step_success_rate_prev` (tasa previa de acierto para ese `step_name`).

7. Features estructurales de la ecuacion:
- Longitud del texto (`step_len`)
- Numero de operadores (`step_num_ops`)
- Presencia de parentesis (`step_has_parentheses`)
- Conteo de digitos (`step_num_digits`)
- Conteo de variables (`step_num_variables`)
- Suma absoluta de constantes (`step_abs_constant_sum`)

8. Construccion de target:
- `difficulty_level` se deriva de `difficulty_score` por cuantiles (0=baja, 1=media, 2=alta).

9. Esquema de validacion:
- Si hay `source_split=train/test`, se respeta ese corte.
- Si no, se aplica `temporal_holdout_split` con `event_order` para no mezclar futuro/pasado.
- Como ultimo respaldo, split aleatorio estratificado.

## Modelos entrenados

Se entrenan automaticamente:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM
- Gaussian Naive Bayes

Se guardan en `models/model_{name}.pkl` y metadata en `models/meta.pkl`.

Metricas principales (multiclase) reportadas por modelo:

- `accuracy`
- `precision` (weighted)
- `recall` (weighted)
- `f1_weighted`
- `f1_macro`
- `auc_ovr_weighted`
- `auc_ovr_macro`

El panel administrador web muestra estas metricas y las graficas por modelo.

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
