# undac-thesis-smart-math-tutor

Juego educativo adaptativo de matematicas para primaria, impulsado por Machine Learning.

Proyecto de tesis para construir un tutor inteligente de matematicas usando el dataset **KDD Cup 2010 (Algebra I)**. Incluye limpieza de datos, ingenieria de caracteristicas base, entrenamiento de multiples modelos, seleccion automatica del mejor modelo y API de prediccion con FastAPI.

## Estructura del proyecto

```text
configs/
data/
  external/   # colocar aqui los archivos KDD (train/test/master)
  raw/
models/
notebooks/
reports/
  figures/
  metrics/
scripts/
src/
  api/
  evaluation/
  models/
  utils/
main.py
```

## Notebooks

- `notebooks/01_data_exploration.ipynb`: guía reproducible del preprocesamiento y del pipeline real de entrenamiento (DataLoader → FeatureEngineer → ModelTrainer), incluyendo validación y artefactos generados.

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

1. Entrenar el modelo del tutor:

```bash
python main.py train
```

Entrenamiento rapido (para pruebas) y sin figuras:

```bash
python main.py train --max-rows 50000 --no-plots
```

2. Levantar backend local:

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

## Prueba rapida de inferencia

```bash
python test_model.py
```

## Diccionario de datos del modelo

### Entrada base (dataset del tutor)

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

### Payload mostrado en UI/API (ejemplo)

En el modal del jugador/admin veras un payload como:

- `inputs_modelo`: diccionario con las **features finales** (equivale a `FeatureEngineer.feature_columns`).
- `salida_modelo`: salida del modelo (`difficulty_level`, `difficulty_label`, `probability`).
- `accion_adaptativa`: decision de la politica (`subir|mantener|bajar`) despues del intento.
- `nivel_antes`, `nivel_despues`: nivel del juego antes/despues de aplicar la politica.
- `puntaje_ronda`: puntos otorgados por la ronda.

Ejemplo (recortado):

```json
{
  "inputs_modelo": {
    "incorrects": 0,
    "hints": 0,
    "step_duration_sec": 12.785186,
    "correct_first_attempt": 1,
    "error_rate": 0,
    "time_efficiency": 0.07254163998947856,
    "student_attempt_count_prev": 0,
    "student_avg_incorrects_prev": 0,
    "student_avg_time_prev": 12.785186,
    "student_accuracy_prev": 1,
    "step_success_rate_prev": 1,
    "step_len": 9,
    "step_num_ops": 2,
    "step_has_parentheses": 0,
    "step_num_digits": 2,
    "step_num_variables": 1,
    "step_abs_constant_sum": 5
  },
  "salida_modelo": {
    "difficulty_level": 1,
    "difficulty_label": "media",
    "probability": 0.9999920611474927
  },
  "accion_adaptativa": "subir",
  "nivel_antes": 1,
  "nivel_despues": 2,
  "puntaje_ronda": 20
}
```

### Como se obtiene `inputs_modelo` desde el dataset/logs

1) **Columnas base (observables post-intento)**  
Se leen directamente del registro del intento (KDD convertido o logs del juego):
- `incorrects`, `hints`, `step_duration_sec`, `correct_first_attempt`  
  - En juego: se calculan en `src/api/main.py` a partir de intentos y tiempo transcurrido.
  - En KDD: se mapean desde columnas como `Incorrects`, `Hints`, `Step Duration (sec)`, `Correct First Attempt` en `src/data_processing/data_loader.py`.

2) **Features derivadas (formulas simples)** (`src/data_processing/feature_engineering.py`)
- `error_rate = incorrects / (incorrects + hints + 1)`
- `time_efficiency = correct_first_attempt / (step_duration_sec + 1)`

3) **Features historicas (solo informacion previa)**  
Se ordena por `event_order` (timestamp si existe; si no, orden del dataset) y se calculan acumulados previos:
- `student_attempt_count_prev`: numero de intentos previos del estudiante (`groupby(student_id).cumcount()`).
- `student_avg_incorrects_prev`, `student_avg_time_prev`, `student_accuracy_prev`: promedios previos por estudiante (sin incluir el intento actual).
- `step_success_rate_prev`: tasa previa de acierto para ese `step_name` (sin incluir el intento actual).

4) **Features estructurales del ejercicio/paso** (derivadas de `step_name`)
- `step_len`: longitud del texto.
- `step_num_ops`: conteo de operadores `+ - * / =`.
- `step_has_parentheses`: 1 si contiene `(` o `)`.
- `step_num_digits`: conteo de digitos.
- `step_num_variables`: conteo de letras (variables).
- `step_abs_constant_sum`: suma de valores absolutos de constantes encontradas en el texto.

5) **Snapshot para auditoria**  
En inferencia, la API guarda las features como `model_inputs` (`src/models/difficulty_model.py`) y ademas las registra en CSV como columnas `model_input_*` (`src/api/main.py`).

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
- Si no, se aplica `group_student_holdout_split` (holdout por `student_id`) cuando es posible.
- Si no, se aplica `temporal_holdout_split` con `event_order` para no mezclar futuro/pasado.
- Como ultimo respaldo, split aleatorio estratificado.

## Modelos entrenados

Se entrenan automaticamente (pipeline del tutor):

- `random_forest`
- `gradient_boosting`
- `logistic_regression`

Artefactos:
- Modelo final: `models/model.pkl`
- Leaderboard: `reports/metrics/model_leaderboard.csv`
- Resumen: `reports/metrics/training_summary.json`

Metricas principales (multiclase) reportadas por modelo:

- `accuracy`
- `precision` (weighted)
- `recall` (weighted)
- `f1_weighted`
- `f1_macro`
- `auc_ovr_weighted`
- `auc_ovr_macro`

El panel administrador web muestra estas metricas y las graficas por modelo.

## Politica adaptativa (post-intento)

El tutor ajusta el nivel **despues** de cada intento usando reglas explicables (ademas del modelo de dificultad).
La idea es que el sistema no se evalue solo por accuracy del clasificador, sino por **mejora observable del estudiante** en los logs.

Implementacion:
- Motor: `src/app/adaptive_engine.py`
- Registro de eventos: `data/raw/gameplay_logs.csv`
- Resumen admin: endpoint `/api/admin/summary` agrega metricas de aprendizaje desde los logs.

### Regla (resumen)

Se calcula un `performance_score` (0..1) usando tiempo, errores y pistas.
Luego se decide `subir/mantener/bajar` con una ventana corta (anti-oscilacion) de los ultimos 3 intentos:
- Bajar si 2/3 sugieren bajar.
- Subir si 2/3 sugieren subir y 0/3 sugieren bajar.
- Si el intento es correcto pero con apoyo (errores/pistas), se recomienda refuerzo sin subir.

## Metricas de aprendizaje (panel admin)

Basadas en `data/raw/gameplay_logs.csv`, el panel admin muestra:
- Accuracy global y por usuario.
- Tiempo/pistas/errores promedio.
- Dependencia de pistas (en intentos correctos).
- Intentos promedio hasta resolver.
- Tasa de oscilacion de nivel (sube y baja rapido).
- Deltas por usuario (2da mitad - 1ra mitad) para ver tendencia: `Δaccuracy`, `Δtiempo`, `Δpistas`, `Δerrores`.

## Donde leer el flujo (guia para entender el codigo)

Entrenamiento del tutor:
- `main_train.py`: orquesta carga → features → entrenamiento → reportes.
- `src/data_processing/data_loader.py`: convierte KDD + logs del juego a un dataframe base.
- `src/data_processing/feature_engineering.py`: crea features historicas/estructurales y `difficulty_level`.
- `src/models/model_trainer.py`: split (incluye holdout por estudiante) + entrenamiento + guardado de modelo/leaderboard.
- `src/evaluation/metrics.py` y `src/evaluation/plots.py`: metricas y graficas.

Tutor (runtime):
- `src/api/main.py`: endpoints del juego, registra `data/raw/gameplay_logs.csv` y expone resumen admin.
- `src/models/difficulty_model.py`: carga `models/model.pkl` y predice dificultad.
- `src/app/adaptive_engine.py`: politica post-intento (subir/mantener/bajar + score + anti-oscilacion).
- `src/app/learning_metrics.py`: calcula metricas de aprendizaje desde los logs para el panel admin.

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
