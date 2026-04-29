# NYC Taxi Fare Prediction

## Resumen del proyecto

Este repositorio desarrolla un flujo completo de machine learning para predecir `fare_amount` en viajes de NYC Yellow Taxi de 2019. El proyecto parte de archivos mensuales crudos, construye una tabla maestra reproducible, limpia los registros, crea variables seguras para modelaje, compara modelos de regresion, ajusta los mejores candidatos y termina con un notebook de interpretacion para explicar que esta aprendiendo el modelo.

La idea central es estimar la tarifa usando informacion disponible antes o cerca del momento de recogida, evitando variables que solo se conocen al final del viaje o despues del pago. Por eso el proyecto no usa variables monetarias posteriores como `total_amount`, `tip_amount`, `tolls_amount` o cargos/surcharges finales como predictores.

## Preguntas que responde

1. Si es posible predecir la tarifa de taxi con buena precision usando distancia, calendario, hora y geografia.
2. Que decisiones de limpieza son necesarias para que el target `fare_amount` sea confiable.
3. Si modelos no lineales, como Random Forest y Gradient Boosting, mejoran sobre una linea base simple y una regresion Ridge.
4. Cual modelo funciona mejor cuando se evalua cronologicamente en meses futuros.
5. Que variables explican la mayor parte de las predicciones y en que segmentos el modelo todavia falla mas.

## Estructura del repositorio

```text
data/
  raw/                       Archivos originales mensuales de NYC Yellow Taxi 2019
  procesed/                  Datasets intermedios y finales en parquet

notebooks/
  01_master_table_creation.ipynb
  02_cleaninig.ipynb
  03_model_comparison.ipynb
  04_hyperparameter_tuning_and_evaluation.ipynb
  05_model_interpretation_and_reporting.ipynb

reports/
  models/                    Modelos entrenados locales; no se versionan en Git
```

Nota de nombres: el repositorio mantiene `data/procesed/` y `02_cleaninig.ipynb` tal como existen actualmente para no romper rutas ni dependencias entre notebooks.

## Datos y alcance

- Fuente: NYC Yellow Taxi trip data, ano 2019.
- Granularidad: un registro por viaje.
- Target: `fare_amount`.
- Tipo de problema: regresion supervisada.
- Evaluacion principal: particion cronologica, no aleatoria.
- Entrenamiento final: enero a octubre de 2019.
- Holdout final: noviembre y diciembre de 2019.

El proyecto crea primero una muestra balanceada de 1,000,000 viajes por mes. Esto produce una tabla maestra de 12,000,000 filas antes de limpieza. Luego, despues de filtros de calidad, el dataset final de modelaje queda con 11,792,502 filas y 28 columnas.

## Documentos del proyecto, uno a uno

### 1. `01_master_table_creation.ipynb`: creacion de tabla maestra

Este notebook construye la base analitica del proyecto a partir de los 12 archivos mensuales de 2019.

Que hace:

- Localiza archivos con patron `yellow_tripdata_2019-*.csv`.
- Valida que existan archivos fuente antes de continuar.
- Calcula la union de columnas observadas en los meses para manejar diferencias de esquema.
- Lee cada mes, lo muestrea a 1,000,000 filas cuando aplica y alinea las columnas.
- Agrega metadatos de trazabilidad como `year`, `month` y `source_file`.
- Concatena los 12 meses en una sola tabla maestra.
- Guarda `data/procesed/master_2019_1M_per_month.parquet`.

Decisiones tomadas:

- Usar todos los meses de 2019 para capturar estacionalidad.
- Mantener una muestra igual por mes para evitar que meses con mas volumen dominen el entrenamiento.
- Usar una semilla fija para reproducibilidad.
- Guardar en parquet para acelerar notebooks posteriores y evitar repetir ingestion cruda.
- No limpiar ni modelar todavia: este documento solo crea una base estable y auditable.

Resultado clave:

```text
MASTER SHAPE: (12000000, 21)
```

### 2. `02_cleaninig.ipynb`: limpieza y feature engineering

Este notebook transforma la tabla maestra en un dataset listo para machine learning. Es el paso donde se toman las decisiones mas importantes de calidad de datos y control de leakage.

Que hace:

- Carga la tabla maestra y el lookup oficial de zonas TLC.
- Audita valores nulos antes de modificar datos.
- Elimina registros estructuralmente incompletos.
- Convierte timestamps de pickup y dropoff para validacion.
- Calcula `duration_min` y `speed_mph` solo como variables auxiliares de limpieza.
- Filtra viajes imposibles o implausibles.
- Convierte pasajeros imposibles en missing values y crea `passenger_count_missing`.
- Crea variables de calendario y hora.
- Crea transformaciones de distancia.
- Une informacion geografica de pickup y dropoff con el taxi zone lookup.
- Elimina columnas que podrian producir leakage.
- Exporta `data/procesed/taxi_2019_modeling_ready.parquet`.

Filtros principales:

- `fare_amount > 0`.
- `trip_distance > 0`.
- `duration_min > 0`.
- `duration_min < 360` minutos.
- `speed_mph > 0`.
- `speed_mph < 120`.

Decisiones de leakage:

- Se eliminan variables monetarias posteriores al viaje: `total_amount`, `tip_amount`, `tolls_amount`, `extra`, `mta_tax`, `improvement_surcharge`, `congestion_surcharge` y `airport_fee`.
- Se eliminan campos de pago o comportamiento posterior como `payment_type`.
- Se eliminan timestamps crudos y variables derivadas de duracion real del viaje.
- `pickup_month_num` se conserva solo para hacer splits cronologicos y analisis por mes; no se usa como predictor directo.

Features finales:

- Target y split: `fare_amount`, `pickup_month_num`.
- Distancia: `trip_distance`, `log_trip_distance`, `trip_distance_sq`.
- Interacciones: `distance_x_rush`, `distance_x_weekend`.
- Pasajeros: `passenger_count`, `passenger_count_missing`.
- Calendario y hora: `pickup_weekday`, `is_weekend`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `is_rush_hour`, `is_night`, `is_peak_daytime`.
- Geografia: `PULocationID`, `DOLocationID`, `pickup_borough`, `dropoff_borough`, `pickup_service_zone`, `dropoff_service_zone`.
- Indicadores espaciales: `is_airport_pickup`, `is_airport_dropoff`, `same_borough_trip`, `manhattan_trip`.

Resultados clave:

```text
rows kept after trip validation: 11792502
share kept: 0.9857
final model shape: (11792502, 28)
```

### 3. `03_model_comparison.ipynb`: comparacion inicial de modelos

Este notebook establece el primer benchmark de modelos usando el dataset limpio. La meta no es optimizar todavia, sino comparar familias de modelos bajo las mismas reglas.

Que hace:

- Carga `taxi_2019_modeling_ready.parquet`.
- Divide el dataset cronologicamente.
- Usa enero a octubre para entrenamiento y noviembre a diciembre para prueba.
- Toma muestras fijas para acelerar la comparacion inicial.
- Define 22 columnas numericas y 4 categoricas.
- Construye pipelines separados para modelos lineales y modelos de arboles.
- Evalua todos los modelos con MAE, RMSE y R2.

Modelos comparados:

- `DummyRegressor` con estrategia de mediana.
- `Ridge` como modelo lineal regularizado.
- `RandomForestRegressor`.
- `GradientBoostingRegressor`.

Decisiones de preprocesamiento:

- Para Ridge: imputacion numerica por mediana, escalado con `StandardScaler`, imputacion categorica por moda y one-hot encoding.
- Para modelos de arboles: imputacion numerica por mediana, imputacion categorica por moda y one-hot encoding sin escalado.
- Usar el mismo target, split y metricas para que la comparacion sea justa.

Resultados de la comparacion inicial:

```text
model              MAE     RMSE      R2
random_forest      1.563   3.736   0.897
gradient_boosting  1.576   3.861   0.890
ridge              1.831   4.027   0.880
dummy_median       6.554  12.212  -0.099
```

Conclusion de este documento:

- Los modelos no lineales superan a Ridge y al baseline simple.
- Random Forest logra el mejor resultado inicial, aunque Gradient Boosting queda muy cerca.
- Tiene sentido pasar a tuning con Random Forest y Gradient Boosting, no con todos los modelos.

### 4. `04_hyperparameter_tuning_and_evaluation.ipynb`: tuning y evaluacion final

Este notebook ajusta los dos modelos mas fuertes de la comparacion inicial y los evalua de forma mas profunda en el holdout cronologico.

Que hace:

- Reutiliza el dataset limpio y el split enero-octubre vs noviembre-diciembre.
- Toma una muestra de tuning de 250,000 filas balanceada por mes.
- Usa `GroupKFold(n_splits=5)` agrupando por `pickup_month_num`.
- Ejecuta `RandomizedSearchCV` con 16 combinaciones por modelo.
- Optimiza con `neg_root_mean_squared_error`.
- Reentrena cada mejor pipeline sobre todo el entrenamiento disponible.
- Evalua en el holdout completo de noviembre y diciembre.
- Analiza errores por mes, por banda de tarifa y casos extremos.

Espacios de busqueda principales:

- Gradient Boosting: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample`, `max_features`.
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`.

Metricas finales en holdout:

```text
model              MAE     RMSE     R2      median_abs_error  p90_abs_error  within_$1  within_$2  within_$5
random_forest      1.520   5.516   0.799   0.863             3.197          0.556      0.794      0.957
gradient_boosting  1.576   5.546   0.797   0.931             3.230          0.528      0.786      0.957
```

Decision de modelaje:

- Random Forest es el mejor modelo final por MAE, RMSE, R2, error mediano y porcentaje de predicciones dentro de 1 y 2 dolares.
- Gradient Boosting queda muy cerca y sigue siendo util, especialmente porque su artefacto guardado es mucho mas liviano y facil de cargar para interpretacion.
- El holdout muestra que diciembre es mas dificil que noviembre para ambos modelos, con RMSE cercano a 7.05 en diciembre frente a alrededor de 3.33-3.40 en noviembre.
- Los peores errores se concentran en tarifas extremadamente altas o casos raros, lo que eleva el RMSE aunque el error mediano sea bajo.

### 5. `05_model_interpretation_and_reporting.ipynb`: interpretacion y reporte

Este notebook convierte los modelos entrenados en explicaciones mas comunicables. Su objetivo es cerrar el proyecto con interpretabilidad y analisis de segmentos, no solo con metricas globales.

Que hace:

- Carga el dataset limpio y revisa modelos guardados en `reports/models`.
- Detecta que `gradient_boosting_tuned.joblib` es liviano y cargable.
- Detecta que `random_forest_tuned.joblib` existe pero pesa aproximadamente 20.393 GB, por lo que no se carga en un notebook normal.
- Evalua el modelo cargable en el holdout.
- Selecciona el mejor modelo cargable para interpretacion.
- Calcula importancias internas del modelo.
- Agrupa importancias one-hot a nivel de feature original.
- Calcula permutation importance sobre una muestra de holdout.
- Genera partial dependence plots para drivers numericos.
- Resume errores por borough y banda de tarifa.

Decision practica:

- Aunque Random Forest es el mejor modelo por metricas finales, Gradient Boosting se usa como modelo interpretado porque su archivo es pequeno y manejable.
- Esta decision separa dos criterios: mejor performance final vs facilidad de interpretacion/reporte.

Resultados del modelo interpretado:

```text
model              MAE     RMSE     R2      median_abs_error  p90_abs_error  within_$2  within_$5
gradient_boosting  1.576   5.546   0.797   0.931             3.230          0.786      0.957
```

Drivers principales por importancia interna:

```text
log_trip_distance  0.405
trip_distance      0.374
trip_distance_sq   0.178
DOLocationID       0.021
hour_cos           0.004
```

Drivers principales por permutation importance:

```text
log_trip_distance     3.573
trip_distance         2.782
trip_distance_sq      1.491
DOLocationID          0.442
pickup_borough        0.223
PULocationID          0.212
hour_cos              0.160
is_airport_dropoff    0.146
```

Lectura del resultado:

- La distancia domina claramente la prediccion de tarifa.
- La geografia tambien importa, especialmente destino, origen, boroughs y aeropuertos.
- Las variables temporales ayudan, pero pesan menos que distancia y ubicacion.
- El error aumenta en segmentos de tarifa mas alta.
- Manhattan tiene el mayor volumen y un MAE bajo, mientras que EWR y Staten Island muestran errores mas altos por menor volumen y viajes mas caros/particulares.

## Decisiones de modelaje explicadas

### 1. Split cronologico

El proyecto entrena con enero-octubre y evalua con noviembre-diciembre. Esta decision evita una evaluacion demasiado optimista que podria ocurrir con un split aleatorio, porque en produccion normalmente se predicen viajes futuros, no registros mezclados del mismo periodo.

### 2. Control de target leakage

El modelo solo debe usar informacion disponible al momento de estimar la tarifa. Por eso se eliminan variables que se conocen despues del viaje, incluyendo montos finales, propinas, peajes y tipo de pago. Tambien se evita usar duracion real como predictor, porque la duracion completa solo se conoce al terminar el viaje.

### 3. Limpieza agresiva de viajes imposibles

La tarifa es sensible a registros corruptos. Un viaje con distancia cero, duracion negativa o velocidad imposible puede distorsionar mucho la funcion aprendida. Por eso se filtran viajes invalidos antes de modelar.

### 4. Distancia como feature principal

Se conserva `trip_distance` y se agregan `log_trip_distance` y `trip_distance_sq`. Esto permite capturar relaciones no lineales: tarifas cortas, medias y largas no necesariamente crecen igual en dolares por milla.

### 5. Variables ciclicas para hora y mes

En vez de usar hora cruda de forma lineal, el proyecto usa `hour_sin`, `hour_cos`, `month_sin` y `month_cos`. Esto representa correctamente que las 23:00 y las 00:00 estan cerca en el ciclo diario.

### 6. Geografia interpretable

El proyecto usa IDs de zona y tambien borough/service zone. Los IDs conservan detalle espacial, mientras que borough y service zone permiten explicar resultados de forma mas comprensible.

### 7. Baseline antes de modelos complejos

Se incluye un modelo dummy por mediana para saber cuanto valor real aporta el machine learning. Tambien se incluye Ridge para tener un punto de comparacion lineal antes de justificar modelos mas costosos.

### 8. Tuning solo para candidatos fuertes

Despues del benchmark inicial, solo Random Forest y Gradient Boosting pasan a tuning. Esto evita gastar computo en modelos que ya mostraron menor potencial.

### 9. Cross-validation agrupada por mes

El tuning usa `GroupKFold` por `pickup_month_num`. Asi se reduce el riesgo de validar con particiones que mezclan demasiado patrones temporales dentro de los mismos meses.

### 10. Performance final vs interpretabilidad

Random Forest gana en metricas finales, pero su artefacto local es muy grande para un flujo de interpretacion comodo. Gradient Boosting queda muy cerca en performance y es mucho mas facil de cargar, por lo que se usa para el notebook de reporte interpretativo.

## Resultado final del proyecto

El mejor modelo por performance final es Random Forest:

```text
MAE: 1.5198
RMSE: 5.5158
R2: 0.7991
Median absolute error: 0.8628
Predicciones dentro de $2: 79.42%
Predicciones dentro de $5: 95.71%
```

El modelo interpretado principal es Gradient Boosting:

```text
MAE: 1.5757
RMSE: 5.5457
R2: 0.7969
Median absolute error: 0.9308
Predicciones dentro de $2: 78.56%
Predicciones dentro de $5: 95.71%
```

La lectura general es que ambos modelos predicen bastante bien la mayoria de viajes normales. El error mediano esta por debajo de 1 dolar, pero el RMSE sube por outliers y tarifas extremadamente altas.

## Como leer el proyecto

Orden recomendado:

1. Leer este README para entender la historia completa.
2. Abrir `notebooks/01_master_table_creation.ipynb` para ver como se construye la tabla maestra.
3. Abrir `notebooks/02_cleaninig.ipynb` para revisar limpieza, leakage y features.
4. Abrir `notebooks/03_model_comparison.ipynb` para ver el benchmark inicial.
5. Abrir `notebooks/04_hyperparameter_tuning_and_evaluation.ipynb` para tuning, evaluacion final y diagnosticos.
6. Abrir `notebooks/05_model_interpretation_and_reporting.ipynb` para interpretacion, importancia de variables y lectura de segmentos.

## Como ejecutar

Ejecutar los notebooks en orden, porque cada etapa depende de archivos producidos por la anterior:

1. `notebooks/01_master_table_creation.ipynb`
2. `notebooks/02_cleaninig.ipynb`
3. `notebooks/03_model_comparison.ipynb`
4. `notebooks/04_hyperparameter_tuning_and_evaluation.ipynb`
5. `notebooks/05_model_interpretation_and_reporting.ipynb`

Los parquet intermedios permiten que los notebooks posteriores no tengan que repetir la carga completa de CSV crudos.

## Entregables

Entregables de datos:

- `data/procesed/master_2019_1M_per_month.parquet`
- `data/procesed/taxi_2019_modeling_ready.parquet`

Entregables analiticos:

- Benchmark inicial de modelos.
- Tuning de Random Forest y Gradient Boosting.
- Evaluacion final en holdout cronologico.
- Diagnosticos por mes y por banda de tarifa.
- Interpretacion con feature importance, permutation importance y partial dependence.

Entregables de comunicacion:

- Notebooks escritos como reporte guiado.
- README actualizado como resumen ejecutivo y tecnico del proyecto.

## Limitaciones y proximos pasos

Limitaciones actuales:

- La muestra esta balanceada a 1,000,000 viajes por mes, no es todo el universo crudo completo.
- `trip_distance` se usa como predictor; esto es razonable si la distancia estimada esta disponible al momento de cotizar, pero debe aclararse para casos de uso en tiempo real.
- Los outliers de tarifas extremadamente altas todavia generan errores grandes.
- El artefacto de Random Forest es muy pesado para interpretacion ligera.

Posibles extensiones:

- Entrenar una version mas compacta del mejor modelo para deployment.
- Probar modelos gradient boosting modernos como HistGradientBoosting, XGBoost, LightGBM o CatBoost.
- Agregar validacion por airport trips, borough pairs y franjas horarias.
- Tratar tarifas extremas con winsorization, modelos robustos o evaluacion separada de outliers.
- Crear un script reproducible de entrenamiento fuera de notebooks.
- Agregar `requirements.txt` o ambiente conda para reproducibilidad completa.
