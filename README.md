# Clasificador Inteligente de PQRS

Sistema de Machine Learning para clasificar Peticiones, Quejas, Reclamos y Sugerencias en español. El proyecto combina entrenamiento AutoML, una API REST, una aplicación interactiva en Streamlit y generación de respuestas institucionales sugeridas.

## Objetivo

Las entidades que reciben PQRS deben leer, clasificar, priorizar y responder grandes volúmenes de mensajes ciudadanos. Este sistema automatiza esa primera etapa: identifica la categoría de la PQRS, estima su urgencia, calcula la confianza del modelo y puede sugerir una respuesta institucional.

## Estructura

```text
.
├── api/
│   ├── main.py              # API REST con FastAPI
│   └── schemas.py           # Contratos de entrada y salida
├── data/
│   └── raw/
│       └── pqrs_dataset.csv # Dataset de entrenamiento
├── models/
│   └── best_pipeline.pkl    # Modelo entrenado
├── reports/
│   ├── confusion_matrix.html
│   ├── experiment_summary.csv
│   ├── model_comparison.html
│   └── optuna_history.html
├── src/
│   ├── automl.py            # Búsqueda de modelos con Optuna
│   ├── evaluate.py          # Métricas y reportes
│   ├── events.py            # Eventos institucionales vigentes
│   ├── pqrs_service.py      # Servicio central de clasificación
│   ├── preprocess.py        # Limpieza de texto
│   └── responder.py         # Respuestas sugeridas
├── app.py                   # Demo interactiva en Streamlit
├── generate_data.py         # Generador de datos sintéticos
├── train.py                 # Entrenamiento completo
├── requirements.txt
├── runtime.txt
└── LICENSE
```

## Instalación

```bash
pip install -r requirements.txt
```

## Entrenamiento

```bash
python train.py --trials 20
```

El entrenamiento ejecuta el flujo completo: carga de datos, limpieza, búsqueda AutoML con Optuna, evaluación y guardado del mejor modelo.

## Aplicación Web

```bash
streamlit run app.py
```

La app permite clasificar textos individuales, simular lotes de PQRS y visualizar probabilidades por categoría.

## API REST

```bash
uvicorn api.main:app --reload
```

Endpoints principales:

```text
GET  /health
POST /classify
POST /classify/batch
GET  /events
POST /events
DELETE /events/{id}
```

Para pedir respuesta sugerida:

```text
POST /classify?incluir_respuesta=true
```

## Tecnologías

- Python
- scikit-learn
- Optuna
- MLflow
- FastAPI
- Streamlit
- Plotly
- NLTK

## Próximos cambios

- Sugerir área responsable y ruta de atención.
- Agregar semáforo de riesgo institucional.
- Mejorar evaluación con datos reales anonimizados.
- Añadir pruebas automatizadas para API y servicio central.
