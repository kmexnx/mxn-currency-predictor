# Instrucciones para ejecutar MXN Currency Predictor

Este documento explica cómo instalar y ejecutar la herramienta de predicción de tipos de cambio basada en el peso mexicano (MXN).

## Requisitos previos

- Python 3.8 o superior
- Pip (gestor de paquetes de Python)
- Git

## Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/kmexnx/mxn-currency-predictor.git
cd mxn-currency-predictor
```

2. Crea un entorno virtual (opcional pero recomendado):

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

### Comandos básicos

Para predecir el tipo de cambio USD/MXN para los próximos 30 días usando el modelo Prophet (configuración por defecto):

```bash
python src/main.py
```

### Opciones disponibles

- `--currency`: Código(s) de moneda para predecir contra MXN. Para varias monedas, use una lista separada por comas (ej. USD,EUR,JPY). Por defecto: USD.
- `--days`: Número de días a predecir. Por defecto: 30.
- `--models`: Modelo(s) a utilizar para la predicción (prophet, arima, lstm, linear, all). Para varios modelos, use una lista separada por comas. Por defecto: prophet.
- `--start-date`: Fecha de inicio para datos históricos (YYYY-MM-DD). Por defecto: hace 2 años.
- `--end-date`: Fecha de fin para datos históricos (YYYY-MM-DD). Por defecto: hoy.
- `--save`: Guardar los resultados y gráficos de la predicción.
- `--output-dir`: Directorio para guardar los archivos de salida. Por defecto: "output".

### Ejemplos de uso

Predecir el tipo de cambio EUR/MXN para los próximos 60 días:

```bash
python src/main.py --currency EUR --days 60
```

Comparar varios modelos para predecir USD/MXN:

```bash
python src/main.py --currency USD --models prophet,arima,lstm
```

Predecir múltiples monedas (USD, EUR, JPY) usando todos los modelos y guardar los resultados:

```bash
python src/main.py --currency USD,EUR,JPY --models all --save
```

## Interpretación de los resultados

La herramienta genera los siguientes resultados:

1. Métricas de evaluación para cada modelo (RMSE, MAE)
2. Gráficos de predicción para cada modelo y moneda
3. Gráficos de comparación entre modelos (cuando se usan múltiples modelos)
4. Un archivo CSV con el resumen de las predicciones (cuando se usa la opción `--save`)

Los resultados se muestran en la consola y, si se especifica la opción `--save`, se guardan en el directorio especificado con `--output-dir`.

## Notas importantes

- Los resultados de predicción son estimaciones basadas en datos históricos y modelos estadísticos/ML.
- Las predicciones a largo plazo (más de 30 días) tienden a ser menos precisas.
- El modelo LSTM requiere más datos históricos y tiempo de procesamiento que otros modelos.
- Para obtener mejores resultados, use múltiples modelos y compare sus predicciones.

## Solución de problemas comunes

Si aparece un error como `ModuleNotFoundError: No module named 'prophet'`, asegúrese de haber instalado todas las dependencias:

```bash
pip install -r requirements.txt
```

Si tiene problemas al instalar Prophet, puede intentar instalarlo por separado:

```bash
pip install prophet
```

En algunos sistemas, puede necesitar instalar dependencias adicionales para Prophet, como Stan. Consulte la documentación oficial de Prophet para más detalles.
