import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ========== 1. Cargar los datos ==========
def cargar_datos():
    path_energy = "Pagina Made Proyectos\Proyecto Tiempo\CSV\energy_dataset.csv"
    path_weather = "Pagina Made Proyectos\Proyecto Tiempo\CSV\weather_features.csv"

    # Cargar con nombres reales de columnas
    df_energy = pd.read_csv(path_energy, parse_dates=["time"])
    df_weather = pd.read_csv(path_weather, parse_dates=["dt_iso"])

    # Renombrar para unificar
    df_energy = df_energy.rename(columns={"time": "date"})
    df_weather = df_weather.rename(columns={
        "dt_iso": "date",
        "city_name": "city",
        "temp": "temperature",
        "humidity": "humidity",
        "wind_speed": "wind speed",
        "clouds_all": "clouds"
    })

    # Filtrar solo Madrid
    df_weather_madrid = df_weather[df_weather["city"] == "Madrid"]

    # Unir por fecha
    df = pd.merge(df_energy, df_weather_madrid, on="date", how="left")

    return df

# ========== 2. Preprocesamiento ==========
def preparar_datos(df):
    df = df.copy()

    # Seleccionar columnas relevantes
    df = df[["date", "total load actual", "temperature", "humidity", "wind speed", "clouds"]]
    df = df.dropna()

    # Renombrar columnas
    df.columns = ["fecha", "consumo", "temperatura", "humedad", "viento", "nubosidad"]

    # Convertir a datetime UTC y luego eliminar info de zona horaria para evitar errores
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    df["fecha"] = df["fecha"].dt.tz_localize(None)

    # Variables temporales
    df["hora"] = df["fecha"].dt.hour
    df["mes"] = df["fecha"].dt.month

    return df

# ========== 3. Entrenar modelo ==========
def entrenar_modelo(df):
    features = ["temperatura", "humedad", "viento", "nubosidad", "hora", "mes"]
    target = "consumo"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ MSE: {mse:.2f}")
    print(f"✅ R²: {r2:.4f}")

    return modelo, X_test, y_test, y_pred

# ========== 4. Exportar resultados ==========
def exportar_resultados(X_test, y_test, y_pred):
    resultados = X_test.copy()
    resultados["consumo_real"] = y_test.values
    resultados["consumo_predicho"] = y_pred
    resultados.to_csv("predicciones_consumo_espana.csv", index=False)
    print("📁 Resultados exportados a 'predicciones_consumo_espana.csv'")

# ========== 5. Visualización ==========
def graficar(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Consumo Real")
    plt.ylabel("Consumo Predicho")
    plt.title("Consumo Energético - España: Real vs Predicho")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== MAIN ==========
if __name__ == "__main__":
    df_raw = cargar_datos()
    df = preparar_datos(df_raw)
    modelo, X_test, y_test, y_pred = entrenar_modelo(df)
    exportar_resultados(X_test, y_test, y_pred)
    graficar(y_test, y_pred)
# Fin del script
