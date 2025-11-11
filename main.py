from fastapi import FastAPI, Form
import joblib
import pandas as pd

app = FastAPI()

# Cargar el pipeline completo
modelo = joblib.load("modelo_entrenado.pkl")

@app.post("/predecir")
def predecir(
    ciudad: str = Form(...),
    tipo: str = Form(...),
    edad: float = Form(...),
    ingresos: float = Form(...)
):
    # Crear DataFrame con las mismas columnas del entrenamiento
    datos = pd.DataFrame([{
        'ciudad': ciudad,
        'tipo': tipo,
        'edad': edad,
        'ingresos': ingresos
    }])

    # Predecir usando el pipeline (hace el one-hot internamente)
    pred = modelo.predict(datos)
    return {"prediccion": float(pred[0])}