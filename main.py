from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import requests
import os

app = FastAPI()

# URL del modelo en Google Drive (reemplaza con tu ID)
MODEL_URL = "https://drive.google.com/uc?id=1MfuAswEbVGrkAfH2KvWuOPEGf1FrYqq4"  # Cambia TU_ID_AQUI
MODEL_PATH = "mejor_modelo_optimizado.pkl"

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    print("📥 Descargando modelo desde Google Drive...")
    resp = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(resp.content)
    print("✅ Modelo descargado correctamente")

# Cargar modelo
modelo = joblib.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>Predicción de Asistencia</title></head>
        <body>
            <h2>🎭 Predicción de Asistencia a Eventos</h2>
            <form action="/predecir" method="post">
                <h3>Variables Categóricas</h3>
                <label>Departamento:</label>
                <input type="text" name="Departamento" required><br><br>

                <label>Municipio:</label>
                <input type="text" name="Municipio" required><br><br>

                <label>Día de la Semana:</label>
                <input type="text" name="DiaSemana" required><br><br>

                <label>Categoría de Funciones:</label>
                <input type="text" name="CategoriaFunciones" required><br><br>

                <label>Es Capital (True/False):</label>
                <input type="text" name="EsCapital" required><br><br>

                <label>Artista:</label>
                <input type="text" name="artista" required><br><br>

                <label>Género:</label>
                <input type="text" name="genero" required><br><br>

                <label>Tipo:</label>
                <input type="text" name="tipo" required><br><br>

                <h3>Variables Numéricas</h3>
                <label>Año:</label>
                <input type="number" name="Año" required><br><br>

                <label>Semana del Año (sin):</label>
                <input type="number" step="any" name="SemanaDelAño_sin" required><br><br>

                <label>Semana del Año (cos):</label>
                <input type="number" step="any" name="SemanaDelAño_cos" required><br><br>

                <label>Día del Año (sin):</label>
                <input type="number" step="any" name="DiaDelAño_sin" required><br><br>

                <label>Día del Año (cos):</label>
                <input type="number" step="any" name="DiaDelAño_cos" required><br><br>

                <label>Número de Funciones:</label>
                <input type="number" name="NumeroFunciones" required><br><br>

                <label>Precio Mínimo:</label>
                <input type="number" step="any" name="PrecioMinimo" required><br><br>

                <label>Precio Máximo:</label>
                <input type="number" step="any" name="PrecioMaximo" required><br><br>

                <label>Cantidad de Artistas en el Evento:</label>
                <input type="number" name="cantidad_artistas_evento" required><br><br>

                <button type="submit">🔮 Predecir</button>
            </form>
        </body>
    </html>
    """

@app.post("/predecir")
def predecir(
    # Categóricas
    Departamento: str = Form(...),
    Municipio: str = Form(...),
    DiaSemana: str = Form(...),
    CategoriaFunciones: str = Form(...),
    EsCapital: str = Form(...),
    artista: str = Form(...),
    genero: str = Form(...),
    tipo: str = Form(...),
    # Numéricas
    Año: int = Form(...),
    SemanaDelAño_sin: float = Form(...),
    SemanaDelAño_cos: float = Form(...),
    DiaDelAño_sin: float = Form(...),
    DiaDelAño_cos: float = Form(...),
    NumeroFunciones: int = Form(...),
    PrecioMinimo: float = Form(...),
    PrecioMaximo: float = Form(...),
    cantidad_artistas_evento: int = Form(...)
):
    # Crear DataFrame con las mismas columnas del entrenamiento
    datos = pd.DataFrame([{
        'Departamento': Departamento,
        'Municipio': Municipio,
        'DiaSemana': DiaSemana,
        'CategoriaFunciones': CategoriaFunciones,
        'EsCapital': EsCapital,
        'artista': artista,
        'genero': genero,
        'tipo': tipo,
        'Año': Año,
        'SemanaDelAño_sin': SemanaDelAño_sin,
        'SemanaDelAño_cos': SemanaDelAño_cos,
        'DiaDelAño_sin': DiaDelAño_sin,
        'DiaDelAño_cos': DiaDelAño_cos,
        'NumeroFunciones': NumeroFunciones,
        'PrecioMinimo': PrecioMinimo,
        'PrecioMaximo': PrecioMaximo,
        'cantidad_artistas_evento': cantidad_artistas_evento
    }])

    # Predecir
    pred = modelo.predict(datos)

    return {
        "prediccion": float(pred[0]),
        "mensaje": f"✅ Predicción exitosa: {pred[0]:.2f}"
    }