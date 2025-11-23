from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import requests
import os
import numpy as np
from datetime import datetime, timedelta
import json

app = FastAPI()

MODEL_URL = "https://drive.google.com/uc?id=198OGyKHW7IOrxC79IqaZnvaNPe2ll0Cv"
MODEL_PATH = "mejor_modelo_optimizado_gb.pkl"

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    print("📥 Descargando modelo desde Google Drive...")
    resp = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(resp.content)
    print("✅ Modelo descargado correctamente")

# Cargar modelo
modelo = joblib.load(MODEL_PATH)

# Departamentos y municipios de Colombia
DEPARTAMENTOS_MUNICIPIOS = {
    "Amazonas": ["Leticia", "Puerto Nariño"],
    "Antioquia": ["Medellín", "Bello", "Itagüí", "Envigado", "Apartadó", "Turbo", "Rionegro", "Sabaneta", "Caucasia", "Caldas"],
    "Arauca": ["Arauca", "Arauquita", "Saravena", "Fortul", "Tame"],
    "Atlántico": ["Barranquilla", "Soledad", "Malambo", "Sabanalarga", "Puerto Colombia", "Galapa", "Baranoa"],
    "Bolívar": ["Cartagena", "Magangué", "Turbaco", "Arjona", "El Carmen de Bolívar", "Mompós"],
    "Boyacá": ["Tunja", "Duitama", "Sogamoso", "Chiquinquirá", "Paipa", "Villa de Leyva", "Moniquirá"],
    "Caldas": ["Manizales", "Villamaría", "La Dorada", "Chinchiná", "Riosucio", "Anserma"],
    "Caquetá": ["Florencia", "San Vicente del Caguán", "Puerto Rico", "El Doncello", "Belén de los Andaquíes"],
    "Casanare": ["Yopal", "Aguazul", "Villanueva", "Monterrey", "Tauramena", "Paz de Ariporo"],
    "Cauca": ["Popayán", "Santander de Quilichao", "Puerto Tejada", "Patía", "Corinto", "Miranda"],
    "Cesar": ["Valledupar", "Aguachica", "Bosconia", "Codazzi", "La Paz", "Curumaní"],
    "Chocó": ["Quibdó", "Istmina", "Condoto", "Tadó", "Acandí", "Bahía Solano"],
    "Córdoba": ["Montería", "Cereté", "Lorica", "Sahagún", "Planeta Rica", "Montelíbano"],
    "Cundinamarca": ["Bogotá", "Soacha", "Facatativá", "Zipaquirá", "Chía", "Fusagasugá", "Madrid", "Mosquera", "Girardot", "Cajicá"],
    "Guainía": ["Inírida"],
    "Guaviare": ["San José del Guaviare", "Calamar", "El Retorno"],
    "Huila": ["Neiva", "Pitalito", "Garzón", "La Plata", "Campoalegre", "Gigante"],
    "La Guajira": ["Riohacha", "Maicao", "Uribia", "Manaure", "Fonseca", "San Juan del Cesar"],
    "Magdalena": ["Santa Marta", "Ciénaga", "Fundación", "Plato", "El Banco", "Zona Bananera"],
    "Meta": ["Villavicencio", "Acacías", "Granada", "Puerto López", "San Martín", "Restrepo"],
    "Nariño": ["Pasto", "Tumaco", "Ipiales", "Túquerres", "Samaniego", "La Unión"],
    "Norte de Santander": ["Cúcuta", "Ocaña", "Pamplona", "Villa del Rosario", "Los Patios", "Tibú"],
    "Putumayo": ["Mocoa", "Puerto Asís", "Orito", "Valle del Guamuez", "Villagarzón"],
    "Quindío": ["Armenia", "Calarcá", "La Tebaida", "Montenegro", "Quimbaya", "Circasia"],
    "Risaralda": ["Pereira", "Dosquebradas", "Santa Rosa de Cabal", "La Virginia", "Marsella"],
    "San Andrés y Providencia": ["San Andrés", "Providencia"],
    "Santander": ["Bucaramanga", "Floridablanca", "Girón", "Piedecuesta", "Barrancabermeja", "San Gil", "Socorro"],
    "Sucre": ["Sincelejo", "Corozal", "Sampués", "Tolú", "Majagual"],
    "Tolima": ["Ibagué", "Espinal", "Melgar", "Honda", "Líbano", "Chaparral"],
    "Valle del Cauca": ["Cali", "Palmira", "Buenaventura", "Tuluá", "Cartago", "Buga", "Jamundí", "Yumbo"],
    "Vaupés": ["Mitú"],
    "Vichada": ["Puerto Carreño", "La Primavera", "Cumaribo"]
}

# Capitales de Colombia
CAPITALES = [
    "Leticia", "Medellín", "Arauca", "Barranquilla", "Cartagena", "Tunja", "Manizales",
    "Florencia", "Yopal", "Popayán", "Valledupar", "Quibdó", "Montería", "Bogotá",
    "Inírida", "San José del Guaviare", "Neiva", "Riohacha", "Santa Marta", "Villavicencio",
    "Pasto", "Cúcuta", "Mocoa", "Armenia", "Pereira", "San Andrés", "Bucaramanga",
    "Sincelejo", "Ibagué", "Cali", "Mitú", "Puerto Carreño"
]


def calcular_features_temporales(fecha: str):
    """Calcula las features temporales a partir de una fecha YYYY-MM-DD"""
    dt = datetime.strptime(fecha, "%Y-%m-%d")

    dia_del_anio = dt.timetuple().tm_yday
    semana_del_anio = dt.isocalendar()[1]

    dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dia_semana = dias[dt.weekday()]

    semana_sin = np.sin(2 * np.pi * semana_del_anio / 52)
    semana_cos = np.cos(2 * np.pi * semana_del_anio / 52)
    dia_sin = np.sin(2 * np.pi * dia_del_anio / 365)
    dia_cos = np.cos(2 * np.pi * dia_del_anio / 365)

    dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    dia_semana_es = dias_es[dt.weekday()]

    return {
        "anio": dt.year,
        "dia_semana": dia_semana,
        "dia_semana_es": dia_semana_es,
        "semana_sin": semana_sin,
        "semana_cos": semana_cos,
        "dia_sin": dia_sin,
        "dia_cos": dia_cos,
        "fecha_formateada": dt.strftime("%d/%m/%Y"),
        "fecha_corta": dt.strftime("%d.%m.%Y"),
    }


@app.get("/", response_class=HTMLResponse)
def home():
    opciones_departamentos = ""
    for dept in sorted(DEPARTAMENTOS_MUNICIPIOS.keys()):
        opciones_departamentos += f'<option value="{dept}">{dept}</option>\n'

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EventIA - Sistema de Gestión y Análisis de Eventos</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                font-size: 3em;
                font-weight: 700;
                color: #1a237e;
                margin-bottom: 5px;
            }}
            
            .header p {{
                color: #5e35b1;
                font-size: 1.1em;
            }}
            
            .main-container {{
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 500px 1fr;
                gap: 20px;
            }}
            
            .form-panel {{
                background: white;
                border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                overflow: hidden;
            }}
            
            .form-header {{
                background: linear-gradient(135deg, #5e35b1 0%, #7e57c2 100%);
                color: white;
                padding: 20px 30px;
                font-size: 1.2em;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .form-body {{
                padding: 30px;
                max-height: calc(100vh - 200px);
                overflow-y: auto;
            }}
            
            .form-group {{
                margin-bottom: 20px;
            }}
            
            .form-group label {{
                display: block;
                font-weight: 600;
                color: #424242;
                margin-bottom: 8px;
                font-size: 0.9em;
            }}
            
            .form-group input,
            .form-group select {{
                width: 100%;
                padding: 12px 14px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1em;
                transition: all 0.3s;
                background: white;
            }}
            
            .form-group input:focus,
            .form-group select:focus {{
                outline: none;
                border-color: #5e35b1;
                box-shadow: 0 0 0 3px rgba(94, 53, 177, 0.1);
            }}
            
            .form-row {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }}
            
            .section-title {{
                font-size: 0.85em;
                font-weight: 700;
                color: #5e35b1;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin: 25px 0 15px 0;
                padding-bottom: 8px;
                border-bottom: 2px solid #e8eaf6;
            }}
            
            .submit-btn {{
                width: 100%;
                padding: 16px;
                background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                margin-top: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }}
            
            .submit-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(67, 160, 71, 0.3);
            }}
            
            .results-panel {{
                background: white;
                border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                padding: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                color: #9e9e9e;
            }}
            
            .empty-state {{
                max-width: 300px;
            }}
            
            .empty-state-icon {{
                width: 80px;
                height: 80px;
                background: #f5f5f5;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 20px;
                font-size: 2em;
            }}
            
            @media (max-width: 1200px) {{
                .main-container {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        <script>
            const municipiosPorDepartamento = {json.dumps(DEPARTAMENTOS_MUNICIPIOS)};
            const capitales = {json.dumps(CAPITALES)};
            
            function cargarMunicipios() {{
                const deptSelect = document.getElementById('Departamento');
                const munSelect = document.getElementById('Municipio');
                const esCapitalSelect = document.getElementById('EsCapital');
                
                const departamento = deptSelect.value;
                
                // Limpiar municipios
                munSelect.innerHTML = '<option value="">Seleccionar municipio</option>';
                
                if (departamento && municipiosPorDepartamento[departamento]) {{
                    municipiosPorDepartamento[departamento].forEach(mun => {{
                        const option = document.createElement('option');
                        option.value = mun;
                        option.textContent = mun;
                        munSelect.appendChild(option);
                    }});
                }}
            }}
            
            function verificarCapital() {{
                const munSelect = document.getElementById('Municipio');
                const esCapitalSelect = document.getElementById('EsCapital');
                const municipio = munSelect.value;
                
                if (capitales.includes(municipio)) {{
                    esCapitalSelect.value = '1';
                }} else {{
                    esCapitalSelect.value = '0';
                }}
            }}
        </script>
    </head>
    <body>
        <div class="header">
            <h1>EventIA</h1>
            <p>Sistema de Gestión y Análisis de Eventos</p>
        </div>
        
        <div class="main-container">
            <div class="form-panel">
                <div class="form-header">
                    🎵 Configuración del Evento
                </div>
                <div class="form-body">
                    <form action="/predecir" method="post">
                        <div class="form-group">
                            <label for="artista">Evento</label>
                            <input type="text" id="artista" name="artista" placeholder="Ej: Guns N' Roses en concierto" required>
                        </div>
                        
                        <div class="section-title">📍 Lugar</div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label for="Departamento">Departamento</label>
                                <select id="Departamento" name="Departamento" onchange="cargarMunicipios()" required>
                                    <option value="">Seleccionar departamento</option>
                                    {opciones_departamentos}
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="Municipio">Municipio</label>
                                <select id="Municipio" name="Municipio" onchange="verificarCapital()" required>
                                    <option value="">Seleccionar municipio</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="EsCapital">¿Es Capital?</label>
                            <select id="EsCapital" name="EsCapital" required>
                                <option value="">Seleccione...</option>
                                <option value="1">Sí</option>
                                <option value="0">No</option>
                            </select>
                        </div>
                        
                        <div class="section-title">📅 Fecha</div>
                        
                        <div class="form-group">
                            <label for="fecha">Fecha del Evento</label>
                            <input type="date" id="fecha" name="fecha" required>
                        </div>
                        
                        <div class="section-title">🎤 Artistas</div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label for="artista_nombre">Artista Principal</label>
                                <input type="text" id="artista_nombre" name="artista_nombre" placeholder="Ej: Guns N' Roses" required>
                            </div>
                            <div class="form-group">
                                <label for="cantidad_artistas_evento">Cantidad de Artistas</label>
                                <input type="number" id="cantidad_artistas_evento" name="cantidad_artistas_evento" min="1" value="1" required>
                            </div>
                        </div>
                        
                        <div class="section-title">🏷️ Géneros</div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label for="genero">Género Musical</label>
                                <input type="text" id="genero" name="genero" placeholder="Ej: Rock" required>
                            </div>
                            <div class="form-group">
                                <label for="tipo">Tipo</label>
                                <select id="tipo" name="tipo" required>
                                    <option value="">Seleccione...</option>
                                    <option value="Mixto">Mixto</option>
                                    <option value="Nacional">Nacional</option>
                                    <option value="Internacional">Internacional</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="section-title">🎟️ Secciones y Precios (COP)</div>
                        
                        <div class="form-group">
                            <label for="CategoriaFunciones">Categoría de Funciones</label>
                            <select id="CategoriaFunciones" name="CategoriaFunciones" required>
                                <option value="">Seleccione...</option>
                                <option value="Unica">Única</option>
                                <option value="Pocas">Pocas</option>
                                <option value="Varias">Varias</option>
                                <option value="Muchas">Muchas</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="NumeroFunciones">Número de Funciones</label>
                            <input type="number" id="NumeroFunciones" name="NumeroFunciones" min="1" value="1" required>
                        </div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label for="PrecioMinimo">Precio Mínimo</label>
                                <input type="number" id="PrecioMinimo" name="PrecioMinimo" step="1000" min="0" placeholder="80000" required>
                            </div>
                            <div class="form-group">
                                <label for="PrecioMaximo">Precio Máximo</label>
                                <input type="number" id="PrecioMaximo" name="PrecioMaximo" step="1000" min="0" placeholder="150000" required>
                            </div>
                        </div>
                        
                        <button type="submit" class="submit-btn">
                            ▶ Simular Evento
                        </button>
                    </form>
                </div>
            </div>
            
            <div class="results-panel">
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <p>Complete el formulario para ver el análisis de ventas</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/predecir", response_class=HTMLResponse)
def predecir(
    fecha: str = Form(...),
    Departamento: str = Form(...),
    Municipio: str = Form(...),
    EsCapital: str = Form(...),
    artista: str = Form(...),
    artista_nombre: str = Form(...),
    genero: str = Form(...),
    tipo: str = Form(...),
    CategoriaFunciones: str = Form(...),
    NumeroFunciones: int = Form(...),
    PrecioMinimo: float = Form(...),
    PrecioMaximo: float = Form(...),
    cantidad_artistas_evento: int = Form(...),
):
    fecha_base = datetime.strptime(fecha, "%Y-%m-%d")
    fechas = [fecha_base + timedelta(days=i) for i in range(-2, 3)]

    predicciones = []
    max_pred = 0.0
    min_pred = float("inf")
    fecha_max = ""
    fecha_min = ""

    for fecha_pred in fechas:
        f_str = fecha_pred.strftime("%Y-%m-%d")
        features = calcular_features_temporales(f_str)
        datos = pd.DataFrame(
            [
                {
                    "Departamento": Departamento,
                    "Municipio": Municipio,
                    "DiaSemana": features["dia_semana"],
                    "CategoriaFunciones": CategoriaFunciones,
                    "EsCapital": EsCapital,
                    "artista": artista_nombre,
                    "genero": genero,
                    "tipo": tipo,
                    "Año": features["anio"],
                    "SemanaDelAño_sin": features["semana_sin"],
                    "SemanaDelAño_cos": features["semana_cos"],
                    "DiaDelAño_sin": features["dia_sin"],
                    "DiaDelAño_cos": features["dia_cos"],
                    "NumeroFunciones": NumeroFunciones,
                    "PrecioMinimo": PrecioMinimo,
                    "PrecioMaximo": PrecioMaximo,
                    "cantidad_artistas_evento": cantidad_artistas_evento,
                }
            ]
        )

        pred = modelo.predict(datos)
        pred_val = float(pred[0])

        if pred_val > max_pred:
            max_pred = pred_val
            fecha_max = features["fecha_corta"]
        if pred_val < min_pred:
            min_pred = pred_val
            fecha_min = features["fecha_corta"]

        predicciones.append(
            {
                "fecha": features["fecha_formateada"],
                "fecha_corta": features["fecha_corta"],
                "dia_semana": features["dia_semana_es"],
                "prediccion": pred_val,
                "es_fecha_seleccionada": fecha_pred.date() == fecha_base.date(),
            }
        )

    promedio = sum(p["prediccion"] for p in predicciones) / len(predicciones)
    ocupacion = min(100, int((promedio / 1000) * 100))
    precio_promedio = (PrecioMinimo + PrecioMaximo) / 2
    ingresos_estimados = int(promedio * precio_promedio)

    max_valor = max(p["prediccion"] for p in predicciones) if predicciones else 1
    datos_grafico = ""
    etiquetas_grafico = ""
    for p in predicciones:
        altura = (p["prediccion"] / max_valor) * 100 if max_valor > 0 else 0
        color = "#7e57c2" if p["es_fecha_seleccionada"] else "#b39ddb"
        datos_grafico += f'<div class="bar" style="height: {altura}%; background: {color};" title="{p["prediccion"]:.0f} asistentes"></div>\n'
        etiquetas_grafico += f'<div class="bar-label">{p["fecha_corta"]}</div>\n'

    opciones_departamentos = ""
    for dept in sorted(DEPARTAMENTOS_MUNICIPIOS.keys()):
        selected = 'selected' if dept == Departamento else ''
        opciones_departamentos += f'<option value="{dept}" {selected}>{dept}</option>\n'

    opciones_municipios = ""
    if Departamento in DEPARTAMENTOS_MUNICIPIOS:
        for mun in DEPARTAMENTOS_MUNICIPIOS[Departamento]:
            selected = 'selected' if mun == Municipio else ''
            opciones_municipios += f'<option value="{mun}" {selected}>{mun}</option>\n'

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EventIA - {artista}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                font-size: 3em;
                font-weight: 700;
                color: #1a237e;
                margin-bottom: 5px;
            }}
            
            .header p {{
                color: #5e35b1;
                font-size: 1.1em;
            }}
            
            .main-container {{
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 500px 1fr;
                gap: 20px;
            }}
            
            .form-panel {{
                background: white;
                border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                overflow: hidden;
            }}
            
            .form-header {{
                background: linear-gradient(135deg, #5e35b1 0%, #7e57c2 100%);
                color: white;
                padding: 20px 30px;
                font-size: 1.2em;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .form-body {{
                padding: 30px;
                max-height: calc(100vh - 200px);
                overflow-y: auto;
            }}
            
            .form-group {{
                margin-bottom: 20px;
            }}
            
            .form-group label {{
                display: block;
                font-weight: 600;
                color: #424242;
                margin-bottom: 8px;
                font-size: 0.9em;
            }}
            
            .form-group input,
            .form-group select {{
                width: 100%;
                padding: 12px 14px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1em;
                transition: all 0.3s;
                background: white;
            }}
            
            .form-group input:focus,
            .form-group select:focus {{
                outline: none;
                border-color: #5e35b1;
                box-shadow: 0 0 0 3px rgba(94, 53, 177, 0.1);
            }}
            
            .form-row {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }}
            
            .section-title {{
                font-size: 0.85em;
                font-weight: 700;
                color: #5e35b1;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin: 25px 0 15px 0;
                padding-bottom: 8px;
                border-bottom: 2px solid #e8eaf6;
            }}
            
            .submit-btn {{
                width: 100%;
                padding: 16px;
                background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                margin-top: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }}
            
            .submit-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(67, 160, 71, 0.3);
            }}
            
            .results-panel {{
                background: white;
                border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                overflow: hidden;
            }}
            
            .results-header {{
                background: linear-gradient(135deg, #7e57c2 0%, #9575cd 100%);
                color: white;
                padding: 20px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .results-header h2 {{
                font-size: 1.3em;
                font-weight: 600;
            }}
            
            .results-body {{
                padding: 30px;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 15px;
                margin-bottom: 30px;
            }}
            
            .metric-card {{
                text-align: center;
                padding: 20px 10px;
            }}
            
            .metric-icon {{
                font-size: 2em;
                margin-bottom: 10px;
            }}
            
            .metric-value {{
                font-size: 2em;
                font-weight: 700;
                color: #1a237e;
                line-height: 1;
                margin-bottom: 5px;
            }}
            
            .metric-label {{
                font-size: 0.75em;
                color: #757575;
                font-weight: 500;
            }}
            
            .metric-sublabel {{
                font-size: 0.7em;
                color: #9e9e9e;
            }}
            
            .chart-section {{
                background: #fafafa;
                border-radius: 12px;
                padding: 25px;
            }}
            
            .chart-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}
            
            .chart-title {{
                font-size: 1.1em;
                font-weight: 600;
                color: #424242;
            }}
            
            .chart-dates {{
                display: flex;
                gap: 15px;
                font-size: 0.85em;
            }}
            
            .date-badge {{
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 500;
            }}
            
            .date-badge.max {{
                background: #e8f5e9;
                color: #2e7d32;
            }}
            
            .date-badge.min {{
                background: #ffebee;
                color: #c62828;
            }}
            
            .chart-container {{
                height: 250px;
                display: flex;
                align-items: flex-end;
                justify-content: space-around;
                gap: 10px;
                margin-bottom: 15px;
                padding: 20px;
                background: white;
                border-radius: 8px;
            }}
            
            .bar {{
                flex: 1;
                max-width: 80px;
                border-radius: 8px 8px 0 0;
                transition: all 0.3s;
                cursor: pointer;
            }}
            
            .bar:hover {{
                opacity: 0.8;
                transform: translateY(-5px);
            }}
            
            .chart-labels {{
                display: flex;
                justify-content: space-around;
                gap: 10px;
                padding: 0 20px;
            }}
            
            .bar-label {{
                flex: 1;
                max-width: 80px;
                text-align: center;
                font-size: 0.8em;
                color: #757575;
                font-weight: 500;
            }}
            
            @media (max-width: 1200px) {{
                .main-container {{
                    grid-template-columns: 1fr;
                }}
                
                .metrics-grid {{
                    grid-template-columns: repeat(3, 1fr);
                }}
            }}
            
            @media (max-width: 768px) {{
                .metrics-grid {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}
        </style>
        <script>
            const municipiosPorDepartamento = {json.dumps(DEPARTAMENTOS_MUNICIPIOS)};
            const capitales = {json.dumps(CAPITALES)};
            
            function cargarMunicipios() {{
                const deptSelect = document.getElementById('Departamento');
                const munSelect = document.getElementById('Municipio');
                const departamento = deptSelect.value;
                
                munSelect.innerHTML = '<option value="">Seleccionar municipio</option>';
                
                if (departamento && municipiosPorDepartamento[departamento]) {{
                    municipiosPorDepartamento[departamento].forEach(mun => {{
                        const option = document.createElement('option');
                        option.value = mun;
                        option.textContent = mun;
                        munSelect.appendChild(option);
                    }});
                }}
            }}
            
            function verificarCapital() {{
                const munSelect = document.getElementById('Municipio');
                const esCapitalSelect = document.getElementById('EsCapital');
                const municipio = munSelect.value;
                
                if (capitales.includes(municipio)) {{
                    esCapitalSelect.value = '1';
                }} else {{
                    esCapitalSelect.value = '0';
                }}
            }}
            
            window.onload = function() {{
                cargarMunicipios();
                document.getElementById('Municipio').value = '{Municipio}';
            }};
        </script>
    </head>
    <body>
        <div class="header">
            <h1>EventIA</h1>
            <p>Sistema de Gestión y Análisis de Eventos</p>
        </div>
        
        <div class="main-container">
            <div class="form-panel">
                <div class="form-header">
                    🎵 Configuración del Evento
                </div>
                <div class="form-body">
                    <form action="/predecir" method="post">
                        <div class="form-group">
                            <label for="artista">Evento</label>
                            <input type="text" id="artista" name="artista" value="{artista}" required>
                        </div>
                        
                        <div class="section-title">📍 Lugar</div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label for="Departamento">Departamento</label>
                                <select id="Departamento" name="Departamento" onchange="cargarMunicipios()" required>
                                    <option value="">Seleccionar departamento</option>
                                    {opciones_departamentos}
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="Municipio">Municipio</label>
                                <select id="Municipio" name="Municipio" onchange="verificarCapital()" required>
                                    <option value="">Seleccionar municipio</option>
                                    {opciones_municipios}
                                </select>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="EsCapital">¿Es Capital?</label>
                            <select id="EsCapital" name="EsCapital" required>
                                <option value="">Seleccione...</option>
                                <option value="1" {"selected" if EsCapital == "1" else ""}>Sí</option>
                                <option value="0" {"selected" if EsCapital == "0" else ""}>No</option>
                            </select>
                        </div>
                        
                        <div class="section-title">📅 Fecha</div>
                        
                        <div class="form-group">
                            <label for="fecha">Fecha del Evento</label>
                            <input type="date" id="fecha" name="fecha" value="{fecha}" required>
                        </div>
                        
                        <div class="section-title">🎤 Artistas</div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label for="artista_nombre">Artista Principal</label>
                                <input type="text" id="artista_nombre" name="artista_nombre" value="{artista_nombre}" required>
                            </div>
                            <div class="form-group">
                                <label for="cantidad_artistas_evento">Cantidad de Artistas</label>
                                <input type="number" id="cantidad_artistas_evento" name="cantidad_artistas_evento" min="1" value="{cantidad_artistas_evento}" required>
                            </div>
                        </div>
                        
                        <div class="section-title">🏷️ Géneros</div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label for="genero">Género Musical</label>
                                <input type="text" id="genero" name="genero" value="{genero}" required>
                            </div>
                            <div class="form-group">
                                <label for="tipo">Tipo</label>
                                <select id="tipo" name="tipo" required>
                                    <option value="">Seleccione...</option>
                                    <option value="Mixto" {"selected" if tipo == "Mixto" else ""}>Mixto</option>
                                    <option value="Nacional" {"selected" if tipo == "Nacional" else ""}>Nacional</option>
                                    <option value="Internacional" {"selected" if tipo == "Internacional" else ""}>Internacional</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="section-title">🎟️ Secciones y Precios (COP)</div>
                        
                        <div class="form-group">
                            <label for="CategoriaFunciones">Categoría de Funciones</label>
                            <select id="CategoriaFunciones" name="CategoriaFunciones" required>
                                <option value="">Seleccione...</option>
                                <option value="Unica" {"selected" if CategoriaFunciones == "Unica" else ""}>Única</option>
                                <option value="Pocas" {"selected" if CategoriaFunciones == "Pocas" else ""}>Pocas</option>
                                <option value="Varias" {"selected" if CategoriaFunciones == "Varias" else ""}>Varias</option>
                                <option value="Muchas" {"selected" if CategoriaFunciones == "Muchas" else ""}>Muchas</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="NumeroFunciones">Número de Funciones</label>
                            <input type="number" id="NumeroFunciones" name="NumeroFunciones" min="1" value="{NumeroFunciones}" required>
                        </div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label for="PrecioMinimo">Precio Mínimo</label>
                                <input type="number" id="PrecioMinimo" name="PrecioMinimo" step="1000" min="0" value="{int(PrecioMinimo)}" required>
                            </div>
                            <div class="form-group">
                                <label for="PrecioMaximo">Precio Máximo</label>
                                <input type="number" id="PrecioMaximo" name="PrecioMaximo" step="1000" min="0" value="{int(PrecioMaximo)}" required>
                            </div>
                        </div>
                        
                        <button type="submit" class="submit-btn">
                            ▶ Simular Evento
                        </button>
                    </form>
                </div>
            </div>
            
            <div class="results-panel">
                <div class="results-header">
                    <h2>{artista}</h2>
                    <div style="display: flex; gap: 10px;">
                        <button style="padding: 8px 16px; background: rgba(255,255,255,0.2); border: 1px solid white; border-radius: 6px; color: white; cursor: pointer; font-weight: 500;">Simulación IA</button>
                    </div>
                </div>
                <div class="results-body">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-icon">👥</div>
                            <div class="metric-value">{int(promedio)}</div>
                            <div class="metric-label">Asistentes</div>
                            <div class="metric-sublabel">Proyectados</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-icon">📈</div>
                            <div class="metric-value">{int(max_pred)}</div>
                            <div class="metric-label">Mejor Día</div>
                            <div class="metric-sublabel">{fecha_max}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-icon">📉</div>
                            <div class="metric-value">{int(min_pred)}</div>
                            <div class="metric-label">Menor Día</div>
                            <div class="metric-sublabel">{fecha_min}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-icon">💜</div>
                            <div class="metric-value">{ocupacion}%</div>
                            <div class="metric-label">Ocupación</div>
                            <div class="metric-sublabel">Estimada</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-icon">💵</div>
                            <div class="metric-value">${int(ingresos_estimados/1000000)}M</div>
                            <div class="metric-label">Ingresos</div>
                            <div class="metric-sublabel">(COP)</div>
                        </div>
                    </div>
                    
                    <div class="chart-section">
                        <div class="chart-header">
                            <div class="chart-title">📊 Análisis de Ventas y Factores Externos</div>
                            <div class="chart-dates">
                                <div class="date-badge max">Pico: {fecha_max}</div>
                                <div class="date-badge min">Mínimo: {fecha_min}</div>
                            </div>
                        </div>
                        <div class="chart-container">
                            {datos_grafico}
                        </div>
                        <div class="chart-labels">
                            {etiquetas_grafico}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """