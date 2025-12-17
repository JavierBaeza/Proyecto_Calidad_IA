from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Implementamos una configuración para recargar templates automáticamente
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Cargamos artefactos
if os.path.exists('modelo_svm.pkl'):
    modelo = joblib.load('modelo_svm.pkl')
    scaler = joblib.load('escalador.pkl')
    metricas = joblib.load('metricas.pkl')
else:
    print("Error: No se encontraron los archivos .pkl. Ejecuta entrenar_modelo.py primero.")

# Ingresamos etiquetas
clases = {0: "DEFECTUOSO", 1: "ESTÁNDAR", 2: "PREMIUM"}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion_texto = ""
    probabilidad = ""
    input_data = {}
    clase_css = ""
    acc_global = f"{metricas['accuracy']*100:.1f}%" if 'metricas' in globals() else "N/A"

    if request.method == 'POST':
        try:
            # Obtenemos los datos 
            largo = float(request.form['largo'])
            ancho = float(request.form['ancho'])
            peso = float(request.form['peso'])
            color = float(request.form['color'])
            textura = float(request.form['textura'])
            
            input_data = {'largo': largo, 'ancho': ancho, 'peso': peso, 'color': color, 'textura': textura}

            # Preprocesamos
            features = np.array([[largo, ancho, peso, color, textura]])
            features_scaled = scaler.transform(features)

            # Predecimos
            pred = modelo.predict(features_scaled)[0]
            probs = modelo.predict_proba(features_scaled)[0]
            
            prediccion_texto = clases[pred]
            probabilidad = f"{probs[pred]*100:.2f}% de confianza"
            
            if pred == 0: clase_css = "danger"
            elif pred == 1: clase_css = "warning"
            else: clase_css = "success"

        except Exception as e:
            prediccion_texto = f"Error: {e}"

    return render_template('index.html', 
                           prediccion=prediccion_texto, 
                           probabilidad=probabilidad,
                           datos=input_data,
                           css_class=clase_css,
                           acc=acc_global)

if __name__ == '__main__':
    app.run(debug=True)