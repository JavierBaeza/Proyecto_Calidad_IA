# AceroIA - Sistema de Control de Calidad Industrial

AceroIA es una solución de Inteligencia Artificial desarrollada para la asignatura de Inteligencia Artificial. El sistema automatiza el control de calidad en líneas de producción simuladas, utilizando algoritmos de Machine Learning (SVM) para clasificar productos basándose en dimensiones físicas y métricas de superficie.

El modelo está diseñado para ser robusto ante fallas sutiles, priorizando la calidad del material como la textura incluso cuando las dimensiones geométricas son perfectas.

## Características Principales

* **Generación de Datos Sintéticos:** Creamos un Script propio que simula 10.200 registros de producción con ruido aleatorio de sensores para un entrenamiento realista.
* **Clasificación Multiclase:**
    * **PREMIUM:** Excelencia absoluta en dimensiones y acabado superficial.
    * **ESTÁNDAR:** Producto funcional con desviaciones menores aceptables.
    * **DEFECTUOSO:** Rechazo por fallas geométricas graves O por mala calidad de textura, el cual es independiente de las medidas.
* **Motor de IA:** Máquina de Soporte Vectorial (SVM) con Kernel RBF y balanceo de pesos (`class_weight='balanced'`).
* **Interfaz Web:** Dashboard moderno desarrollado en **Flask** para la operación en tiempo real.

## Tecnologías

* **Python 3.13**
* **Flask** (Backend Web)
* **Scikit-Learn** (Modelo SVM)
* **Pandas & NumPy** (Procesamiento de Datos)
* **Joblib** (Persistencia del Modelo)
* **HTML5/CSS3** (Frontend Responsive)

## Instalación

1. **Clonar el repositorio**
```text
git clone [https://github.com/JavierBaeza/Proyecto_Calidad_IA]
cd Proyecto_Calidad_IA
```
2. **Crear entorno virtual (Opcional pero recomendado):**
```text
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias:**
```text
pip install -r requirements.txt
```

## Ejecución
* **Debe seguir estrictamente en el mismo orden los pasos presentados a continuación.**

1. **Generar el Datasets**
```text
python generar_dataset.py
```

2. **Entrenar el Modelo**
```text
python entrenar_modelo.py
```

3. **Iniciar el Dashboard**
```text
python app.py
```
* **Abre tu navegador en: http://127.0.0.1:5000**