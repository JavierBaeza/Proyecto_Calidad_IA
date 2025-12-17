import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
import os

sns.set_theme(style="white")

if not os.path.exists('dataset_industrial.csv'):
    print("Error: No se encontró 'dataset_industrial.csv'. Ejecuta generar_dataset.py primero.")
    exit()

print("Cargando datos...")
df = pd.read_csv('dataset_industrial.csv')

nombres_clases = {0: 'Defectuoso', 1: 'Estándar', 2: 'Premium'}
df['Nombre_Calidad'] = df['calidad'].map(nombres_clases)

output_dir = 'visualizaciones_informe'
os.makedirs(output_dir, exist_ok=True)
print(f"Las imágenes se guardarán en la carpeta: /{output_dir}/")

print("Generando Mapa de Correlación...")
plt.figure(figsize=(10, 8))

correlation_matrix = df.drop('Nombre_Calidad', axis=1).corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Mapa de Correlación de Variables Industriales', fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/1_mapa_correlacion.png', dpi=300)
print("1_mapa_correlacion.png guardado.")
plt.close()

print("Generando Pairplot (esto puede tardar unos segundos)...")

cols_to_plot = ['largo', 'ancho', 'textura_index', 'color_index']

g = sns.pairplot(df, vars=cols_to_plot, hue='Nombre_Calidad', palette={'Defectuoso': '#e74c3c', 'Estándar': '#f39c12', 'Premium': '#27ae60'},
                 plot_kws={'alpha': 0.6, 's': 30}, height=2.5)

g.fig.suptitle("Mapa de Separación de Clases por Pares de Variables", y=1.02, fontsize=16)
plt.savefig(f'{output_dir}/2_mapa_separacion_pares.png', dpi=300)
print("2_mapa_separacion_pares.png guardado.")
plt.close()

print("Generando Matriz de Confusión del modelo entrenado...")
if not os.path.exists('modelo_svm.pkl'):
     print("Advertencia: No se encontró 'modelo_svm.pkl'. Saltando gráfico 3.")
else:
    modelo = joblib.load('modelo_svm.pkl')
    scaler = joblib.load('escalador.pkl')

    X = df[['largo', 'ancho', 'peso', 'color_index', 'textura_index']]
    y_true = df['calidad']
    X_scaled = scaler.transform(X)

    y_pred = modelo.predict(X_scaled)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    labels = ['Defectuoso', 'Estándar', 'Premium']
    
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    
    plt.title('Matriz de Confusión: Comportamiento Real del SVM', fontsize=16)
    plt.ylabel('Clase Real (Datos)', fontsize=12)
    plt.xlabel('Clase Predicha por el Modelo', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_matriz_confusion_modelo.png', dpi=300)
    print("3_matriz_confusion_modelo.png guardado.")
    plt.close()

print(f"\n¡Proceso completado! Revisa la carpeta '{output_dir}'.")