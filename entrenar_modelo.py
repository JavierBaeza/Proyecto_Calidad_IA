import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Cargamos dataset
print("Cargando dataset ' dataset_industrial.csv'...")
df = pd.read_csv('dataset_industrial.csv')

# Separamos características y etiquetas
X = df[['largo', 'ancho', 'peso', 'color_index', 'textura_index']]
y = df['calidad']

# Dividimos datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamos datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamos el modelo SVM
print("Entrenando modelo SVM (esto puede tardar unos segundos)...")
modelo = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)
modelo.fit(X_train_scaled, y_train)

# Evaluamos el modelo
y_pred = modelo.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"Entrenamiento completado.")
print(f"Precisión del modelo: {acc*100:.2f}%")

# En este punto, guardamos los artefactos
joblib.dump(modelo, 'modelo_svm.pkl')
joblib.dump(scaler, 'escalador.pkl')
joblib.dump(report, 'metricas.pkl')
print("Archivos .pkl guardados correctamente.")