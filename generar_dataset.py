import pandas as pd
import numpy as np

np.random.seed(42)
# Datos generados por clases para generar los 10.200 registros totales
n_por_clase = 3400  

# Dimensiones casi perfectas con textura alta
premium = pd.DataFrame({
    'largo': np.random.normal(50, 0.2, n_por_clase), 
    'ancho': np.random.normal(20, 0.2, n_por_clase),
    'peso': np.random.normal(100, 2, n_por_clase),
    'color_index': np.random.uniform(0.7, 1.0, n_por_clase),
    'textura_index': np.random.uniform(0.85, 1.0, n_por_clase),
    'calidad': 2
})

# Dimensiones con varianza normal y textura media
estandar = pd.DataFrame({
    'largo': np.random.normal(50, 1.5, n_por_clase),
    'ancho': np.random.normal(20, 1.0, n_por_clase),
    'peso': np.random.normal(100, 5, n_por_clase),
    'color_index': np.random.uniform(0.4, 0.9, n_por_clase),
    'textura_index': np.random.uniform(0.4, 0.84, n_por_clase),
    'calidad': 1
})

# Dividimos los defectuosos en 3 grupos para indicarle al modelo los distintos tipos de fallas

# A: Falla por dimensiones
n_A = int(n_por_clase / 3)
defectuoso_dim = pd.DataFrame({
    'largo': np.random.normal(58, 3, n_A),
    'ancho': np.random.normal(25, 2, n_A),
    'peso': np.random.normal(100, 10, n_A),
    'color_index': np.random.uniform(0, 1, n_A),
    'textura_index': np.random.uniform(0, 1, n_A),
    'calidad': 0
})

# B: Falla por textura
n_B = int(n_por_clase / 3)
defectuoso_text = pd.DataFrame({
    'largo': np.random.normal(50, 0.2, n_B),
    'ancho': np.random.normal(20, 0.2, n_B),
    'peso': np.random.normal(100, 2, n_B),
    'color_index': np.random.uniform(0, 1, n_B),
    'textura_index': np.random.uniform(0.0, 0.35, n_B), # Textura PÉSIMA (max 0.35)
    'calidad': 0
})

# C: Falla por color
n_C = n_por_clase - n_A - n_B
defectuoso_color = pd.DataFrame({
    'largo': np.random.normal(50, 1.0, n_C),
    'ancho': np.random.normal(20, 1.0, n_C),
    'peso': np.random.normal(100, 5, n_C),
    'color_index': np.random.uniform(0.0, 0.3, n_C),
    'textura_index': np.random.uniform(0.5, 1.0, n_C),
    'calidad': 0
})

# Acá unimos todos los dataframes
df = pd.concat([premium, estandar, defectuoso_dim, defectuoso_text, defectuoso_color])

# Agregamos pequeñas imperfecciones aleatorias para que no sea tan perfecto y se vuelva mas realista
ruido = np.random.normal(0, 0.02, size=(len(df), 5))
df[['largo', 'ancho', 'peso', 'color_index', 'textura_index']] += ruido

df['color_index'] = df['color_index'].clip(0, 1)
df['textura_index'] = df['textura_index'].clip(0, 1)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('dataset_industrial.csv', index=False)

print(f"Dataset REFORZADO generado exitosamente con {len(df)} registros.")
print("Distribución de Clases:")
print(df['calidad'].value_counts())