# Importar librerías
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Ruta relativa al archivo
base_path = os.getcwd() # Directorio actual
file_path = Path(os.path.join(base_path, "src", "data", "healthcare_dataset_stroke_data.csv"))

# Cargar datos
df = pd.read_csv(file_path, sep=';')

# Limpieza de la data
df['gender'] = df['gender'].map({'Male': 0.0, 'Female': 1.0, 'Other': 2.0})
df['smoking_status'] = df['smoking_status'].map({'formerly smoked': 0.0, 'never smoked': 1.0, 'smokes': 2.0})
df = df.dropna()

file_path = Path(os.path.join(base_path, "src", "data", 'datos_guardados.csv'))
df[df['stroke'] > 0].to_csv(file_path, index=False)

# Preprocesamiento
# Estandarizar los datos
scaler = StandardScaler()
scaler_data = scaler.fit_transform(df)

# Elegir un número óptimo de clusteres
loss = []
for i in range(1, 600):
  kmeans = KMeans(n_clusters=i, init="k-means++", random_state=50)
  kmeans.fit(scaler_data)
  loss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 600), loss, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de clusters')
plt.ylabel('Pérdida')
plt.grid(True)
plt.show()
