import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ruta relativa al archivo
base_path = os.getcwd() # Directorio actual
file_path = Path(os.path.join(base_path, "src", "data", "healthcare_dataset_stroke_data.csv"))

# Carga de datos
df = pd.read_csv(file_path, sep=';')

# Limpieza de la data
df['gender'] = df['gender'].map({'Male': 0.0, 'Female': 1.0, 'Other': 2.0})
df['smoking_status'] = df['smoking_status'].map({'formerly smoked': 0.0, 'never smoked': 1.0, 'smokes': 2.0})
df = df.dropna()

file_path = Path(os.path.join(base_path, "src", "data", 'datos_guardados.csv'))
df[df['stroke'] > 0].to_csv(file_path, index=False)

# Preprocesamiento de datos
# Identificar variables predictores (X) y variable objetivo (y)
X = df.drop(columns=['stroke', 'id'])
y = df['stroke']

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=50)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=50)
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
accurancy = accuracy_score(y_test, y_pred)

print(f'Precisión del modelo: {accurancy:.2f}')

# Realizar predicciones con nuevos datos
new_data = pd.DataFrame([[0.0, 67.0, 0, 1, 228.69, 36.6, 0.0]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status'])
new_data_scaled = scaler.transform(new_data)
pred = model.predict(new_data_scaled)

print(f'Predicción para nuevos datos: {pred[0]}')
