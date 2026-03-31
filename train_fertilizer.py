import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('FertilizerPrediction.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Encode categorical features
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

df['Soil Type Encoded'] = soil_encoder.fit_transform(df['Soil Type'])
df['Crop Type Encoded'] = crop_encoder.fit_transform(df['Crop Type'])
df['Fertilizer Encoded'] = fertilizer_encoder.fit_transform(df['Fertilizer Name'])

# Features: Temperature, Humidity, Moisture, Soil Type, Crop Type, N, K, P
feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type Encoded', 'Crop Type Encoded', 'Nitrogen', 'Potassium', 'Phosphorous']
X = df[feature_cols]
y = df['Fertilizer Encoded']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=range(len(fertilizer_encoder.classes_)), target_names=fertilizer_encoder.classes_, zero_division=0))

# Save model and encoders
with open('fertilizer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('fertilizer_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('fertilizer_encoders.pkl', 'wb') as f:
    pickle.dump({
        'soil': soil_encoder,
        'crop': crop_encoder,
        'fertilizer': fertilizer_encoder,
        'feature_cols': feature_cols
    }, f)

print("Fertilizer model and encoders saved successfully.")
