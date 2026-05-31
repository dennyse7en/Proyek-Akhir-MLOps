import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Memuat Data
print("Memuat data...")
df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Preprocessing
print("Melakukan preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Setup MLflow
# Memaksa MLflow menggunakan database SQLite agar tidak tersasar di Windows
mlflow.set_tracking_uri("sqlite:///mlflow.db") 

# Mengatur nama eksperimen di MLflow
mlflow.set_experiment("Eksperimen_Heart_Disease")

# Memulai run MLflow
with mlflow.start_run(run_name="RandomForest_Tuning"):
    print("Melatih model dan melakukan Hyperparameter Tuning...")
    
    # 4. Membuat Model & Hyperparameter Tuning (Untuk Nilai Advanced)
    rf = RandomForestClassifier(random_state=42)
    
    # Parameter yang akan diuji
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }
    
    # Mencari parameter terbaik
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    # Mengambil model terbaik dan parameternya
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # LOGGING KE MLFLOW: Menyimpan parameter terbaik
    mlflow.log_params(best_params)
    
    # 5. Evaluasi Model
    print("Mengevaluasi model...")
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    # LOGGING KE MLFLOW: Menyimpan metrik evaluasi
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    
    # LOGGING KE MLFLOW: Menyimpan artifak model
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    print(f"\nSukses! Model dilatih dengan Akurasi: {acc:.2f}")
    print("Silakan jalankan 'mlflow ui' di terminal untuk melihat hasilnya.")