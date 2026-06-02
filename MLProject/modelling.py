import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. MEMUAT DATASET
# ==========================================
print("Memuat data...")
# Pastikan file heart.csv berada di folder yang sama saat mengeksekusi script ini
df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
y = df['target']

# ==========================================
# 2. PREPROCESSING DATA
# ==========================================
print("Melakukan preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. SETUP MLFLOW TRACKING
# ==========================================
# Menggunakan database SQLite lokal sesuai standar proyek Anda
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Eksperimen_Heart_Disease_Baseline")

# ==========================================
# 4. REVISI UTAMA: AUTOLOG & TRAINING MODEL
# ==========================================
# WAJIB: Aktifkan autolog tepat SEBELUM mlflow.start_run()
# Fungsi ini otomatis mencatat semua parameter, metrik (akurasi, dll), dan model artifact
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Baseline"):
    print("Melatih model dasar (TANPA Hyperparameter Tuning)...")
    
    # Membuat model Random Forest Polos tanpa param_grid / GridSearchCV
    rf_model = RandomForestClassifier(random_state=42)
    
    # Melatih model menggunakan data yang sudah di-scaling
    # Proses .fit() ini akan memicu autolog mencatat semuanya ke MLflow secara otomatis
    rf_model.fit(X_train_scaled, y_train)
    
    print("--- Sukses! Model dasar berhasil dilatih ---")
    print("Seluruh metrik dan model telah dicatat otomatis oleh MLflow Autolog.")
    print("Silakan jalankan 'mlflow ui' di terminal untuk melihat hasilnya.")