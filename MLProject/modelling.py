import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. MEMUAT DATASET
# ==========================================
print("Memuat dataset hasil preprocessing...")
# Pastikan file ini adalah data yang SUDAH bersih dan di-scaling dari eksperimen
df = pd.read_csv('heart_preprocessed.csv')

# Memisahkan fitur dan target
X = df.drop('target', axis=1)
y = df['target']

# Membagi data latih dan uji (hanya untuk proses validasi saat training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. SETUP MLFLOW TRACKING
# ==========================================
# MATIKAN DUA BARIS INI UNTUK GITHUB ACTIONS (Sesuai perbaikan sebelumnya)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_experiment("Eksperimen_Heart_Disease_Baseline")

# ==========================================
# 3. AUTOLOG & TRAINING MODEL
# ==========================================
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Baseline"):
    print("Melatih model dasar (TANPA Hyperparameter Tuning)...")
    
    rf_model = RandomForestClassifier(random_state=42)
    
    # Model langsung dilatih menggunakan data yang sudah preprocessed
    rf_model.fit(X_train, y_train)
    
    print("--- Sukses! Model dasar berhasil dilatih dengan data ---")