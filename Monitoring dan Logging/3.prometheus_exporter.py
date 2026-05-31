import time
from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter, Histogram
from inference import make_prediction

app = FastAPI()

# Menambahkan rute /metrics standar untuk Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# --- DEKLARASI METRIK PROMETHEUS ---
REQUEST_COUNT = Counter('api_requests_total', 'Total request ke API')
PREDICTION_COUNT = Counter('model_predictions_total', 'Total prediksi yang dihasilkan', ['result'])
LATENCY = Histogram('api_latency_seconds', 'Waktu respons API')

@app.post("/predict")
def predict(age: int, sex: int, cp: int, trestbps: int, chol: int, fbs: int, 
            restecg: int, thalach: int, exang: int, oldpeak: float, slope: int, ca: int, thal: int):
    
    # Mencatat waktu mulai
    start_time = time.time()
    REQUEST_COUNT.inc() # Menambah counter request
    
    # Menyiapkan data
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    
    # Melakukan prediksi menggunakan file inference.py
    hasil = make_prediction(data)
    
    # Mencatat hasil prediksi ke metrik
    label_hasil = "sakit" if hasil == 1 else "sehat"
    PREDICTION_COUNT.labels(result=label_hasil).inc()
    
    # Mencatat waktu selesai (latensi)
    process_time = time.time() - start_time
    LATENCY.observe(process_time)
    
    return {"prediction": hasil, "status": label_hasil}