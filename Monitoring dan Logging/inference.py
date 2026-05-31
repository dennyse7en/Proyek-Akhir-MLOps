import pickle
import pandas as pd

# Fungsi untuk memuat model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk melakukan prediksi
def make_prediction(data_dict):
    model = load_model()
    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)
    return int(prediction[0])