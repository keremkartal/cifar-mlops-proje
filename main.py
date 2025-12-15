import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from PIL import Image
import mlflow.keras 
import os

app = FastAPI(title="CIFAR-10 MLflow API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.on_event("startup")
def load_model():
    global model
    try:
        model_name = "CIFAR10_CNN_Model"
        model_stage = "Staging"
        
        print(f"MLflow'dan model indiriliyor... ({MLFLOW_TRACKING_URI})")
        
        model_uri = f"models:/{model_name}/{model_stage}"
        
        model = mlflow.keras.load_model(model_uri)
        print(f"Model ({model_name} - {model_stage}) başarıyla yüklendi!")
        
    except Exception as e:
        print(f"KRİTİK HATA: Model MLflow'dan çekilemedi. Detay: {e}")

@app.get("/")
def read_root():
    return {"message": "CIFAR-10 Model API'sine hoş geldiniz. Test için /docs adresine gidin."}

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Gelen byte'ları modele uygun hale getirir."""
    image = Image.open(io.BytesIO(image_bytes))
    
    image = image.resize((32, 32))
    
    image_array = np.array(image)
    
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    image_array = image_array / 255.0
    
    return np.expand_dims(image_array, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Bir resim dosyası alır, işler ve tahmin sonucunu döndürür.
    """
    if model is None:
        return {"error": "Model yüklenemedi. Lütfen sunucu loglarını kontrol edin."}
        
    if not file.content_type.startswith('image/'):
        return {"error": "Hatalı dosya tipi. Lütfen bir resim dosyası yükleyin."}

    image_bytes = await file.read()
    
    try:
        processed_image = preprocess_image(image_bytes)
    except Exception as e:
        return {"error": f"Resim işlenirken hata oluştu: {e}"}

    try:
        prediction = model.predict(processed_image)
        
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(prediction))
        
        return {
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "all_predictions": {class_names[i]: float(prediction[0][i]) for i in range(10)}
        }
    except Exception as e:
        return {"error": f"Tahmin yapılırken hata oluştu: {e}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
