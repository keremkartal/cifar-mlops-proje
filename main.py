import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

app = FastAPI(title="CIFAR-10 Görüntü Sınıflandırma API")

try:
    model = tf.keras.models.load_model('cifar10_model.h5')
    print("Model 'cifar10_model.h5' başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Model yüklenemedi. {e}")
    model = None

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