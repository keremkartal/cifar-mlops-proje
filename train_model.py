import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from mlflow.models.signature import infer_signature
import numpy as np
import os

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "CIFAR-10-Advanced-Proje"
MODEL_NAME = "CIFAR10_CNN_Production_Model"
ACCURACY_THRESHOLD = 0.60  
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def train_advanced():
    print("Veri yükleniyor...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    EPOCHS = 3
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    OPTIMIZER_NAME = "Adam"

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    with mlflow.start_run(run_name="Advanced-CNN-Training") as run:
        
        mlflow.keras.autolog(log_models=False) 
        print("Parametreler loglanıyor...")
        mlflow.log_param("custom_epochs", EPOCHS)
        mlflow.log_param("custom_batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer_name", OPTIMIZER_NAME)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("threshold_limit", ACCURACY_THRESHOLD)

        print("Eğitim başlıyor...")
        history = model.fit(train_images, train_labels, 
                            epochs=EPOCHS, 
                            batch_size=BATCH_SIZE,
                            validation_data=(test_images, test_labels))

        print("Model değerlendiriliyor...")
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
        
        mlflow.log_metric("final_test_loss", loss)
        mlflow.log_metric("final_test_accuracy", accuracy)
        
        print(f"Final Accuracy: {accuracy}")

        with open("class_names.txt", "w") as f:
            f.write("\n".join(class_names))
        mlflow.log_artifact("class_names.txt", artifact_path="configs")
      
        with open("model_summary.txt", "w") as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact("model_summary.txt", artifact_path="model_info")

        if accuracy >= ACCURACY_THRESHOLD:
            print(f"BAŞARILI! Model doğruluğu ({accuracy:.4f}) eşiği ({ACCURACY_THRESHOLD}) geçti.")
         
            model.save("cifar10_model.h5")
            print("Model yerel diske 'cifar10_model.h5' olarak da kaydedildi (Docker için).")

            input_example = train_images[0:1]
            prediction_example = model.predict(input_example)
            signature = infer_signature(input_example, prediction_example)
            
            tracking_url_type_store = mlflow.get_tracking_uri()
            
            model_info = mlflow.keras.log_model(
                model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=MODEL_NAME, 
                pip_requirements=["tensorflow==2.16.1", "numpy", "pandas"] 
            )
           
            client = MlflowClient()
            
            latest_version_info = client.get_latest_versions(MODEL_NAME, stages=["None"])
            if latest_version_info:
                latest_version = latest_version_info[0].version
                
                client.update_model_version(
                    name=MODEL_NAME,
                    version=latest_version,
                    description=f"Bu model Accuracy > {ACCURACY_THRESHOLD} koşulunu sağladı. Otomatik Staging'e taşındı."
                )
                
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=latest_version,
                    stage="Staging",
                    archive_existing_versions=True 
                )
                print(f"Model Version {latest_version} 'Staging' aşamasına taşındı.")
            
        else:
            print(f"BAŞARISIZ: Model doğruluğu ({accuracy:.4f}) eşiğin ({ACCURACY_THRESHOLD}) altında kaldı.")
            print("Model Registry'ye kaydedilmedi.")

        if os.path.exists("class_names.txt"): os.remove("class_names.txt")
        if os.path.exists("model_summary.txt"): os.remove("model_summary.txt")

if __name__ == "__main__":
    train_advanced()