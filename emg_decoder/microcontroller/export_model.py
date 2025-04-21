
import tensorflow as tf

def export_to_tflite(model_path="models/cnn_lstm_model.h5", out_path="models/cnn_lstm_model.tflite"):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Model exported to {out_path}")

if __name__ == "__main__":
    export_to_tflite()
