import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow.keras.models import load_model
import json

def export_to_tflite(
    model_path="models/emg_decoder.h5",
    out_dir="microcontroller",
    quantize=True,
    optimize=True,
    target_platform="arduino"
):
    """
    Export a Keras model to TensorFlow Lite format with optimizations.
    
    Args:
        model_path: Path to the input Keras model
        out_dir: Directory to save the exported model
        quantize: Whether to apply quantization
        optimize: Whether to apply optimizations
        target_platform: Target platform ('arduino', 'esp32', or 'generic')
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    
    # Configure converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    if optimize:
        print("Applying optimizations...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Platform-specific optimizations
        if target_platform == "arduino":
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.target_spec.supported_types = [tf.float16]
        elif target_platform == "esp32":
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.target_spec.supported_types = [tf.int8]
    
    # Apply quantization if requested
    if quantize:
        print("Applying quantization...")
        def representative_dataset():
            # Generate random input data for quantization
            # This should be replaced with actual calibration data
            for _ in range(100):
                data = np.random.random((1, 200, 8)).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert model
    print("Converting model...")
    tflite_model = converter.convert()
    
    # Save model
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    out_path = os.path.join(out_dir, f"{model_name}.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    
    # Save model metadata
    metadata = {
        "model_name": model_name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "quantized": quantize,
        "optimized": optimize,
        "target_platform": target_platform,
        "tensorflow_version": tf.__version__,
        "model_size_bytes": os.path.getsize(out_path)
    }
    
    metadata_path = os.path.join(out_dir, f"{model_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model exported to {out_path}")
    print(f"✅ Metadata saved to {metadata_path}")
    
    # Print model size
    model_size_kb = os.path.getsize(out_path) / 1024
    print(f"Model size: {model_size_kb:.1f} KB")
    
    return out_path, metadata_path

def generate_arduino_header(model_path, out_dir="microcontroller"):
    """Generate Arduino header file for the model"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    header_path = os.path.join(out_dir, f"{model_name}.h")
    
    # Read model data
    with open(model_path, "rb") as f:
        model_data = f.read()
    
    # Generate header
    header_content = f"""#ifndef {model_name.upper()}_H
#define {model_name.upper()}_H

#include <Arduino.h>

// Model data
const unsigned char {model_name}[] = {{
    {', '.join(f'0x{b:02x}' for b in model_data)}
}};

const unsigned int {model_name}_len = {len(model_data)};

#endif // {model_name.upper()}_H
"""
    
    with open(header_path, "w") as f:
        f.write(header_content)
    
    print(f"✅ Arduino header generated: {header_path}")

def main():
    parser = argparse.ArgumentParser(description="Export Keras model to TFLite format")
    parser.add_argument("--model", default="models/emg_decoder.h5",
                      help="Path to input Keras model")
    parser.add_argument("--out-dir", default="microcontroller",
                      help="Output directory for exported files")
    parser.add_argument("--no-quantize", action="store_true",
                      help="Disable quantization")
    parser.add_argument("--no-optimize", action="store_true",
                      help="Disable optimizations")
    parser.add_argument("--platform", default="arduino",
                      choices=["arduino", "esp32", "generic"],
                      help="Target platform")
    parser.add_argument("--generate-header", action="store_true",
                      help="Generate Arduino header file")
    
    args = parser.parse_args()
    
    # Export model
    model_path, metadata_path = export_to_tflite(
        model_path=args.model,
        out_dir=args.out_dir,
        quantize=not args.no_quantize,
        optimize=not args.no_optimize,
        target_platform=args.platform
    )
    
    # Generate Arduino header if requested
    if args.generate_header:
        generate_arduino_header(model_path, args.out_dir)

if __name__ == "__main__":
    main()
