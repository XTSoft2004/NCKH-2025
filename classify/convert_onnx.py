import os
import argparse
import keras
import tf2onnx
import tensorflow as tf
import onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the Keras model file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the ONNX model file",
    )
    args = parser.parse_args()

    # Validate input arguments
    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"Keras model file not found: {args.input_path}")
    if not args.output_path.endswith(".onnx"):
        raise ValueError("Output path must have a .onnx extension")

    print(
        f"Converting Keras model from {args.input_path} to ONNX model at {args.output_path}"
    )

    # Load the Keras model
    keras_model = keras.models.load_model(args.input_path)

    # Convert the Keras model to ONNX format
    spec = (tf.TensorSpec(keras_model.inputs[0].shape, tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(
        keras_model, input_signature=spec, opset=13
    )

    # Save the ONNX model to the specified output path
    onnx.save(onnx_model, args.output_path)

    print(f"ONNX model saved to {args.output_path}")
