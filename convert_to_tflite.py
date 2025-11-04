import tensorflow as tf

# Convert the TensorFlow SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("face_parsing_tf_savedmodel")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # for quantization
tflite_model = converter.convert()

with open("face_parsing.tflite", "wb") as f:
    f.write(tflite_model)
print("face_parsing.tflite created successfully.")
