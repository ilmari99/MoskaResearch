import tensorflow as tf

model = tf.keras.models.load_model("/home/ilmari/python/moska/Model6-39/model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("./Model6-39/model.tflite", "wb") as f:
    f.write(tflite_model)


