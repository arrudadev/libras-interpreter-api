import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import joblib

dir_path = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(dir_path, 'saved_model')
scaler_path = os.path.join(dir_path, 'scaler.pkl')
pca_path = os.path.join(dir_path, 'pca.pkl')

signals_model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)
