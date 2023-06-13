import streamlit as st
import cv2
import numpy as np
from PIL import Image

from tensorflow import keras
from tensorflow.keras.applications import xception
from keras.models import load_model
from tensorflow.keras import models
from tensorflow.core.protobuf import rpc_options_pb2
import tensorflow as tf
print(tf.__version__)


# Charger le premier modèle pour prédire organique ou recyclable
organic_recyclable_model = load_model('model_o_r.h5')

# Charger le deuxième modèle pour prédire le type de matériau
material_model_weight_path = load_model('model_type_materiel_2.h5')

# Dictionnaire pour mapper les étiquettes de matériau aux classes
material_categories = {0: 'paper', 1: 'cardboard', 
                       2: 'plastic', 3: 'metal', 4: 'trash', 
                       5: 'battery', 6: 'shoes', 7: 'clothes', 
                       8: 'green-glass', 
                       9: 'brown-glass', 10: 'white-glass', 
                       11: 'biological'}




# Créer le modèle Xception avec les mêmes paramètres que vous avez utilisés auparavant
xception_layer = xception.Xception(include_top=False, input_shape=(224, 224, 3),
                                  weights='imagenet')

# Ne pas entraîner les poids importés
xception_layer.trainable = False

def xception_preprocessing(img):
    return xception.preprocess_input(img)

model = keras.Sequential()
model.add(keras.Input(shape=(224, 224, 3)))
model.add(keras.layers.Lambda(xception_preprocessing))
model.add(xception_layer)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(len(material_categories), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary()

# Charger les poids à partir du fichier
model.load_weights(material_model_weight_path)

material_model = model

def predict_organic_recyclable(img):
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224, 3])
    result = np.argmax(organic_recyclable_model.predict(img))
    return result

def predict_material(img):
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224, 3])
    result = np.argmax(material_model.predict(img))
    return result

def main():
    st.title("Prédiction de recyclabilité et de matériau")
    st.write("Téléchargez une photo et nous vous dirons si elle est organique ou recyclable, "
             "et si elle est recyclable, nous prédirons son type de matériau.")

    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        st.image(image, caption='Image téléchargée', use_column_width=True)

        organic_recyclable_prediction = predict_organic_recyclable(img_array)

        if organic_recyclable_prediction == 0:
            st.write("Prédiction : Cette image est organique.")
        elif organic_recyclable_prediction == 1:
            st.write("Prédiction : Cette image est recyclable.")

            material_prediction = predict_material(img_array)
            material_label = material_categories[material_prediction]
            st.write("Prédiction de matériau : ", material_label)


if __name__ == "__main__":
    main()
