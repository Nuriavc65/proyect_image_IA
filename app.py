import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
import urllib.request
import requests
import gdown

st.set_page_config(page_title="Clasificador de estilos artisticos", layout="centered")
st.title("IA Clasificadora de estilos artisticos")

  
def descargar_modelo(): 
    modelo_url = "https://drive.google.com/uc?export=download&id=13VFre1SU4i4UzrQmsRxfYCJOkCkeAKzu"
    if not os.path.exists("modelo_clasificador_Arte.keras"):
        with st.spinner("Descargando modelo ....."):
            urllib.request.urlretrieve(modelo_url, "modelo_clasificador_Arte.keras")
            st.success("Modelo descargado exitosamente.")

def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_clasificador_Arte.keras")
    with open("clases.json", "r") as f:
        clases = json.load(f)
    indice_a_clase = {v: k for k, v in clases.items()}
    return modelo, indice_a_clase

descargar_modelo()
modelo, indice_a_clase = cargar_modelo()

st.markdown("Sube una imagen: ")
archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen_pil = Image.open(archivo).convert('RGB') #convierte la imagen a escalas de rojo, verde y azul
    st.image(imagen_pil, caption="📷 Imagen cargada", use_column_width=True)

    # Preprocesar imagen
    imagen_array = imagen_pil.resize((128, 128)) #Redimensiona la imagen a 128x128 píxeles,
    imagen_array = tf.keras.preprocessing.image.img_to_array(imagen_array)
    imagen_array = np.expand_dims(imagen_array, axis=0) #Añade una dimensión extra al array
    imagen_array = imagen_array / 255.0 #mejorar el rendimiento y estabilidad 

    # Predecir la imagen 
    predicciones = modelo.predict(imagen_array)
    indice = np.argmax(predicciones[0])
    probabilidad = predicciones[0][indice]
    clase_predicha = indice_a_clase[indice]

    st.markdown(f"Su imagen pertenece al estilo : **{clase_predicha}**")
    st.markdown(f"Con un accuracy del **{probabilidad*100:.2f}%**")
