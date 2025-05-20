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
   modelo_url = "https://drive.google.com/uc?id=1ZQUu0LRuZdnbDQJJaIeXgv-IUYVYppoo"
   modelo_path = "modelo_clasificador_Arte.h5"
   if not os.path.exists(modelo_path):
       with st.spinner("Descargando modelo ...."):
            gdown.download(modelo_url, modelo_path, quiet=False)
            st.success("Modelo descargado correctamente.")

def cargar_modelo():
    descargar_modelo()
    modelo = tf.keras.models.load_model("modelo_clasificador_Arte.h5")
    with open("clases.json", "r") as f:
        clases = json.load(f)
    indice_a_clase = {v: k for k, v in clases.items()}
    return modelo, indice_a_clase


st.markdown("Sube una imagen: ")
# Cargar modelo y clases
modelo, indice_a_clase = cargar_modelo()

archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen_pil = Image.open(archivo).convert('RGB') #convierte la imagen a escalas de rojo, verde y azul
    st.image(imagen_pil, caption="📷 Imagen cargada", use_column_width=True)

    # Preprocesar imagen
    imagen_array = imagen_pil.resize((224, 224)) #Redimensiona la imagen a 224x224 píxeles,
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
