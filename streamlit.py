import streamlit as st
import numpy as np
import tensorflow as ts
from PIL import Image
import json

st.set_page_config(page_title="Clasificador de estilos artisticos", layout="centered")
st.title("IA Clasificadora de estilos artisticos")

st.markdown("Sube una imagen: ")

def cargar_modelo():
    modelo = ts.keras.models.load_model("modelo_clasificador_Arte.h5") #cargar el modelo guardado en la clase modelo
    with open("clases.json", "r") as f: #se cargan todas las clases posibles
        clases = json.load(f)
    indice_a_clase = {v: k for k, v in clases.items()}
    return modelo, indice_a_clase

modelo, indice_a_clase = cargar_modelo()

archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen_pil = Image.open(archivo).convert('RGB') #convierte la imagen a escalas de rojo, verde y azul
    st.image(imagen_pil, caption="📷 Imagen cargada", use_column_width=True)

    # Preprocesar imagen
    imagen_array = imagen_pil.resize((224, 224)) #Redimensiona la imagen a 224x224 píxeles,
    imagen_array = ts.keras.preprocessing.image.img_to_array(imagen_array)
    imagen_array = np.expand_dims(imagen_array, axis=0) #Añade una dimensión extra al array
    imagen_array = imagen_array / 255.0 #mejorar el rendimiento y estabilidad 

    # Predecir la imagen 
    predicciones = modelo.predict(imagen_array)
    indice = np.argmax(predicciones[0])
    probabilidad = predicciones[0][indice]
    clase_predicha = indice_a_clase[indice]

    st.markdown(f"Su imagen pertenece al estilo : **{clase_predicha}**")
    st.markdown(f"Con un accuracy del **{probabilidad*100:.2f}%")
