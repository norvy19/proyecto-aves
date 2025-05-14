import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import os
from tensorflow.keras.applications.vgg16 import preprocess_input

# --- Ruta base de las imágenes ---
RUTA_IMAGENES = "dataset/test"

# --- Cargar el modelo TFLite ---
@st.cache_resource
def cargar_modelo():
    interpreter = tf.lite.Interpreter(model_path="modelo_convertido.tflite")
    interpreter.allocate_tensors()
    return interpreter

# --- Preprocesamiento para VGG16 ---
def preparar_imagen_vgg16(imagen):
    imagen = imagen.convert("RGB")
    imagen = imagen.resize((224, 224))
    matriz = np.array(imagen)
    matriz = preprocess_input(matriz)
    matriz = np.expand_dims(matriz, axis=0)
    return matriz.astype(np.float32)

# --- Etiquetas del modelo ---
etiquetas = [
    'Dacnis Carinegro', 'Tangara Cabecibaya', 'Tangara Carafuego',
    'Tangara Cenicienta', 'Tangara Coronigualda', 'Tangara Dorada',
    'Tangara Goliplateada', 'Tangara Golondrina', 'Tangara Verdinegra'
]

# --- Cargar información adicional desde Excel ---
@st.cache_data
def cargar_info_aves():
    return pd.read_excel("aves_info.xlsx", sheet_name="informacion")

info_aves = cargar_info_aves()

# --- Título de la aplicación ---
st.title("🦜 Clasificación de Aves con VGG16 + TensorFlow Lite")
st.write("Sube una imagen de un ave para predecir su especie.")

# --- Cargar imagen del usuario ---
archivo_imagen = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    imagen_preparada = preparar_imagen_vgg16(imagen)

    # --- Cargar modelo e inferencia ---
    interpreter = cargar_modelo()
    entrada = interpreter.get_input_details()
    salida = interpreter.get_output_details()

    interpreter.set_tensor(entrada[0]['index'], imagen_preparada)
    interpreter.invoke()
    salida_predicha = interpreter.get_tensor(salida[0]['index'])

    # --- Obtener top 3 predicciones ---
    probabilidades = salida_predicha[0]
    top_3_indices = np.argsort(probabilidades)[-3:][::-1]

    st.subheader("🔝 Top 3 especies más probables:")
    for i in top_3_indices:
        especie = etiquetas[i]
        probabilidad = probabilidades[i] * 100
        st.success(f"🧠 {especie}: {probabilidad:.2f}%")

        # Mostrar información y una imagen con un expander
        with st.expander(f"🔎 Ver información de {especie}"):
            info = info_aves[info_aves["especie"] == especie]
            if not info.empty:
                st.write(f"**Nombre científico:** {info.iloc[0]['nombre_cientifico']}")
                st.write(f"**Descripción:** {info.iloc[0]['descripcion']}")
                st.write(f"**Hábitat:** {info.iloc[0]['habitat']}")
                st.write(f"**Peso promedio:** {info.iloc[0]['peso_promedio']}")
                st.write(f"**Alimentación:** {info.iloc[0]['alimentacion']}")
            else:
                st.warning("No se encontró información para esta especie.")

            # Buscar imagen representativa de la especie
            ruta_carpeta = os.path.join(RUTA_IMAGENES, especie)
            if os.path.exists(ruta_carpeta):
                archivos = [f for f in os.listdir(ruta_carpeta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if archivos:
                    ruta_imagen_especie = os.path.join(ruta_carpeta, archivos[0])
                    imagen_especie = Image.open(ruta_imagen_especie)
                    st.image(imagen_especie, caption=f"Ejemplo de {especie}", use_container_width=True)
                else:
                    st.info("No se encontraron imágenes en la carpeta de esta especie.")
            else:
                st.info("No existe carpeta con el nombre de esta especie.")


