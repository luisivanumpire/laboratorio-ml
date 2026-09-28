import streamlit as st
import numpy as np
import pandas as pd
import joblib


# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="¿Cuánto vale mi piso?",
    page_icon="🏠",
    layout="centered"
)


# ==========================================
# CARGA DEL MODELO
# ==========================================

modelo_viviendas = joblib.load("modelos/modelo.joblib")
scaler_viviendas = joblib.load("modelos/scaler.joblib")


# ==========================================
# TÍTULO
# ==========================================

st.title("🏠 ¿Cuánto vale mi piso?")

st.markdown("""
### Predicción del precio de una vivienda

Introduce las características de tu vivienda y el modelo de
Machine Learning estimará su precio.
""")


# ==========================================
# DATOS DE ENTRADA
# ==========================================

st.subheader("Características de la vivienda")

metros_cuadrados = st.number_input(
    "Metros cuadrados",
    min_value=20,
    max_value=500,
    value=80
)

habitaciones = st.number_input(
    "Habitaciones",
    min_value=1,
    max_value=10,
    value=3
)

ban = st.number_input(
    "Baños",
    min_value=1,
    max_value=10,
    value=2
)

planta = st.number_input(
    "Planta",
    min_value=0,
    max_value=50,
    value=3
)

terraza = st.selectbox(
    "Terraza",
    ["Sí", "No"]
)

ascensor = st.selectbox(
    "Ascensor",
    ["Sí", "No"]
)


# ==========================================
# CONVERSIÓN DE VARIABLES CATEGÓRICAS
# ==========================================

terraza_num = 1 if terraza == "Sí" else 0
ascensor_num = 1 if ascensor == "Sí" else 0


# ==========================================
# PREDICCIÓN
# ==========================================

if st.button("💰 Calcular precio"):

    datos = np.array([[
        metros_cuadrados,
        habitaciones,
        ban,
        planta,
        terraza_num,
        ascensor_num
    ]])

    # Aplicar el mismo scaler utilizado durante
    # el entrenamiento del modelo
    datos_scaled = scaler_viviendas.transform(datos)

    # Realizar predicción
    prediccion = modelo_viviendas.predict(datos_scaled)

    precio = round(prediccion[0], 0)

    # ==========================================
    # RESULTADO
    # ==========================================

    st.success(
        f"💰 Precio estimado: {precio:,.0f} €"
    )

    st.info(
        "Modelo de Machine Learning de tipo Regresión."
    )


# ==========================================
# INFORMACIÓN
# ==========================================

st.markdown("---")

st.caption(
    "Laboratorio de IA · Demostración de Machine Learning"
)
```
