import streamlit as st
import numpy as np
import pandas as pd
import joblib


# ==========================================
# CONFIGURACION
# ==========================================

st.set_page_config(
    page_title="Laboratorio IA",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# CARGA DE MODELOS
# ==========================================

modelo_viviendas = joblib.load("modelos/modelo.joblib")
scaler_viviendas = joblib.load("modelos/scaler.joblib")

modelo_creditos = joblib.load("modelos/creditos_logistic.joblib")

# ==========================================
# TITULO GENERAL
# ==========================================

st.title("🤖 Laboratorio de IA")

st.markdown("""
Demostraciones prácticas de Inteligencia Artificial aplicadas a negocio.

**Machine Learning → NLP → IA Generativa → Agentes**
""")

# ==========================================
# TABS
# ==========================================

tab1, tab2 = st.tabs([
    "🏠 ¿Cuánto vale mi piso?",
    "💳 ¿Me aprobarán el crédito?"
])

# ======================================================
# TAB 1 - VIVIENDAS
# ======================================================

with tab1:

    st.header("🏠 Predicción del precio de una vivienda")

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

    terraza_num = 1 if terraza == "Sí" else 0
    ascensor_num = 1 if ascensor == "Sí" else 0

    if st.button("Calcular precio"):

        datos = np.array([[
            metros_cuadrados,
            habitaciones,
            ban,
            planta,
            terraza_num,
            ascensor_num
        ]])

        datos_scaled = scaler_viviendas.transform(datos)

        prediccion = modelo_viviendas.predict(datos_scaled)

        precio = round(prediccion[0], 0)

        st.success(
            f"💰 Precio estimado: {precio:,.0f} €"
        )

        st.info(
            "Modelo de Machine Learning de tipo Regresión."
        )

# ======================================================
# TAB 2 - CREDITOS
# ======================================================

with tab2:

    st.header("💳 Evaluación de crédito")

    st.markdown("""
    Ejemplo de Machine Learning aplicado a la evaluación
    de solicitudes de crédito.
    """)

    amount = st.slider(
        "Importe solicitado (€)",
        min_value=250,
        max_value=18424,
        value=5000,
        step=250
    )

    duration = st.slider(
        "Duración del crédito (meses)",
        min_value=3,
        max_value=72,
        value=24,
        step=1
    )

    credit_history_options = {
        "Retrasos de pago anteriores":
            "delay in paying off in the past",

        "Historial crediticio delicado":
            "critical account/other credits existing",

        "Buen historial de pagos":
            "existing credits paid back duly till now",

        "Sin incidencias crediticias":
            "no credits taken/all credits paid back duly",

        "Sin historial bancario":
            "all credits at this bank paid back duly"
    }

    credit_history_text = st.selectbox(
        "Historial crediticio",
        list(credit_history_options.keys())
    )

    credit_history = credit_history_options[
        credit_history_text
    ]

    savings_options = {
        "Sin ahorros o desconocido":
            "unknown/no savings account",

        "Ahorros bajos":
            "... < 100 DM",

        "Ahorros medios":
            "100 <= ... < 500 DM",

        "Ahorros altos":
            "500 <= ... < 1000 DM",

        "Ahorros muy altos":
            "... >= 1000 DM"
    }

    savings_text = st.selectbox(
        "Nivel de ahorro",
        list(savings_options.keys())
    )

    savings = savings_options[
        savings_text
    ]

    employment_options = {
        "Desempleado":
            "unemployed",

        "Menos de 1 año":
            "... < 1 year",

        "Entre 1 y 4 años":
            "1 <= ... < 4 years",

        "Entre 4 y 7 años":
            "4 <= ... < 7 years",

        "Más de 7 años":
            "... >= 7 years"
    }

    employment_text = st.selectbox(
        "Antigüedad laboral",
        list(employment_options.keys())
    )

    employment_duration = employment_options[
        employment_text
    ]

    if st.button("Evaluar crédito"):

        datos = pd.DataFrame([{
            "amount": amount,
            "duration": duration,
            "credit_history": credit_history,
            "savings": savings,
            "employment_duration": employment_duration
        }])

        probas = modelo_creditos.predict_proba(
            datos
        )[0]

        # En tu modelo:
        # Clase 1 = buen cliente
        # Clase 0 = mal cliente

        prob_aprobacion = probas[1]
        prob_riesgo = probas[0]

        st.metric(
            "Probabilidad de aprobación",
            f"{prob_aprobacion*100:.1f}%"
        )

        if prob_aprobacion >= 0.70:

            st.success(
                "✅ Crédito recomendado"
            )

            decision = "Aprobado"

        elif prob_aprobacion >= 0.40:

            st.warning(
                "⚠️ Revisión manual"
            )

            decision = "Revisión"

        else:

            st.error(
                "❌ Riesgo elevado"
            )

            decision = "Rechazado"

        st.markdown("---")

        st.subheader("Resultado")

        st.write(
            f"**Decisión:** {decision}"
        )

        st.write(
            f"**Probabilidad de aprobación:** {prob_aprobacion*100:.1f}%"
        )

        st.write(
            f"**Probabilidad de riesgo:** {prob_riesgo*100:.1f}%"
        )

        st.markdown("---")

        st.subheader("Perfil evaluado")

        st.write(
            f"💶 Importe solicitado: {amount:,.0f} €"
        )

        st.write(
            f"📅 Duración: {duration} meses"
        )

        st.write(
            f"🏦 Historial: {credit_history_text}"
        )

        st.write(
            f"💰 Ahorros: {savings_text}"
        )

        st.write(
            f"💼 Antigüedad laboral: {employment_text}"
        )

        st.info(
            "Modelo Logistic Regression."
        )
        
