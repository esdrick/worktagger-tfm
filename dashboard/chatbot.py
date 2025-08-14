import streamlit as st
import pandas as pd
import requests

# -----------------------------
# 🤖 Chatbot de recomendaciones personalizadas
# -----------------------------
def show_productivity_chatbot():
    with st.expander("🤖 Asistente de productividad (beta)", expanded=False):
        st.markdown("Hazle preguntas al asistente sobre cómo estás usando tu tiempo o cómo podrías mejorar tu productividad.")

        openrouter_key = st.text_input("🔐 Clave API de OpenRouter", type="password", key="openrouter_key_prod")
        user_question = st.text_area("🧠 ¿En qué necesitas ayuda?", placeholder="¿En qué actividad gasto más tiempo? ¿Cómo puedo organizar mejor mi semana?")

        if st.button("🔍 Consultar al Chatbot de productividad"):
            if not openrouter_key:
                st.warning("⚠️ Debes ingresar tu clave de OpenRouter.")
            elif not user_question.strip():
                st.warning("⚠️ Escribe una pregunta para consultar.")
            else:
                with st.spinner("💬 Pensando..."):
                    df_graph = st.session_state.get("df_graph")
                    if df_graph is not None and not df_graph.empty:
                        resumen = df_graph.groupby("Subactivity")["Duration"].sum().sort_values(ascending=False).head(10)
                        resumen_str = "\n".join([f"{k}: {int(v/60)} minutos" for k, v in resumen.items()])
                    else:
                        resumen_str = "No hay datos disponibles para analizar."

                    prompt = f"""Tengo un registro de actividades laborales con esta distribución de tiempo:
{resumen_str}

Ahora quiero que me ayudes con esta pregunta: {user_question}
Responde como un coach experto en gestión del tiempo en el trabajo, con lenguaje claro y práctico."""

                    api_url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://tu-aplicacion.streamlit.app"
                    }

                    payload = {
                        "model": "mistralai/mistral-7b-instruct:free",
                        "messages": [
                            {"role": "system", "content": "Eres un coach experto en productividad que analiza datos de tiempo laboral."},
                            {"role": "user", "content": prompt}
                        ]
                    }

                    response = requests.post(api_url, headers=headers, json=payload)

                    if response.status_code == 200:
                        respuesta = response.json()["choices"][0]["message"]["content"]
                        st.success("💡 Recomendación del asistente:")
                        st.markdown(respuesta)
                    else:
                        st.error(f"❌ Error al conectar con OpenRouter:\n{response.status_code} - {response.text}")