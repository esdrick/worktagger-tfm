import streamlit as st
import pandas as pd
import requests
import time

# -----------------------------
# 🤖 Chatbot de recomendaciones personalizadas
# -----------------------------
def show_productivity_chatbot():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente de productividad. ¿En qué puedo ayudarte hoy?"}
        ]
    
    # CSS más específico para forzar los estilos
    st.markdown("""
    <style>
    /* Contenedor principal */
    .chat-container {
        border: 2px solid #e1e5e9;
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
        background-color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Header del chat */
    .chat-header {
        background-color: #f8f9fa;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #e1e5e9;
        font-weight: 600;
        color: #333;
        text-align: left;
        font-size: 1.1rem;
    }
    
    /* Forzar estilos en mensajes del usuario */
    .stChatMessage[data-testid="user-message-container"] {
        flex-direction: row-reverse !important;
    }
    
    .stChatMessage[data-testid="user-message-container"] > div:first-child {
        background-color: #FFA500 !important;
        color: white !important;
        border-radius: 18px 18px 5px 18px !important;
        margin-left: 20% !important;
        margin-right: 0 !important;
    }
    
    /* Forzar estilos en mensajes del asistente */
    .stChatMessage[data-testid="assistant-message-container"] > div:first-child {
        background-color: white !important;
        color: #333 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 18px 18px 18px 5px !important;
        margin-right: 20% !important;
        margin-left: 0 !important;
    }
    
    /* Alternativa: usar selectores más generales */
    div[data-testid="stChatMessage"] {
        margin-bottom: 1rem;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="user-avatar"]) {
        display: flex;
        flex-direction: row-reverse;
        justify-content: flex-start;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="user-avatar"]) > div {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 18px 18px 5px 18px !important;
        max-width: 75% !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="assistant-avatar"]) > div {
        background-color: white !important;
        color: #333 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 18px 18px 18px 5px !important;
        max-width: 75% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Instrucciones fuera del contenedor
    st.markdown("Hazle preguntas al asistente sobre cómo estás usando tu tiempo o cómo podrías mejorar tu productividad.")
    
    # Usar contenedor nativo de Streamlit con CSS personalizado
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-header">💬 Chat con tu asistente de productividad</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-body">', unsafe_allow_html=True)
        
        # Mostrar historial de chat
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                # Mensaje del usuario con alineación manual
                col1, col2 = st.columns([1, 3])
                with col2:
                    st.markdown(f"""
                    <div style="background-color: #ff4d00; color: white; padding: 0.75rem 1rem; 
                                border-radius: 18px 18px 5px 18px; margin-bottom: 0.5rem; 
                                margin-left: 2rem; text-align: left;">
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Mensaje del asistente con alineación manual
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background-color: white; color: #333; padding: 0.75rem 1rem; 
                                border: 1px solid #e1e5e9; border-radius: 18px 18px 18px 5px; 
                                margin-bottom: 0.5rem; margin-right: 2rem; text-align: left;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Input del chat
    openrouter_key = st.secrets["openrouter"]["key"]
    prompt = st.chat_input("💭 Escribe tu pregunta sobre productividad aquí...")
    
    if prompt:
        # Añadir mensaje del usuario al historial
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Procesar respuesta
        df_graph = st.session_state.get("df_graph")
        if df_graph is None and "df_original" in st.session_state:
            df_graph = st.session_state.df_original

        if df_graph is not None and not df_graph.empty:
            resumen_subact = df_graph.groupby("Subactivity")["Duration"].sum().sort_values(ascending=False).head(10)
            subactividad_str = "\n".join([f"{k}: {int(v/60)} minutos" for k, v in resumen_subact.items()])

            eisenhower_summary = (
                df_graph[df_graph["Eisenhower"].notna()]
                .groupby(["Subactivity", "Eisenhower"])["Duration"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            eisen_str = "\n".join([f"{sub} ({eis}): {int(dur/60)} min" for (sub, eis), dur in eisenhower_summary.items()])

            app_summary = (
                df_graph.groupby("App")["Duration"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            app_str = "\n".join([f"{app}: {int(dur/60)} min" for app, dur in app_summary.items()])

            improductivas = df_graph[df_graph["Eisenhower"] == "No urgente / No importante"]
            imp_duracion = improductivas["Duration"].sum()
            imp_str = f"Tiempo improductivo detectado: {int(imp_duracion/60)} min"
        else:
            resumen_str = "No hay datos disponibles para analizar."
        
        extended_prompt = f"""
        Tengo un registro de actividades laborales. A continuación, un resumen:

        ▶️ Subactividades más frecuentes:
        {subactividad_str}

        🧭 Clasificación Eisenhower:
        {eisen_str}

        💻 Aplicaciones más utilizadas:
        {app_str}

        ⚠️ {imp_str}

        Pregunta del usuario: {prompt}
        """
        
        messages_for_model = st.session_state.chat_history[:-1] + [{"role": "user", "content": extended_prompt}]
        
        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": messages_for_model
        }
        
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tu-aplicacion.streamlit.app"
        }
        
        # Mostrar indicador de carga
        with st.spinner("El asistente está pensando..."):
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    respuesta = response.json()["choices"][0]["message"]["content"]
                    st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
                    st.rerun()
                else:
                    error_msg = f"❌ Error al conectar con OpenRouter: {response.status_code}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    st.rerun()
            except Exception as e:
                error_msg = f"❌ Error de conexión: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.rerun()