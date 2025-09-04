
import streamlit as st

def render_menu_selector():
    # Asegura que se marque como cargado correctamente después de asignar df_original
    if "navbar_selection" not in st.session_state:
        st.session_state["navbar_selection"] = "📋 Pantalla principal"
    with st.container():
        # Estilos solo para navbar
        st.markdown("""
            <style>
            div[data-testid="column"] button {
                height: 100px !important;
                white-space: normal !important;
                font-size: 16px !important;
                font-weight: bold !important;
                text-align: center !important;
                padding: 10px !important;
            }
            /* Color naranja correcto para botón activo */
            button[kind="primary"] {
                background-color: #FA3E25 !important;
                border-color: #FA3E25 !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Opciones de navegación
        nav_options = {
            "📋 Pantalla principal": "main",
            "📊 Dashboard de Actividades": "dashboard",
            "🧭 Matriz de Eisenhower – Ver detalle por subactividad": "matrix",
            "🤖 Asistente de productividad (beta)": "assistant"
        }

        # Crear columnas para los botones
        nav_cols = st.columns(len(nav_options))
        
        # Obtener selección actual
        current_selection = st.session_state.get("navbar_selection", "📋 Pantalla principal")

        # Mostrar botones con detección de estado activo
        for idx, (label, value) in enumerate(nav_options.items()):
            with nav_cols[idx]:
                is_active = (label == current_selection)
                if st.button(
                    label, 
                    key=label, 
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state["navbar_selection"] = label
                    st.rerun() 

    return st.session_state.get("navbar_selection", "📋 Pantalla principal")