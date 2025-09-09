import streamlit as st

def render_menu_selector():
    # Ensure it's marked as loaded correctly after assigning df_original
    if "navbar_selection" not in st.session_state:
        st.session_state["navbar_selection"] = "📋 Main Screen"
    
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
            /* Estilo para botones deshabilitados */
            button[disabled] {
                opacity: 0.5 !important;
                cursor: not-allowed !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Navigation options
        nav_options = {
            "📋 Main Screen": "main",
            "📊 Activities Dashboard": "dashboard",
            "🧭 Eisenhower Matrix – View details by subactivity": "matrix",
            "🤖 Productivity Assistant (beta)": "assistant"
        }

        # Create columns for the buttons
        nav_cols = st.columns(len(nav_options))
        
        # Check if data is loaded
        has_data = "df_original" in st.session_state
        
        # Get current selection
        current_selection = st.session_state.get("navbar_selection", "📋 Main Screen")

        # Show buttons with active state detection
        for idx, (label, value) in enumerate(nav_options.items()):
            with nav_cols[idx]:
                is_active = (label == current_selection)
                
                # Disable buttons if no data loaded (except Main Screen)
                is_disabled = not has_data and value != "main"
                
                if st.button(
                    label, 
                    key=label,
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    disabled=is_disabled  # 👈 ÚNICA MEJORA SUGERIDA
                ):
                    st.session_state["navbar_selection"] = label
                    st.rerun()
    
    return st.session_state.get("navbar_selection", "📋 Main Screen")