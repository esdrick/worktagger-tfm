import pandas as pd
import streamlit as st
import logging
import requests
import re
import analysis as wt
from heuristic_rules import clasificar_por_heuristica
from heuristic_eisenhower import clasificar_eisenhower_por_heuristica
import clasificacion_core_act
import math  
from custom_heuristic_manager import CustomHeuristicManager, show_custom_heuristic_interface
from custom_eisenhower_manager import CustomEisenhowerManager, show_eisenhower_rules_interface

# Added missing imports for undefined references
from config.constants import EISEN_OPTIONS

# Import para tutorial
try:
    from tutorial.onboarding import integrate_tutorial_in_classification
except ImportError:
    def integrate_tutorial_in_classification():
        pass  # Fallback si no existe el módulo

def load_view_options():
    """Función helper para cargar opciones de vista"""
    try:
        import views
        return {
            "Time view": views.TimeView(),
            "Active window view": views.ActiveWindowView(),
            "Activity view": views.ActivityView(),
            "Work slot view": views.WorkSlotView()
        }
    except ImportError as e:
        st.error(f"Error loading views: {e}")
        return {}

def _show_classification_statistics():
    """Muestra estadísticas en tiempo real del progreso de clasificación - Versión con Streamlit nativo"""
    if "df_original" not in st.session_state or st.session_state.df_original.empty:
        return
    
    df = st.session_state.df_original
    # ✅ AÑADIR ESTA VALIDACIÓN
    if 'Eisenhower' not in df.columns:
        df['Eisenhower'] = None
        st.session_state.df_original = df

    total_activities = len(df)
    
    # Calcular estadísticas
    activities_classified = df['Activity'].notna().sum()
    subactivities_classified = df['Subactivity'].notna().sum()
    eisenhower_classified = df['Eisenhower'].notna().sum()
    cases_classified = df['Case'].notna().sum()
    
    # Calcular porcentajes
    activity_pct = (activities_classified / total_activities * 100) if total_activities > 0 else 0
    subactivity_pct = (subactivities_classified / total_activities * 100) if total_activities > 0 else 0
    eisenhower_pct = (eisenhower_classified / total_activities * 100) if total_activities > 0 else 0
    cases_pct = (cases_classified / total_activities * 100) if total_activities > 0 else 0
    
    # Determinar el estado general
    avg_completion = (activity_pct + subactivity_pct + eisenhower_pct + cases_pct) / 4
    
    if avg_completion >= 80:
        status_color = "#10B981"
        status_icon = "🎉"
        status_text = "Excellent progress"
    elif avg_completion >= 50:
        status_color = "#F59E0B"
        status_icon = "🚀"
        status_text = "Good progress"
    elif avg_completion >= 20:
        status_color = "#3B82F6"
        status_icon = "📈"
        status_text = "In progress"
    else:
        status_color = "#EF4444"
        status_icon = "🎯"
        status_text = "Getting started"
    
    # === HEADER ===
    st.markdown(f"""
    <div style="
        border: 2px solid {status_color};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <h3 style="color: {status_color}; margin: 0;">
            {status_icon} Classification Progress - {status_text}
        </h3>
        <p style="color: #666; margin: 4px 0 0 0; font-size: 14px;">
            {avg_completion:.0f}% average completion
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # === MÉTRICAS USANDO COLUMNAS NATIVAS DE STREAMLIT ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="ACTIVITIES", 
            value=f"{activity_pct:.0f}%",
            delta=f"{activities_classified}/{total_activities}"
        )
    
    with col2:
        st.metric(
            label="SUBACTIVITIES", 
            value=f"{subactivity_pct:.0f}%",
            delta=f"{subactivities_classified}/{total_activities}"
        )
    
    with col3:
        st.metric(
            label="EISENHOWER", 
            value=f"{eisenhower_pct:.0f}%",
            delta=f"{eisenhower_classified}/{total_activities}"
        )
    
    with col4:
        st.metric(
            label="CASES", 
            value=f"{cases_pct:.0f}%",
            delta=f"{cases_classified}/{total_activities}"
        )
    
    # Barra de progreso general
    st.progress(avg_completion / 100, text=f"Overall progress: {avg_completion:.0f}%")
    
    # Recomendaciones contextuales
    _show_contextual_recommendations(activity_pct, subactivity_pct, eisenhower_pct, cases_pct)

def _show_contextual_recommendations(activity_pct, subactivity_pct, eisenhower_pct, cases_pct):
    """Muestra recomendaciones contextuales basadas en el estado de clasificación"""
    
    recommendations = []
    
    if activity_pct < 20:
        recommendations.append("🎯 **Start with manual classification** to get familiar with the categories")
    elif activity_pct < 50:
        recommendations.append("⚡ **Use heuristic classification** to speed up the process")
    elif activity_pct < 80:
        recommendations.append("🤖 **Try AI** to classify the remaining activities")
    else:
        recommendations.append("🎉 **Excellent!** Activities are almost fully classified")
    
    if eisenhower_pct < 30 and activity_pct > 40:
        recommendations.append("📊 **Classify Eisenhower** for productivity analysis")
    
    if cases_pct < 20 and subactivity_pct > 50:
        recommendations.append("📋 **Add cases** to organize by projects")
    
    if all(pct > 70 for pct in [activity_pct, subactivity_pct, eisenhower_pct]):
        recommendations.append("🚀 **Go to the Dashboard** to see your detailed analyses")
    
    # Mostrar recomendaciones si las hay
    if recommendations:
        st.markdown("**💡 Recommended next steps:**")
        for rec in recommendations[:2]:  # Máximo 2 recomendaciones
            st.markdown(f"• {rec}")

def cases_classification():
    """Clasificación de casos"""
    dicc_core_color = st.session_state.get("dicc_core_color", {})
    
    def apply_label_to_selection(**kwargs):
        if "df_original" not in st.session_state or "filas_seleccionadas" not in st.session_state:
            st.warning("No rows selected or the dataset hasn't been loaded.")
            return

        df_original = st.session_state.df_original
        selected_ids = st.session_state.filas_seleccionadas['ID'].tolist()

        for key, value in kwargs.items():
            df_original.loc[df_original['ID'].isin(selected_ids), key] = value

        st.session_state.df_original = df_original

    def save_case_button(case_name):
        try:
            apply_label_to_selection(Case=case_name)
        except Exception as e:
            logging.exception(f"There was an error saving button {case_name}", exc_info=e)
            st.error("Error saving")

    def add_new_case():
        case_name = st.session_state.new_case_label
        if case_name is not None and case_name != "":
            try:
                apply_label_to_selection(Case=case_name)
                st.session_state.all_cases.add(case_name)
            except Exception as e:
                logging.exception(f"There was an error saving button {case_name}", exc_info=e)
                st.error("Error saving")

    with st.form(key='new_cases', clear_on_submit=True, border=False):
        [col1, col2] = st.columns([0.7, 0.3])
        with col1:
            st.text_input(label="Case label", label_visibility="collapsed", placeholder="Case label", key="new_case_label")
        with col2:
            st.form_submit_button(label="Assign", on_click=add_new_case)

    if len(st.session_state.all_cases) > 1:
        with st.container():
            st.markdown("### Case Labels")
            for case in st.session_state.all_cases:
                if case != "":
                    st.button(case, on_click=save_case_button, args=(case,), use_container_width=True)

def manual_classification_sidebar(dicc_core, dicc_subact, dicc_core_color, all_sub, apply_label_to_selection, change_color):
    """Clasificación manual de actividades y subactividades"""

    # Obtener gestor de reglas personalizadas
    heuristic_manager = st.session_state.get("heuristic_manager")
    
    # Combinar TODAS las subactividades (predefinidas + personalizadas)
    if heuristic_manager:
        all_sub_combined = list(all_sub)
        
        for activity in heuristic_manager.get_all_activities():
            for subact in heuristic_manager.get_all_subactivities(activity):
                sub_label = f"{subact} - {activity}"
                if sub_label not in all_sub_combined:
                    all_sub_combined.append(sub_label)
    else:
        all_sub_combined = all_sub

    # === FUNCIONES HELPER (sin cambios) ===
    def update_last_3_buttons(core, subact):
        if not "last_acts" in st.session_state:
            return
        dicc_aux = {"core_act": core, "subact": subact}
        if dicc_aux not in st.session_state.last_acts:
            if len(st.session_state.last_acts) > 2:
                st.session_state.last_acts.pop(0)
            st.session_state.last_acts.append(dicc_aux)

    def save_button(core_act, sub_act):
        try:
            apply_label_to_selection(Activity=core_act, Subactivity=sub_act)
        except Exception as e:
            logging.exception(f"There was an error saving button {core_act}, {sub_act}", exc_info=e)
            st.error("Error saving")

    def save_all_select():
        try:
            selected = st.session_state.all_select
            if not selected:
                return
            split_selection = selected.split(" - ")
            if len(split_selection) == 2:
                seleccion_subact = split_selection[0]
                seleccion_core_act = split_selection[1]
                apply_label_to_selection(Activity=seleccion_core_act, Subactivity=seleccion_subact)
                update_last_3_buttons(seleccion_core_act, seleccion_subact)
                st.session_state.all_select = None
        except Exception as e:
            logging.exception(f"There was an error saving all_select", exc_info=e)
            st.error("Error saving")

    def save_select(core):
        try:
            subact = st.session_state.get(core)
            if subact:
                apply_label_to_selection(Activity=core, Subactivity=subact)
                update_last_3_buttons(core, subact)
                st.session_state[core] = None
        except Exception as e:
            logging.exception(f"There was an error saving select {core}", exc_info=e)
            st.error("Error saving")

    # === LAST SUBACTIVITIES ===
    if len(st.session_state.last_acts) > 0:
        with st.container():
            st.markdown("### Last Subactivities")
            ll = [x for x in st.session_state.last_acts if x != ""]
            subacts = []
            for activity in ll:
                if activity['subact'] is not None and activity['subact'] not in subacts:
                    subacts.append(activity['subact'])
                    st.button(
                        activity['subact'], 
                        key=f'boton_{activity["subact"]}', 
                        on_click=save_button, 
                        args=(activity['core_act'], activity['subact']), 
                        use_container_width=True
                    )
                    if activity['core_act'] in dicc_core_color:
                        change_color('button', activity['subact'], 'black', dicc_core_color[activity['core_act']])

    # === BUSCADOR UNIVERSAL (incluye predefinidas + personalizadas) ===
    st.selectbox(
        "Search all subactivities", 
        key="all_select", 
        options=all_sub_combined,  # Ya incluye personalizadas
        index=None, 
        placeholder="Search all subactivities", 
        label_visibility='collapsed', 
        on_change=save_all_select
    )

    # === SOLO CATEGORÍAS PREDEFINIDAS (sin sección custom separada) ===
    for category in dicc_core.keys():
        with st.container():
            st.markdown(f"### {category}")
            for activity in dicc_core[category]:
                core_act = activity['core_activity']
                
                # Combinar subactividades predefinidas + personalizadas para esta actividad
                combined_subacts = list(dicc_subact[core_act])
                
                if heuristic_manager:
                    custom_subs = heuristic_manager.get_all_subactivities(core_act)
                    for sub in custom_subs:
                        if sub not in combined_subacts:
                            combined_subacts.append(sub)
                
                st.selectbox(
                    core_act,
                    key=core_act,
                    options=combined_subacts,
                    index=None,
                    placeholder=core_act,
                    label_visibility='collapsed',
                    on_change=save_select,
                    args=(core_act,)
                )
    
    # ELIMINADO: Sección "Custom Activities" completa

def automated_classification(view_options, mensaje_container):
    """Clasificación automática con IA"""
    def run_auto_classify():
        # AGREGAR ESTAS VALIDACIONES AL INICIO
        openai_key = st.session_state.get('openai_key', '').strip()
        if not openai_key:
            st.error("⚠️ **API Key required**: Please enter your OpenAI API key to continue.")
            return
            
        # VALIDAR SELECCIÓN DE FILAS SI ES NECESARIO
        select_class = st.session_state.auto_type
        if select_class == "Selected rows":
            if 'filas_seleccionadas' not in st.session_state:
                st.error("⚠️ **No selection**: Select some rows in the main table first.")
                return
            selected_rows = st.session_state.filas_seleccionadas['ID'].tolist()
            if not selected_rows:
                st.error("⚠️ **No rows selected**: Select some rows in the main table.")
                return
        
        openai_org = st.session_state.openai_org
        all = st.session_state.df_original

        if select_class=="Selected date":
            a_datetime = st.session_state.a_datetime
            next_day = st.session_state.next_day
            filter_app = (all['Begin'] >= a_datetime) & (all['Begin'] < next_day)
            to_classify = all[filter_app]
        elif select_class == "Selected rows":
            selected_rows = st.session_state.filas_seleccionadas['ID'].tolist()
            index = [x - 1 for x in selected_rows]
            to_classify = all.iloc[index]
            filter_app = all['ID'].isin(selected_rows)
        else:
            to_classify = all
            filter_app = None

        mensaje_container.write(f"Classifying with GPT {len(to_classify)} elements (it might take a while)...")
        classification = clasificacion_core_act.classify(to_classify, openai_key, openai_org)
        st.session_state.undo_df = all.copy()
        
        if filter_app is not None:
            mask = all.loc[filter_app, 'Subactivity'].isin([None, "", "Unclassified"])
            all.loc[filter_app, 'Activity'] = all.loc[filter_app, 'Activity'].where(~mask, classification)
            all.loc[filter_app, 'Subactivity'] = all.loc[filter_app, 'Subactivity'].where(~mask, ["Unspecified " + c for c in classification])
        else:
            mask = all['Subactivity'].isin([None, "", "Unclassified"])
            all['Activity'] = all['Activity'].where(~mask, classification)
            all['Subactivity'] = all['Subactivity'].where(~mask, ["Unspecified " + c for c in classification])

    with st.form(key='auto_labeling'):
        st.text_input("Set OpenAI key", type="password", key='openai_key')
        st.text_input("Set OpenAI org", type="password", key='openai_org')

        if view_options and st.session_state.get('view_type') and view_options[st.session_state.view_type].has_time_blocks:
            options = ["All", "Selected date", "Selected rows"]
            index = 1
        else:
            options = ["All"]
            index = 0

        st.selectbox("Choose what data you want to classify", options, index=index, key='auto_type')
        st.form_submit_button("Start classification", on_click=run_auto_classify)

def heuristic_prediction():
    """Clasificación heurística basada en apps y títulos - CON REGLAS PERSONALIZADAS"""
    
    # Inicializar gestor de reglas
    if "heuristic_manager" not in st.session_state:
        st.session_state.heuristic_manager = CustomHeuristicManager()
    
    def run_prediction():
        heuristic_manager = st.session_state.heuristic_manager
        all = st.session_state.df_original
        st.session_state.undo_df = all.copy()

        tipo_datos = st.session_state.heuristic_data_type
        if tipo_datos == "Selected rows":
            if 'filas_seleccionadas' not in st.session_state:
                st.error("No rows selected")
                return
            selected_ids = st.session_state.filas_seleccionadas['ID'].tolist()
            to_classify = all[all["ID"].isin(selected_ids)]
        else:
            to_classify = all

        # USAR EL GESTOR DE REGLAS PERSONALIZADAS
        results = to_classify.apply(
            lambda row: heuristic_manager.match_heuristic_rule(row['App'], row['Merged_titles']), 
            axis=1, 
            result_type='expand'
        )
        
        if results.empty:
            st.warning("There are no rows to label with heuristics.")
            return
            
        all.loc[to_classify.index, 'PredictedSubactivity'] = results[0]
        all.loc[to_classify.index, 'PredictedActivity'] = results[1]

        mask_sub = all['Subactivity'].isna() | all['Subactivity'].str.startswith("Unspecified")
        mask_sub = mask_sub & all['PredictedSubactivity'].notna()
        all.loc[mask_sub, 'Subactivity'] = all.loc[mask_sub, 'PredictedSubactivity']

        mask_act = all['Activity'].isna() | all['Activity'].str.startswith("Unspecified")
        mask_act = mask_act & all['PredictedActivity'].notna()
        all.loc[mask_act, 'Activity'] = all.loc[mask_act, 'PredictedActivity']

        st.success("Heuristic prediction successfully applied with custom rules.")
    
    with st.form(key='heuristic_prediction_form', clear_on_submit=True):
        st.markdown("This tool predicts a base category for each activity based on the app and window title using **your custom rules + default rules**.")
        st.selectbox("Choose what data you want to classify", ["All", "Selected rows"], key="heuristic_data_type", index=0)

        st.form_submit_button("Predict categories", on_click=run_prediction)

def custom_rules_management_section():
    """Sección para gestionar reglas personalizadas"""
    
    if "heuristic_manager" not in st.session_state:
        st.session_state.heuristic_manager = CustomHeuristicManager()
    
    # Inicializar estado si no existe
    if 'show_rule_management' not in st.session_state:
        st.session_state.show_rule_management = False
    
    # Botón toggle
    if st.button(
        "⚙️ Manage Custom Heuristic Rules", 
        use_container_width=True, 
        type="secondary"
    ):
        st.session_state.show_rule_management = not st.session_state.show_rule_management
    
    # Mostrar interfaz
    if st.session_state.show_rule_management:
        st.markdown("### ⚙️ Custom Heuristic Rules Manager")
        show_custom_heuristic_interface(st.session_state.heuristic_manager)
    
    # Estadísticas
    heuristic_manager = st.session_state.heuristic_manager
    active_rules = len(heuristic_manager.get_active_rules())
    total_custom = len(heuristic_manager.user_rules)
    
    if total_custom > 0:
        st.info(f"📊 Custom rules: {active_rules} active out of {total_custom} total")

def heuristic_classification():
    """Expansión heurística de etiquetas"""
    def run_expand_labels():
        interval = st.session_state.heuristic_interval
        all = st.session_state.df_original
        st.session_state.undo_df = all.copy()

        temporal_slots = wt.find_temporal_slots(all, inactivity_threshold=pd.Timedelta(f'{interval}s'))
        case_expand = wt.expand_slots(all, temporal_slots, column='Case')
        all['Case'] = case_expand

    with st.form(key='heuristic_labeling'):
        st.slider("Interval size (in seconds)", min_value=0, max_value=300, key='heuristic_interval')
        st.form_submit_button("Expand case labels", on_click=run_expand_labels)

def manual_eisenhower_classification(apply_label_to_selection):
    """Clasificación manual de Eisenhower"""
    def save_eisen():
        try:
            selected_q = st.session_state["eisen_select"]
            if selected_q:
                apply_label_to_selection(Eisenhower=selected_q)
                st.session_state["eisen_select"] = None
        except Exception as e:
            logging.exception("There was an error saving Eisenhower label", exc_info=e)
            st.error("Error saving")

    st.markdown("### Eisenhower Quadrant")
    eisen_options = EISEN_OPTIONS
    st.selectbox(
        "Eisenhower quadrant",
        key="eisen_select",
        options=eisen_options,
        index=None,
        placeholder="Select quadrant",
        label_visibility="collapsed",
        on_change=save_eisen
    )

def classify_eisenhower_auto():
    """Clasificación automática de Eisenhower con GPT - VERSIÓN SIN EMOJIS"""
    try:
        # 1. VALIDAR API KEY
        openai_key = st.session_state.get('openai_key_eisen', '').strip()
        if not openai_key:
            st.error("API Key required: Enter your API key to continue.")
            return
            
        # 2. OBTENER DATOS
        openai_org = st.session_state.get('openai_org_eisen', '').strip()
        tipo_datos = st.session_state.get('eisen_data_type', 'All')
        df = st.session_state.df_original
        
        if tipo_datos == "Selected rows":
            if 'filas_seleccionadas' not in st.session_state:
                st.error("No selection: Select some rows in the main table first.")
                return
            selected_ids = st.session_state.filas_seleccionadas['ID'].tolist()
            if not selected_ids:
                st.error("No rows selected: Select some rows in the main table.")
                return
            to_classify = df[df["ID"].isin(selected_ids) & df['Subactivity'].notna()]
        else:
            to_classify = df[df['Subactivity'].notna() & df['Eisenhower'].isna()]

        if to_classify.empty:
            st.warning("No rows available for classification.")
            return

        # 3. FUNCIÓN PARA LIMPIAR TEXTO
        def clean_text(text):
            """Limpia el texto de caracteres problemáticos"""
            if pd.isna(text) or text is None:
                return ""
            # Convertir a string y limpiar emojis/caracteres especiales
            text = str(text)
            # Remover caracteres no ASCII
            text = text.encode('ascii', errors='ignore').decode('ascii')
            return text.strip()

        # 4. CONFIGURAR API
        usar_openai = st.session_state.get("usar_openai", False)
        
        if usar_openai:
            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json; charset=utf-8"
            }
            if openai_org:
                headers["OpenAI-Organization"] = openai_org
            model = "gpt-3.5-turbo"
        else:
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json; charset=utf-8",
                "HTTP-Referer": "https://your-app.streamlit.app"
            }
            model = "mistralai/mistral-7b-instruct:free"

        # 5. PROCESAR EN LOTES
        st.session_state.undo_df = df.copy()
        resultados_finales = []
        batch_size = 3
        total_batches = math.ceil(len(to_classify) / batch_size)

        progress_bar = st.progress(0, text="Starting classification...")

        for batch_idx, start in enumerate(range(0, len(to_classify), batch_size)):
            batch = to_classify.iloc[start:start + batch_size]
            
            progress = (batch_idx + 1) / total_batches
            progress_bar.progress(progress, text=f"Processing batch {batch_idx + 1}/{total_batches}")
            
            # Preparar mensajes CON TEXTO LIMPIO
            tasks_text = []
            for i, (_, row) in enumerate(batch.iterrows(), 1):
                # LIMPIAR TEXTO AQUÍ
                desc = clean_text(row['Subactivity']) or clean_text(row['Merged_titles'])
                duration = row['Duration'] / 60
                tasks_text.append(f"{i}. {desc} ({duration:.1f} min)")

            mensajes = [
                {
                    "role": "system",
                    "content": "You are a productivity expert. Classify each task into exactly one Eisenhower quadrant. Respond with only the quadrant number (I, II, III, or IV) for each task."
                },
                {
                    "role": "user", 
                    "content": "Classify these tasks into Eisenhower quadrants (I, II, III, IV):\n\n" + 
                              "\n".join(tasks_text) + 
                              "\n\nRespond with only: 1. I\n2. II\n3. III\n(etc.)"
                }
            ]

            # Realizar petición
            payload = {
                "model": model,
                "messages": mensajes,
                "temperature": 0.1,
                "max_tokens": 100
            }

            try:
                # ENVIAR CON ENCODING EXPLÍCITO
                response = requests.post(
                    api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    
                    # Parsear respuesta
                    lines = content.strip().split('\n')
                    batch_results = []
                    
                    for line in lines:
                        match = re.search(r'\b(I|II|III|IV)\b', line)
                        if match:
                            quadrant = match.group(1)
                            full_label = {
                                'I': 'I: Urgent & Important',
                                'II': 'II: Not urgent but Important', 
                                'III': 'III: Urgent but Not important',
                                'IV': 'IV: Not urgent & Not important'
                            }.get(quadrant)
                            batch_results.append(full_label)
                    
                    # Completar batch si es necesario
                    while len(batch_results) < len(batch):
                        batch_results.append('IV: Not urgent & Not important')
                    
                    resultados_finales.extend(batch_results[:len(batch)])
                    
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    resultados_finales.extend([None] * len(batch))
                    
            except Exception as e:
                st.error(f"Request error: {str(e)}")
                resultados_finales.extend([None] * len(batch))

        # 6. APLICAR RESULTADOS
        progress_bar.progress(1.0, text="Applying results...")
        
        if len(resultados_finales) == len(to_classify):
            df.loc[to_classify.index, "Eisenhower"] = resultados_finales
            
            successful_count = sum(1 for r in resultados_finales if r is not None)
            st.success(f"Successfully classified {successful_count}/{len(to_classify)} activities")
            
            if successful_count < len(to_classify):
                st.warning(f"{len(to_classify) - successful_count} activities could not be classified")
        else:
            st.error("Classification failed due to response mismatch")
            
        progress_bar.empty()

    except Exception as e:
        logging.exception("Error in Eisenhower classification", exc_info=e)
        st.error(f"Unexpected error: {str(e)}")

def classify_eisenhower_heuristic():
    """Clasificación heurística de Eisenhower"""
    try:
        df = st.session_state.df_original
        st.session_state.undo_df = df.copy()

        tipo_datos = st.session_state.eisen_data_type_heuristic
        if tipo_datos == "All":
            to_classify = df[df['Subactivity'].notna() & df['Eisenhower'].isna()]
        elif tipo_datos == "Selected rows":
            selected_ids = st.session_state.filas_seleccionadas['ID'].tolist()
            to_classify = df[df["ID"].isin(selected_ids) & df['Subactivity'].notna()]
        else:
            st.warning("Invalid data type.")
            return

        if to_classify.empty:
            st.warning("There are no rows that can be labeled heuristically.")
            return

        resultados = to_classify['Subactivity'].apply(clasificar_eisenhower_por_heuristica)
        df.loc[to_classify.index, "Eisenhower"] = resultados
        st.success("Heuristic classification completed.")
    except Exception as e:
        logging.exception("Error en clasificación heurística Eisenhower", exc_info=e)
        st.error("Unexpected error during heuristic classification.")


def eisenhower_heuristic_sidebar():
    """Sidebar para clasificación heurística de Eisenhower"""
    
    # Inicializar gestor
    if "eisenhower_manager" not in st.session_state:
        st.session_state.eisenhower_manager = CustomEisenhowerManager()
    
    with st.form(key="heuristic_eisenhower_form", clear_on_submit=True):
        st.markdown("Apply heuristic classification using **your custom rules + default rules**")
        
        if "df_original" not in st.session_state:
            st.warning("Upload a file to enable this option.")
            return
        
        st.selectbox(
            "Choose what data you want to classify", 
            ["All", "Selected rows"], 
            key="eisen_data_type_heuristic", 
            index=0
        )
        
        st.form_submit_button("🏷️ Classify heuristically", on_click=classify_eisenhower_heuristic_custom)
    
    # Botón para gestionar reglas personalizadas
    if st.button("⚙️ Manage Eisenhower Rules", use_container_width=True, type="secondary"):
        st.session_state.show_eisenhower_rules = not st.session_state.get("show_eisenhower_rules", False)
    
    if st.session_state.get("show_eisenhower_rules", False):
        show_eisenhower_rules_interface(st.session_state.eisenhower_manager)

def classify_eisenhower_heuristic_custom():
    """Clasificación heurística CON REGLAS PERSONALIZADAS"""
    try:
        eisenhower_manager = st.session_state.eisenhower_manager
        df = st.session_state.df_original
        st.session_state.undo_df = df.copy()

        tipo_datos = st.session_state.eisen_data_type_heuristic
        
        if tipo_datos == "All":
            to_classify = df[df['Subactivity'].notna() & df['Eisenhower'].isna()]
        elif tipo_datos == "Selected rows":
            selected_ids = st.session_state.filas_seleccionadas['ID'].tolist()
            to_classify = df[df["ID"].isin(selected_ids) & df['Subactivity'].notna()]
        else:
            st.warning("Invalid data type.")
            return

        if to_classify.empty:
            st.warning("No rows available for heuristic classification.")
            return

        # USAR EL GESTOR DE REGLAS PERSONALIZADAS
        resultados = to_classify['Subactivity'].apply(eisenhower_manager.classify_eisenhower)
        df.loc[to_classify.index, "Eisenhower"] = resultados
        
        st.success("Heuristic classification completed with custom rules.")
        
    except Exception as e:
        logging.exception("Error in Eisenhower heuristic classification", exc_info=e)
        st.error("Unexpected error during classification.")

def display_improved_label_palette(selected_df, dicc_core, dicc_subact, dicc_core_color, all_sub, apply_label_to_selection, change_color, mensaje_container):
    """Paleta de etiquetas mejorada con el orden específico solicitado"""
    
    if len(selected_df) == 0:
        st.title("Label Cases")
        st.warning("No data to label")
        st.title("Label Activities")
        st.warning("No data to label")
        return
    
    try:
        # === TUTORIAL Y ESTADÍSTICAS ===
        integrate_tutorial_in_classification()
        _show_classification_statistics()
        
        # === 1. ETIQUETADO DE CASOS (PRIMERO) ===
        st.title("📋 Label Cases")
        st.markdown("Organize your activities into **cases** or **projects** for better tracking.")
        
        with st.expander("📋 **Case Classification** - Project organization", expanded=False):
            st.markdown("Assign case labels to group activities related by project or context")
            cases_classification()
        
        # === 2. ETIQUETADO DE VENTANAS/ACTIVIDADES ===
        st.title("🏷️ Label Activities")
        st.markdown("**Available tools:**")
        
        # 2.1 Manual
        if dicc_core and dicc_subact:
            with st.expander("✋ **Manual Classification** - Full control", expanded=False):
                st.markdown("Select activities and assign categories manually")
                manual_classification_sidebar(dicc_core, dicc_subact, dicc_core_color, all_sub, apply_label_to_selection, change_color)
        else:
            st.error("⚠️ Activity data not loaded properly. Please refresh the page.")
        
        # 2.2 Automático
        if "df_original" in st.session_state:
            view_options = load_view_options()
            with st.expander("🤖 **Automated Classification** - AI", expanded=False):
                st.markdown("Use AI models for smart classification")
                automated_classification(view_options, mensaje_container)
        
        # 2.3 Heurístico 1 - Predicción por App
        with st.expander("🧠 **Heuristic Prediction** - Apps and titles", expanded=False):
            st.markdown("Classify based on app names and window titles")
            heuristic_prediction()
            custom_rules_management_section()
        
        # 2.4 Heurístico 2 - Expansión de etiquetas
        with st.expander("🔗 **Label Expansion** - Smart fill", expanded=False):
            st.markdown("Extend case labels to nearby activities")
            heuristic_classification()
        
        # === 4. MATRIZ EISENHOWER (AL FINAL) ===
        st.title("🧭 Eisenhower Matrix Classification")
        st.markdown("**Classify your activities by urgency and importance:**")
        
        # 4.1 Manual Eisenhower
        with st.expander("✋ **Manual Eisenhower** - Direct control", expanded=False):
            st.markdown("Select the quadrant manually for each activity")
            manual_eisenhower_classification(apply_label_to_selection)
        
        # 4.2 Automático Eisenhower
        with st.expander("🤖 **Automated Eisenhower** - AI", expanded=False):
            st.markdown("Automatic classification using GPT")
            with st.form(key="eisen_gpt_classification_form", clear_on_submit=True):
                st.selectbox("Choose what data you want to classify", ["All", "Selected rows"], key="eisen_data_type", index=0)
                st.checkbox("Use OpenAI (instead of OpenRouter)", key="usar_openai")
                st.text_input("🔐 API Key", type="password", key="openai_key_eisen")
                st.text_input("🔑 Org (optional)", type="password", key="openai_org_eisen")
                st.form_submit_button("🏷️ Classify automatically", on_click=classify_eisenhower_auto)
        
        # 4.3 Heurístico Eisenhower
        with st.expander("🧠 **Heuristic Eisenhower** - Automatic", expanded=False):
            st.markdown("Heuristic classification based on patterns")
            eisenhower_heuristic_sidebar()
        
    except Exception as e:
        logging.exception(f"There was an error while displaying the label palette", exc_info=e)
        st.error("There was an error processing the request. Try again")