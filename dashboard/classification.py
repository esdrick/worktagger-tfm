import pandas as pd
import streamlit as st
import logging
import requests
import re
import analysis as wt
from heuristic_rules import clasificar_por_heuristica
from heuristic_eisenhower import clasificar_eisenhower_por_heuristica
import clasificacion_core_act
import streamlit as st

# Added missing imports for undefined references
from config.constants import EISEN_OPTIONS

def manual_classification_sidebar(dicc_core, dicc_subact, dicc_core_color, all_sub, apply_label_to_selection, change_color):

    def update_last_3_buttons(core, subact):
        if not "last_acts" in st.session_state:
            return

        dicc_aux = {"core_act": core, "subact":subact}
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
            split_selection = selected.split(" - ")

            if len(split_selection) == 2:
                seleccion_core_act = split_selection[1]
                seleccion_subact = split_selection[0]

                apply_label_to_selection(Activity=seleccion_core_act, Subactivity=seleccion_subact)
                update_last_3_buttons(seleccion_core_act, seleccion_subact)
                st.session_state.all_select = None

        except Exception as e:
            logging.exception(f"There was an error saving all_select", exc_info=e)
            st.error("Error saving")

    def save_select(core):
        try:
            subact = st.session_state[core]
            apply_label_to_selection(Activity=core, Subactivity=subact)
            update_last_3_buttons(core, subact)
            st.session_state[core] = None
        except Exception as e:
            logging.exception(f"There was an error saving select {core}", exc_info=e)
            st.error("Error saving")

    if len(st.session_state.last_acts) > 0:
        with st.container():
            st.markdown("###  Last Subactivities")
            ll = [x for x in st.session_state.last_acts if x != ""]
            subacts = []
            for activity in ll:
                if activity['subact'] is not None and activity['subact'] not in subacts:
                    subacts.append(activity['subact'])
                    st.button(activity['subact'], key=f'boton_{activity["subact"]}', on_click=save_button, args=(activity['core_act'], activity['subact']), use_container_width=True)
                    change_color('button', activity['subact'], 'black', dicc_core_color[activity['core_act']])

    st.selectbox("Search all subactivities", key="all_select", options = all_sub, index=None, placeholder="Search all subactivities", label_visibility='collapsed', on_change=save_all_select)

    for category in dicc_core.keys():
        with st.container():
            st.markdown(f"### {category}")
            for activity in dicc_core[category]:
                # Define core_act and color just before selectbox
                core_act = activity['core_activity']
                color = dicc_core_color[core_act]
                st.selectbox(
                    core_act,
                    key=core_act,
                    options=dicc_subact[core_act],
                    index=None,
                    placeholder=core_act,
                    label_visibility='collapsed',
                    on_change=save_select,
                    args=(core_act,)
                )

    # --- Eisenhower quadrant manual labelling ---------------------------
    def save_eisen():
        try:
            selected_q = st.session_state["eisen_select"]
            if selected_q:
                apply_label_to_selection(Eisenhower=selected_q)
                # clear selector
                st.session_state["eisen_select"] = None
        except Exception as e:
            logging.exception("There was an error saving Eisenhower label", exc_info=e)
            st.error("Error saving")

    st.markdown("## Eisenhower Quadrant")
    eisen_options = EISEN_OPTIONS
    st.selectbox(
        "Eisenhower quadrant",
        key="eisen_select",
        options=eisen_options,
        index=None,
        placeholder="Selecciona cuadrante",
        label_visibility="collapsed",
        on_change=save_eisen
    )

    # --- Etiquetado automático de Eisenhower con GPT (debajo del manual) ---
    with st.expander("🧠 Clasificación automática con GPT"):
        with st.form(key="eisen_gpt_classification_form", clear_on_submit=True):
            st.selectbox("Choose what data you want to classify", ["All", "Selected rows"], key="eisen_data_type", index=0)
            st.checkbox("Usar OpenAI (en lugar de OpenRouter)", key="usar_openai")
            st.text_input("🔐 OpenRouter Key", type="password", key="openai_key_eisen")
            st.text_input("🔑 OpenRouter Org (opcional)", type="password", key="openai_org_eisen")
            st.form_submit_button("🏷️ Etiquetar automáticamente", on_click=classify_eisenhower_auto)

# --- Eisenhower automatic classification function (moved to top-level) ---
def classify_eisenhower_auto():
    try:
        openai_key = st.session_state.openai_key_eisen
        openai_org = st.session_state.openai_org_eisen

        df = st.session_state.df_original

        # Nuevo selector de tipo de datos a clasificar
        tipo_datos = st.session_state.eisen_data_type
        if tipo_datos == "All":
            to_classify = df[df['Subactivity'].notna() & df['Eisenhower'].isna()]
        elif tipo_datos == "Selected rows":
            selected_ids = st.session_state.filas_seleccionadas['ID'].tolist()
            to_classify = df[df["ID"].isin(selected_ids) & df['Subactivity'].notna()]
        else:
            st.warning("Tipo de datos no válido.")
            return

        if to_classify.empty:
            st.warning("No hay filas que puedan ser clasificadas automáticamente.")
            return

        st.session_state.undo_df = df.copy()

        EISEN_OPTIONS = [
            "I: Urgente & Importante",
            "II: No urgente pero Importante",
            "III: Urgente pero No importante",
            "IV: No urgente & No importante"
        ]

        # Bandera de control para elegir OpenAI u OpenRouter
        usar_openai = st.session_state.get("usar_openai", False)

        resultados_finales = []

        batch_size = 10
        import re
        for start in range(0, len(to_classify), batch_size):
            batch = to_classify.iloc[start:start + batch_size]
            mensajes = [
                {
                    "role": "system",
                    "content": "Eres un asistente experto en productividad. Etiqueta cada actividad en un cuadrante de Eisenhower: 'I: Urgente & Importante', 'II: No urgente pero Importante', 'III: Urgente pero No importante', 'IV: No urgente & No importante'."
                }
            ]

            inputs = []
            for _, row in batch.iterrows():
                desc = row['Subactivity'] or row['Merged_titles']
                duracion = row['Duration'] / 60
                inputs.append(f"Tarea: {desc}\nDuración: {duracion:.1f} minutos")

            prompts = "\n\n".join([f"{i+1}. {txt}" for i, txt in enumerate(inputs)])
            mensajes.append({
              "role": "user",
              "content": f"Clasifica las siguientes tareas una por una. Devuelve exactamente una etiqueta por cada una, usando únicamente:\n\n"
                         "I: Urgente & Importante\n"
                         "II: No urgente pero Importante\n"
                         "III: Urgente pero No importante\n"
                         "IV: No urgente & No importante\n\n"
                         f"Ejemplo de formato:\n1. I: Urgente & Importante\n2. II: No urgente pero Importante\n\nTareas:\n{prompts}"
            })

            # BLOQUE condicional para OpenAI/OpenRouter
            if usar_openai:
                api_url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                }
                model = "gpt-3.5-turbo"
            else:
                api_url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://tu-aplicacion.streamlit.app"
                }
                model = "mistralai/mistral-7b-instruct:free"

            payload = {
                "model": model,
                "messages": mensajes,
                "temperature": 0.2
            }

            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code == 200:
                texto = response.json()["choices"][0]["message"]["content"]

                # VERIFICACIÓN DE RESPUESTA VACÍA
                if not texto.strip():
                    st.warning(f"El modelo no devolvió ninguna respuesta útil en el lote {start//batch_size + 1}.")
                    resultados_finales.extend([None] * len(batch))
                    continue

                # Extrae las etiquetas línea por línea, validando exactamente una etiqueta por línea y acumulando en orden
                lineas = texto.strip().splitlines()
                etiquetas_extraidas = []
                for linea in lineas:
                    match = re.match(r"\s*\d+\.\s*(I|II|III|IV)\s*:", linea.strip())
                    if match:
                        etiqueta = match.group(1)
                        for op in EISEN_OPTIONS:
                            if etiqueta in op:
                                etiquetas_extraidas.append(op)
                                break
                        else:
                            etiquetas_extraidas.append(None)

                # Agregar las etiquetas extraídas al resultado final si la cantidad es correcta
                if len(etiquetas_extraidas) == len(batch):
                    resultados_finales.extend(etiquetas_extraidas)
                else:
                    st.error(f"⚠️ Error al procesar lote {start//batch_size + 1}: se esperaban {len(batch)} etiquetas, pero se obtuvieron {len(etiquetas_extraidas)}.")
                    resultados_finales.extend([None] * len(batch))
            else:
                resultados_finales.extend([None] * len(batch))

                if response.status_code == 401:
                    st.error("❌ La clave API es inválida o ha expirado. Verifica tu configuración.")
                elif response.status_code == 429:
                    st.warning("🚫 Has alcanzado el límite de uso de la API. Revisa tu plan en OpenAI o intenta más tarde.")
                elif response.status_code == 403:
                    st.warning("🔒 Acceso denegado. Puede que tu cuenta no tenga permisos suficientes.")
                else:
                    st.error(f"❌ Error inesperado (código {response.status_code}). Intenta de nuevo o revisa la conexión.")

        # --- Validación antes de asignar etiquetas ---
        if len(resultados_finales) != len(to_classify):
            if len(resultados_finales) == 1:
                # Si solo una etiqueta fue devuelta, la aplicamos a todas las tareas
                resultados_finales = resultados_finales * len(to_classify)
                st.info(f"Se aplicó una única etiqueta '{resultados_finales[0]}' a todas las {len(to_classify)} tareas.")
            else:
                st.error(f"Número de etiquetas no coincide: se esperaban {len(to_classify)}, se obtuvieron {len(resultados_finales)}.")
                return

        # Nuevo bloque: solo sobrescribe filas cuyo valor previo es nulo
        resultados_series = pd.Series(resultados_finales, index=to_classify.index)
        mask = df.loc[to_classify.index, "Eisenhower"].isna()
        df.loc[to_classify.index[mask], "Eisenhower"] = resultados_series.loc[mask]
        # Solo mostramos mensaje de éxito si no hubo ningún None en los resultados
        if all(res is not None for res in resultados_finales):
            st.success("Clasificación completada.")
        else:
            st.warning("Clasificación incompleta. Algunas tareas no pudieron ser etiquetadas.")
        # Agregado: advertencia si alguna fila no se pudo sobrescribir
        if any(r is None for r in resultados_finales):
            st.info("Las filas con etiquetas previas no fueron sobrescritas. Asegúrate de tener suficiente cuota o clave válida.")

    except Exception as e:
        logging.exception("Error clasificando Eisenhower automático", exc_info=e)
        st.error("Error inesperado durante clasificación automática.")

def heuristic_prediction():
    def run_prediction():
        all = st.session_state.df_original
        st.session_state.undo_df = all.copy()

        tipo_datos = st.session_state.heuristic_data_type
        if tipo_datos == "Selected rows":
            selected_ids = st.session_state.filas_seleccionadas['ID'].tolist()
            to_classify = all[all["ID"].isin(selected_ids)]
        else:
            to_classify = all

        results = to_classify.apply(lambda row: clasificar_por_heuristica(row['App'], row['Merged_titles']), axis=1, result_type='expand')
        if results.empty:
            st.warning("No hay filas para etiquetar con heurística.")
            return
        all.loc[to_classify.index, 'PredictedSubactivity'] = results[0]
        all.loc[to_classify.index, 'PredictedActivity'] = results[1]

        # Completa Subactivity solo si está vacía o Unspecified
        mask_sub = all['Subactivity'].isna() | all['Subactivity'].str.startswith("Unspecified")
        mask_sub = mask_sub & all['PredictedSubactivity'].notna()
        all.loc[mask_sub, 'Subactivity'] = all.loc[mask_sub, 'PredictedSubactivity']

        # Completa Activity solo si está vacía o Unspecified
        mask_act = all['Activity'].isna() | all['Activity'].str.startswith("Unspecified")
        mask_act = mask_act & all['PredictedActivity'].notna()
        all.loc[mask_act, 'Activity'] = all.loc[mask_act, 'PredictedActivity']

        st.success("Heuristic prediction successfully applied.")

    with st.expander("🧠 Heuristic Prediction based on App and Title"):
        with st.form(key='heuristic_prediction_form', clear_on_submit=True):
            st.markdown("This tool predicts a base category for each activity based on the app and window title.")
            st.selectbox("Choose what data you want to classify", ["All", "Selected rows"], key="heuristic_data_type", index=0)
            st.form_submit_button("Predict categories", on_click=run_prediction)
        
# --- Eisenhower heuristic sidebar function ---
def eisenhower_heuristic_sidebar():
    with st.expander("🧠 Clasificación heurística de Eisenhower"):
        with st.form(key="heuristic_eisenhower_form", clear_on_submit=True):
            st.markdown("Este módulo aplica una clasificación heurística (sin usar GPT) a las actividades según su subactividad.")
            if "df_original" not in st.session_state:
                st.warning("Carga un archivo para habilitar esta opción.")
                return
            st.selectbox("Selecciona qué datos quieres clasificar", ["All", "Selected rows"], key="eisen_data_type_heuristic", index=0)
            st.form_submit_button("🏷️ Clasificar heurísticamente", on_click=classify_eisenhower_heuristic)


def heuristic_classification():
    def run_expand_labels():
        interval = st.session_state.heuristic_interval
        all = st.session_state.df_original
        st.session_state.undo_df = all.copy()

        temporal_slots = wt.find_temporal_slots(all, inactivity_threshold=pd.Timedelta(f'{interval}s'))
        case_expand = wt.expand_slots(all, temporal_slots, column='Case')

        all['Case'] = case_expand

    with st.expander("Heuristic labeling"):
        with st.form(key='heuristic_labeling'):
            st.slider("Interval size (in seconds)", min_value=0, max_value=300, key='heuristic_interval')
            st.form_submit_button("Expand case labels", on_click=run_expand_labels)

def automated_classification(view_options, mensaje_container):
    def run_auto_classify():
        select_class = st.session_state.auto_type
        openai_key = st.session_state.openai_key
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
        # Nueva lógica: solo sobrescribe si Subactivity es None, "", o "Unclassified"
        if filter_app is not None:
            mask = all.loc[filter_app, 'Subactivity'].isin([None, "", "Unclassified"])
            all.loc[filter_app, 'Activity'] = all.loc[filter_app, 'Activity'].where(~mask, classification)
            all.loc[filter_app, 'Subactivity'] = all.loc[filter_app, 'Subactivity'].where(~mask, ["Unspecified " + c for c in classification])
        else:
            mask = all['Subactivity'].isin([None, "", "Unclassified"])
            all['Activity'] = all['Activity'].where(~mask, classification)
            all['Subactivity'] = all['Subactivity'].where(~mask, ["Unspecified " + c for c in classification])


    with st.expander("Automated labeling"):
        with st.form(key='auto_labeling'):
            st.text_input("Set OpenAI key", type="password", key='openai_key')
            st.text_input("Set OpenAI org", type="password", key='openai_org')

            if view_options[st.session_state.view_type].has_time_blocks:
                options = ["All", "Selected date", "Selected rows"]
                index = 1
            else:
                options = ["All"]
                index = 0

            st.selectbox("Choose what data you want to classify", options, index=index, key='auto_type')

            st.form_submit_button("Start classification", on_click=run_auto_classify)


def classify_eisenhower_heuristic():
    try:
        df = st.session_state.df_original
        st.session_state.undo_df = df.copy()

        tipo_datos = st.session_state.eisen_data_type
        if tipo_datos == "All":
            to_classify = df[df['Subactivity'].notna() & df['Eisenhower'].isna()]
        elif tipo_datos == "Selected rows":
            selected_ids = st.session_state.filas_seleccionadas['ID'].tolist()
            to_classify = df[df["ID"].isin(selected_ids) & df['Subactivity'].notna()]
        else:
            st.warning("Tipo de datos no válido.")
            return

        if to_classify.empty:
            st.warning("No hay filas que puedan ser etiquetadas heurísticamente.")
            return

        resultados = to_classify['Subactivity'].apply(clasificar_eisenhower_por_heuristica)
        df.loc[to_classify.index, "Eisenhower"] = resultados
        st.success("Clasificación heurística completada.")
    except Exception as e:
        logging.exception("Error en clasificación heurística Eisenhower", exc_info=e)
        st.error("Error inesperado durante la clasificación heurística.")

def cases_classification():
    dicc_core_color = st.session_state.get("dicc_core_color", {})
    def apply_label_to_selection(**kwargs):
        if "df_original" not in st.session_state or "filas_seleccionadas" not in st.session_state:
            st.warning("No hay filas seleccionadas o no se ha cargado el dataset.")
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
            st.markdown("### Case labels")
            for case in st.session_state.all_cases:
                if case != "":
                    st.button(case, on_click=save_case_button, args=(case,), use_container_width=True)
