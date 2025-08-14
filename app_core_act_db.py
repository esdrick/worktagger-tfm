import io
import logging
import math

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import clasificacion_core_act
import core_act as activities_loader
import analysis as wt
import views

from utils.styles import change_color
from dashboard.eisenhower import plot_eisenhower_matrix
from dashboard.recommendations import show_productivity_recommendations
from dashboard.chatbot import show_productivity_chatbot
from dashboard.charts import show_activity_dashboard
from dashboard.classification import (
    manual_classification_sidebar,
    eisenhower_heuristic_sidebar,
    heuristic_prediction,
    automated_classification,
    heuristic_classification
)

from config.constants import EISEN_OPTIONS


SAMPLE_DATA_URL = "https://raw.githubusercontent.com/project-pivot/labelled-awt-data/main/data/awt_data_1_pseudonymized.csv"

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO, # Set the logging level
    handlers=[logging.StreamHandler()] # Ensures output to Streamlit's terminal
)

@st.cache_data
def load_activities():
    return activities_loader.load_activities()

# @st.cache_resource
def load_view_options():
    return {
        "Time view": views.TimeView(),
        "Active window view": views.ActiveWindowView(),
        "Activity view": views.ActivityView(),
        "Work slot view": views.WorkSlotView()
    }


def split_df(input_df, batch_size, current_page):
    start_idx = (current_page - 1) * batch_size
    end_idx = start_idx + batch_size

    return input_df.iloc[start_idx:end_idx]


def paginate_df(dataset):
    def update_input_current_page_before():
        st.session_state.current_page = st.session_state.input_current_page_before
        st.session_state.input_current_page_after = st.session_state.current_page
        st.session_state.input_current_page_before = st.session_state.current_page

    pagination_menu = st.columns((4,1,1))
    with pagination_menu[2]:
        batch_size = st.selectbox("Page Size", options=[10,20,50,100, "all day"], index=2, key="page_size")
        if batch_size=="all day":
            batch_size = len(dataset)

    with pagination_menu[1]:
        total_pages = math.ceil(len(dataset)/ batch_size)

        if st.session_state.current_page > total_pages:
            st.session_state.current_page = total_pages

        st.session_state.input_current_page_after = st.session_state.current_page
        st.session_state.input_current_page_before = st.session_state.current_page
        st.number_input('Page',min_value = 1, max_value = total_pages,  key='input_current_page_before', on_change=update_input_current_page_before)

    with pagination_menu[0]:
        st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}** ")

    page = split_df(dataset, batch_size, st.session_state.current_page)

    return page, batch_size, total_pages

def apply_styles(page, format_table):

    toggle_block_colours = format_table['toggle_block_colours']
    toggle_begin_end_colours = format_table['toggle_begin_end_colours']
    max_time_between_activities = format_table['max_time_between_activities']


    def resaltar_principio_fin_bloques(fila):
        valor_actual_b = fila['Begin']
        valor_actual_e = fila['End']
        valor_actual_e = pd.to_datetime(valor_actual_e, format='%d/%m/%Y %H:%M')
        fila_sig = st.session_state.df_original.iloc[fila.name + 1] if fila.name + 1 < len(st.session_state.df_original) else None
        fila_ant = st.session_state.df_original.iloc[fila.name - 1] if fila.name - 1 >= 0 else None
        if fila_ant is not None:
            fila_ant['End'] = pd.to_datetime(fila_ant['End'], format='%d/%m/%Y %H:%M')

        dif_tiempo_ant = (valor_actual_b-fila_ant['End']).total_seconds()/60 if fila_ant is not None else 0

        dif_tiempo_sig = (fila_sig['Begin']-valor_actual_e).total_seconds()/60 if fila_sig is not None else 0

        ls_estilos = asignar_color(fila)
        if fila_sig is None or dif_tiempo_sig>max_time_between_activities :

            ls_estilos[6] = 'background-color:#808080'
        if fila_ant is None or dif_tiempo_ant>max_time_between_activities or fila.name==0 :

            ls_estilos[5] = 'background-color:#808080'
        return ls_estilos


    if toggle_block_colours and not toggle_begin_end_colours:
        result = page.style.apply(asignar_color,axis=1)
    elif toggle_begin_end_colours:
        result = page.style.apply(resaltar_principio_fin_bloques, axis=1)
    else:
        result = page.style.apply(asignar_color_sin_estilos,axis=1)

    return result

def to_csv(df):
    output = io.BytesIO()
    df.to_csv(output, sep = ";",  index=False, date_format= '%d/%m/%Y %H:%M:%S')
    return output.getvalue().decode('utf-8')

def download_csv(df):
    excel_data = to_csv(df.drop(columns=['Change', 'Begin Time', 'Ending Time', 'ID']))
    st.download_button(
        label="⬇️ Download CSV",
        data=excel_data,
        file_name='dataframe.csv',
        mime='text/csv',
        use_container_width=True
    )

def asignar_color(s):
    col = '#FFFFFF'
    if isinstance(s.Activity, list):
        if len(s.Activity) == 1:
            activity = s.Activity[0]
        else:
            activity = None
    else:
        activity = s.Activity

    if activity in dicc_core_color:
        col = dicc_core_color[activity]

    return [f'background-color:{col}']*len(s)


def asignar_color_sin_estilos(s):
    return ['background-color:#FFFFFF'] * len(s)

def display_undo_button():
    def undo_last_action():
        st.session_state.df_original = st.session_state.undo_df
        st.session_state.undo_df = None

    st.button("↩️ Undo", disabled=(st.session_state.undo_df is None), on_click = undo_last_action, use_container_width=True)

def display_select_all_button():
    return st.button("✅ Select all in this page", use_container_width=True)

@st.fragment
def display_events_table(df, format_table, batch_size, column_config, column_order=None):
    select_actions_col = st.columns(3)
    with select_actions_col[0]:
        select_all = st.button("✅ Select all in this page", use_container_width=True)
        if select_all:
            df.loc[:,"Change"] = True
    with select_actions_col[1]:
        select_none = st.button("🚫 Select none in this page", use_container_width=True)
        if select_none:
            df.loc[:,"Change"] = False
    with select_actions_col[2]:
        select_invert = st.button("🔄 Invert selection", use_container_width=True)
        if select_invert:
            df.loc[:,"Change"] = ~(df["ID"].isin(st.session_state.filas_seleccionadas["ID"]))


    styled_df = apply_styles(df, format_table)

    disabled = df.columns.difference(['Change'])

    # Shows table
    edited_df = st.data_editor(
        styled_df,
        column_config=column_config,
        column_order=column_order,
        disabled=disabled,
        hide_index=True,
        key="selector",
        use_container_width = True,
        height= int(35.2*(batch_size+1))
    )


    # Filter rows that have been selected
    filas_seleccionadas = edited_df[edited_df['Change']]
    st.session_state.filas_seleccionadas = filas_seleccionadas

    return filas_seleccionadas

def display_pagination_bottom(total_pages):
    def update_input_current_page_after():
        st.session_state.current_page = st.session_state.input_current_page_after
        st.session_state.input_current_page_before = st.session_state.current_page
        st.session_state.input_current_page_after = st.session_state.current_page

    botton_menu = st.columns((4,1,1))
    with botton_menu[2]:
        st.session_state.input_current_page_after = st.session_state.current_page
        st.number_input('Page',min_value = 1, max_value = total_pages, key='input_current_page_after', on_change=update_input_current_page_after)

def changed_file():
    if "df_original" in st.session_state:
        del st.session_state["df_original"]

def reset_current_page():
    st.session_state["current_page"] = 1

@st.fragment
def display_view(selected_view, selected_df, format_table):
    if len(selected_df) == 0:
        st.error("There is no data for the selected filters 😞. Why don't you try with another one? 😉")
    else:
        try:
            button_column = st.columns(3)
            with button_column[0]:
                display_undo_button()
            with button_column[2]:
                download_csv(st.session_state.df_original)

            page, batch_size, total_pages = paginate_df(selected_df)

            column_config, column_order = selected_view.view_config(max_dur=selected_df["Duration"].max())

            selected_rows = display_events_table(page, format_table, batch_size, column_config, column_order)
            display_pagination_bottom(total_pages)
        except Exception as e:
            logging.exception(f"There was an error while displaying the table", exc_info=e)
            st.error("There was an error processing the request. Try again")

def display_label_palette(selected_df):
    if len(selected_df) == 0:
        st.title("Label cases")
        st.warning("No data to label")
        st.title("Label activities")
        st.warning("No data to label")
    else:
        try:
            st.title("Label cases")
            # Classification logic for cases is now handled in dashboard/classification.py
            st.title("Label activities")
            # Clasificación automática primero, luego heurística, luego manual
            automated_classification(view_options, mensaje_container)

            heuristic_classification()

            manual_classification_sidebar(dicc_core, dicc_subact, dicc_core_color, all_sub, apply_label_to_selection, change_color)
        except Exception as e:
            logging.exception(f"There was an error while displaying the sidebar", exc_info=e)
            st.error("There was an error processing the request. Try again")


# --- Lógica de etiquetado: aplicar etiquetas a la selección actual ---
def apply_label_to_selection(**kwargs):
    if "df_original" not in st.session_state or "filas_seleccionadas" not in st.session_state:
        st.warning("No hay filas seleccionadas o no se ha cargado el dataset.")
        return

    df_original = st.session_state.df_original
    selected_ids = st.session_state.filas_seleccionadas['ID'].tolist()

    for key, value in kwargs.items():
        df_original.loc[df_original['ID'].isin(selected_ids), key] = value

    st.session_state.df_original = df_original

def display_table_formatter(selected_view):
    blocks_column, begin_column = st.columns(2)
    max_time_between_activities = 0
    with blocks_column:
        toggle_block_colours = st.toggle('Blocks colours', value=True)
    with begin_column:
        if selected_view.has_time_blocks:
            toggle_begin_end_colours = st.toggle('Begin-End colours')
            if toggle_begin_end_colours:
                max_time_between_activities = st.slider("Maximum time between activities (minutes)", min_value=0, max_value=30, value=5)
        else:
            toggle_begin_end_colours = False

    return {
        'toggle_block_colours': toggle_block_colours,
        'toggle_begin_end_colours': toggle_begin_end_colours,
        'max_time_between_activities': max_time_between_activities
    }


st.set_page_config(layout="wide")

# --- Custom Styles ---
st.markdown("""
<style>
h1, h2, h3, h4 {
    color: #003366;
}
div[data-testid="stExpander"] {
    border: 2px solid #cccccc;
    border-radius: 10px;
    background-color: #f9f9f9;
}
</style>
""", unsafe_allow_html=True)


dicc_core, dicc_subact, dicc_map_subact, dicc_core_color = load_activities()
all_sub = [f"{s} - {c}" for c in dicc_subact for s in dicc_subact[c]]

with st.expander("📁 Upload your data", expanded=True):
    st.markdown("""
    Upload your Tockler data file (`CSV` or `tracker.db`) exported from Tockler.
    You can obtain it from: **Tockler > Search > Export**, or locate `tracker.db` manually.

    **Accepted formats**: `.csv`, `.db`
    **Maximum size**: `200MB`
    """)

    archivo_cargado = st.file_uploader(
        label="Choose your Tockler data file",
        type=["csv", "db"],
        key="source_file",
        on_change=changed_file
    )

    filter_by_time = st.slider(
        "Remove active windows shorter than (seconds):",
        min_value=0, max_value=300, value=0,
        help="Any activity shorter than this will be discarded.",
        on_change=changed_file
    )

    st.divider()

    st.markdown("""
    You can also try the app with example data:
    [👉 Sample dataset (GitHub)](https://github.com/project-pivot/labelled-awt-data)
    """)

    load_sample_data = st.button("🔄 Load sample data", type="primary", on_click=changed_file)

# Contenedor de mensajes para mostrar estados de carga
mensaje_container = st.empty()

#
# --- Data Initialization ---
#
if "df_original" not in st.session_state:
    st.session_state["current_page"] = 1
    st.session_state["last_acts"] = []
    st.session_state["next_day"] = None
    st.session_state["a_datetime"] = None
    st.session_state["undo_df"] = None
    st.session_state["all_cases"] = set()

    if archivo_cargado is not None or load_sample_data:
        mensaje_container.write("Loading...")
        if load_sample_data:
            data_expanded = clasificacion_core_act.simple_load_file(url_link=SAMPLE_DATA_URL, dayfirst=True)
        elif archivo_cargado is not None and archivo_cargado.name.endswith(".db"):
            import sqlite3
            import tempfile
            import os
            # Save the uploaded .db file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(archivo_cargado.read())
                tmp_file_path = tmp_file.name
            # Connect to the SQLite database
            conn = sqlite3.connect(tmp_file_path)
            # Select the required columns from the TrackItems table
            query = """
            SELECT id, app, taskName, title, url, color, beginDate, endDate
            FROM TrackItems
            """
            df_sqlite = pd.read_sql_query(query, conn)
            conn.close()
            os.unlink(tmp_file_path)
            # Convert beginDate and endDate from ms since epoch to datetime
            df_sqlite['Begin'] = pd.to_datetime(df_sqlite['beginDate'], unit='ms')
            df_sqlite['End'] = pd.to_datetime(df_sqlite['endDate'], unit='ms')
            df_sqlite['Duration'] = (df_sqlite['End'] - df_sqlite['Begin']).dt.total_seconds()
            # Compose the expected columns
            df_sqlite['Merged_titles'] = df_sqlite['title']
            df_sqlite['App'] = df_sqlite['app']
            df_sqlite['Activity'] = None
            df_sqlite['Subactivity'] = None
            df_sqlite['Case'] = None
            df_sqlite['Eisenhower'] = None
            df_sqlite['Change'] = False
            df_sqlite['Begin Time'] = df_sqlite['Begin'].dt.strftime('%H:%M:%S')
            df_sqlite['Ending Time'] = df_sqlite['End'].dt.strftime('%H:%M:%S')
            df_sqlite['ID'] = range(1, len(df_sqlite) + 1)
            # Arrange columns as expected
            df_sqlite = df_sqlite[[
                'Change', 'ID', 'Merged_titles', 'Begin', 'End', 'Begin Time', 'Ending Time',
                'Duration', 'Activity', 'Subactivity', 'Case', 'Eisenhower', 'App'
            ]]
            data_expanded = df_sqlite
        elif archivo_cargado is not None:
            data_expanded = clasificacion_core_act.simple_load_file(loaded_file=archivo_cargado)
        if filter_by_time > 0:
            data_expanded = data_expanded[data_expanded['Duration'] >= filter_by_time]

        mensaje_container.write("File loaded")

        data_expanded['ID'] = range(1,len(data_expanded)+1)
        data_expanded = data_expanded.reset_index(drop=True)
        # Only parse if not already datetime (i.e., only for CSVs)
        if not (archivo_cargado is not None and archivo_cargado.name.endswith(".db")):
            data_expanded['Begin'] = pd.to_datetime(data_expanded['Begin'], format='%d/%m/%Y %H:%M:%S')
            data_expanded['End'] = pd.to_datetime(data_expanded['End'], format='%d/%m/%Y %H:%M:%S')
            data_expanded['Begin Time'] =data_expanded['Begin'].dt.strftime('%H:%M:%S')
            data_expanded['Ending Time']= data_expanded['End'].dt.strftime('%H:%M:%S')
            data_expanded['Change'] = False
            # Initialize Eisenhower quadrant label (manual classification)
            data_expanded['Eisenhower'] = None
        st.session_state.df_original = data_expanded[['Change','ID','Merged_titles','Begin','End','Begin Time','Ending Time', 'Duration', 'Activity', 'Subactivity', 'Case', 'Eisenhower', 'App']]

        st.session_state.all_cases = set(data_expanded["Case"].dropna().unique())

if "df_original" in st.session_state:
    view_options = load_view_options()

    view_type = st.radio(label="Select view", options=view_options.keys(), format_func=lambda x: view_options[x].label, key='view_type', horizontal=True, on_change=reset_current_page)
    selected_view = view_options[view_type]

    selected_df = selected_view.view_filter(st.session_state.df_original, reset_current_page)

    format_table = display_table_formatter(selected_view)

    display_view(selected_view, selected_df, format_table)

    with st.sidebar:
        display_label_palette(selected_df)
        heuristic_prediction()
        eisenhower_heuristic_sidebar()

    with st.expander("📊 Dashboard de Actividades", expanded=True):
        st.markdown("### 🧩 Panel de Visualizaciones")
        if len(st.session_state.df_original) > 0:
            show_activity_dashboard(st.session_state.df_original)

    with st.expander("🧭 Matriz de Eisenhower – Ver detalle por subactividad", expanded=False):
        plot_eisenhower_matrix(st.session_state.df_original)
        show_productivity_recommendations()

    show_productivity_chatbot()