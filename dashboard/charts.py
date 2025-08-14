import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd

def show_activity_dashboard(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.warning("No hay datos para mostrar en el dashboard.")
        return

    df_graph = df.copy()
    st.session_state["df_graph"] = df_graph
    df_graph['Date'] = df_graph['Begin'].dt.date

    subactividades_disponibles = sorted(df_graph['Subactivity'].dropna().unique())
    subactividades_seleccionadas = st.multiselect(
        "🔍 Filtrar por subactividad",
        options=subactividades_disponibles,
        default=subactividades_disponibles,
        help="Selecciona una o varias subactividades para visualizar"
    )

    if subactividades_seleccionadas:
        df_graph = df_graph[df_graph['Subactivity'].isin(subactividades_seleccionadas)]

    df_graph = df_graph[df_graph['Subactivity'].notnull()]

    if df_graph.empty:
        st.warning("No se han asignado subactividades aún. Etiqueta algunas actividades para ver las visualizaciones.")
        return

    df_grouped = df_graph.groupby(['Date', 'Subactivity'])['Duration'].sum().reset_index()
    df_grouped['Duration (min)'] = df_grouped['Duration'] / 60

    chart = alt.Chart(df_grouped).mark_bar().encode(
        x='Date:T',
        y='Duration (min):Q',
        color='Subactivity:N',
        tooltip=['Date', 'Subactivity', 'Duration (min)']
    ).properties(width=700, height=400)

    df_pie = df_graph.groupby('Subactivity')['Duration'].sum().reset_index()
    df_pie['Duration (min)'] = df_pie['Duration'] / 60

    df_line = df_grouped.groupby('Date')['Duration (min)'].sum().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📌 Duración por día y subactividad")
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.markdown("#### 📈 Evolución diaria total")
        fig_line = px.line(df_line, x='Date', y='Duration (min)', markers=True)
        fig_line.update_layout(title_text=None, height=400)
        st.plotly_chart(fig_line, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🧯 Mapa de Calor: Duración por Hora y Fecha")
        df_graph['Hour'] = df_graph['Begin'].dt.hour
        df_heatmap = df_graph.groupby(['Date', 'Hour'])['Duration'].sum().reset_index()
        df_heatmap['Duration (min)'] = df_heatmap['Duration'] / 60
        heatmap = alt.Chart(df_heatmap).mark_rect().encode(
            x=alt.X('Hour:O', title='Hora del día'),
            y=alt.Y('Date:T', title='Fecha'),
            color=alt.Color('Duration (min):Q', scale=alt.Scale(scheme='blues')),
            tooltip=['Date', 'Hour', 'Duration (min)']
        ).properties(width=700, height=400)
        st.altair_chart(heatmap, use_container_width=True)
    with col4:
        st.markdown("#### 🎯 Distribución total de tiempo")
        fig_pie = px.pie(df_pie, names='Subactivity', values='Duration (min)')
        fig_pie.update_layout(title_text=None, showlegend=False, height=400)
        fig_pie.update_traces(textinfo='none', hovertemplate='%{label}: %{value:.1f} min<extra></extra>')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### 📋 Tabla de tiempo total por subactividad")
    st.dataframe(df_pie, use_container_width=True)