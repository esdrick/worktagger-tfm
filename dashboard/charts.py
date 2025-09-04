import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def show_activity_dashboard(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.warning("There is no data to display on the dashboard.")
        return

    # SINCRONIZAR CON SESSION STATE - SOLUCIONADO
    df_graph = df.copy()
    df_graph['Date'] = df_graph['Begin'].dt.date
    df_graph['Hour'] = df_graph['Begin'].dt.hour
    
    # Actualizar session state con datos actuales
    st.session_state["df_graph"] = df_graph

    # VERIFICAR SI HAY SUBACTIVIDADES
    subactividades_disponibles = sorted(df_graph['Subactivity'].dropna().unique())
    
    if not subactividades_disponibles:
        st.info("📝 **To see visualizations, first label some activities with subactivities.**")
        st.markdown("Go to the **📋 Main Screen** and use the sidebar tools to classify your activities.")
        return

    # FILTRO MEJORADO 
    subactividades_seleccionadas = st.multiselect(
        "🔍 Filter by subactivity",
        options=subactividades_disponibles,
        default=subactividades_disponibles,
        help="Select one or more subactivities to visualize"
    )

    # APLICAR FILTROS
    if subactividades_seleccionadas:
        df_graph = df_graph[df_graph['Subactivity'].isin(subactividades_seleccionadas)]
    
    df_graph = df_graph[df_graph['Subactivity'].notnull()]

    if df_graph.empty:
        st.warning("There is no data for the selected subactivities.")
        return

    # PREPARAR DATOS PARA VISUALIZACIONES
    df_grouped = df_graph.groupby(['Date', 'Subactivity'])['Duration'].sum().reset_index()
    df_grouped['Duration (min)'] = df_grouped['Duration'] / 60

    df_pie = df_graph.groupby('Subactivity')['Duration'].sum().reset_index()
    df_pie['Duration (min)'] = df_pie['Duration'] / 60
    df_pie = df_pie.sort_values('Duration (min)', ascending=False)

    df_line = df_grouped.groupby('Date')['Duration (min)'].sum().reset_index()

    # GRÁFICOS MEJORADOS
    st.markdown("### 📊 Activity Visualizations")
    
    # Fila 1: Distribución y evolución temporal
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Distribution by Subactivity")
        
        # Gráfico de dona más moderno
        fig_pie = go.Figure(data=[go.Pie(
            labels=df_pie['Subactivity'],
            values=df_pie['Duration (min)'],
            hole=0.5,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )])
        
        fig_pie.update_layout(
            height=400,
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            font=dict(size=12)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Time Evolution")
        
        # Línea de tiempo mejorada
        fig_line = px.line(
            df_line, 
            x='Date', 
            y='Duration (min)', 
            markers=True,
            line_shape='spline'
        )
        fig_line.update_traces(
            line_color='#F54927',
            marker_color='#F54927',
            marker_size=8
        )
        fig_line.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis_title="Date",
            yaxis_title="Time (minutes)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Fila 2: Análisis detallado
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 📅 Activity by Day")
        
        # Gráfico de barras apiladas más claro
        chart = alt.Chart(df_grouped).mark_bar().encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Duration (min):Q', title='Time (min)'),
            color=alt.Color('Subactivity:N', legend=alt.Legend(title="Subactivities")),
            tooltip=[
                alt.Tooltip('Date:T', title='Date'),
                alt.Tooltip('Subactivity:N', title='Subactivity'),
                alt.Tooltip('Duration (min):Q', title='Time (min)', format='.1f')
            ]
        ).properties(
            height=400
        ).resolve_scale(
            color='independent'
        )
        
        st.altair_chart(chart, use_container_width=True)
    
    with col4:
        st.markdown("#### 🕐 Hourly Pattern")
        
        # Heatmap simplificado
        df_heatmap = df_graph.groupby(['Date', 'Hour'])['Duration'].sum().reset_index()
        df_heatmap['Duration (min)'] = df_heatmap['Duration'] / 60
        
        if not df_heatmap.empty:
            heatmap = alt.Chart(df_heatmap).mark_rect().encode(
                x=alt.X('Hour:O', title='Hour of day', scale=alt.Scale(domain=list(range(24)))),
                y=alt.Y('Date:T', title='Date'),
                color=alt.Color(
                    'Duration (min):Q', 
                    scale=alt.Scale(scheme='oranges'),
                    legend=alt.Legend(title="Time (min)")
                ),
                tooltip=[
                    alt.Tooltip('Date:T', title='Date'),
                    alt.Tooltip('Hour:O', title='Hour'),
                    alt.Tooltip('Duration (min):Q', title='Time (min)', format='.1f')
                ]
            ).properties(height=400)
            
            st.altair_chart(heatmap, use_container_width=True)
        else:
            st.info("There is not enough data to show the hourly pattern.")

    # TABLA MEJORADA Y ACTUALIZADA
    st.markdown("### 📋 Detailed Summary")
    
    # Agregar métricas calculadas a la tabla
    df_table = df_pie.copy()
    df_table['Percentage'] = (df_table['Duration (min)'] / df_table['Duration (min)'].sum() * 100).round(1)
    df_table['Time (hours)'] = (df_table['Duration (min)'] / 60).round(2)
    
    # Reordenar columnas
    df_table = df_table[['Subactivity', 'Duration (min)', 'Time (hours)', 'Percentage']]
    df_table.columns = ['Subactivity', 'Time (min)', 'Time (h)', 'Percentage (%)']
    
    # Mostrar con formato mejorado
    st.dataframe(
        df_table,
        use_container_width=True,
        column_config={
            "Time (min)": st.column_config.NumberColumn(
                "Time (min)",
                format="%.0f"
            ),
            "Time (h)": st.column_config.NumberColumn(
                "Time (h)",
                format="%.2f"
            ),
            "Percentage (%)": st.column_config.NumberColumn(
                "Percentage (%)",
                format="%.1f%%"
            )
        }
    )
    
    # INSIGHTS AUTOMÁTICOS
    st.markdown("### 🧠 Automatic Insights")

    # Calcular métricas generales
    total_tiempo = df_graph['Duration'].sum() / 60
    total_actividades = len(df_graph['Subactivity'].dropna().unique())
    dias_activos = len(df_line)
    tiempo_promedio = df_line['Duration (min)'].mean()

    # Todas las métricas con el mismo estilo st.info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info(f"⏱️ **Total time**: {total_tiempo:.0f} min")

    with col2:
        st.info(f"📊 **Subactivities**: {total_actividades}")

    with col3:
        st.info(f"📅 **Active days**: {dias_activos}")

    with col4:
        st.info(f"⏱️ **Daily average**: {tiempo_promedio:.0f} min")

    # Fila inferior: Insights específicos (mismo estilo)
    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        top_actividad = df_pie.iloc[0]
        st.info(f"🏆 **Top subactivity**: {top_actividad['Subactivity']} ({top_actividad['Duration (min)']:.0f} min)")

    with insights_col2:
        hora_pico = df_graph.groupby('Hour')['Duration'].sum().idxmax()
        st.info(f"🕐 **Most active hour**: {hora_pico}:00")

def generate_activity_insights(df):
    """Generates intelligent insights about activity patterns"""
    insights = []
    
    if df.empty:
        return ["Not enough data to generate insights"]
    
    # Análisis temporal
    if 'Begin' in df.columns:
        df['Hour'] = pd.to_datetime(df['Begin']).dt.hour
        df['Weekday'] = pd.to_datetime(df['Begin']).dt.day_name()
        
        # Hora más productiva
        hour_activity = df.groupby('Hour')['Duration'].sum()
        peak_hour = hour_activity.idxmax()
        insights.append(f"🕐 Your most productive hour is {peak_hour}:00")
        
        # Día más activo
        day_activity = df.groupby('Weekday')['Duration'].sum()
        peak_day = day_activity.idxmax()
        insights.append(f"📅 Your most active day is {peak_day}")
    
    # Análisis de concentración
    if 'Subactivity' in df.columns:
        subact_counts = df['Subactivity'].value_counts()
        if len(subact_counts) > 0:
            most_frequent = subact_counts.index[0]
            insights.append(f"🎯 Your most frequent subactivity: {most_frequent}")
    
    return insights