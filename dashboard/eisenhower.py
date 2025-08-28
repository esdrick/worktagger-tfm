import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
from config.constants import EISEN_OPTIONS

def truncate(text, max_len=40):
    return text if len(text) <= max_len else text[:max_len - 3] + '...'

def plot_eisenhower_matrix_plotly(df, max_items=10):
    """
    Matriz de Eisenhower minimalista y moderna usando Plotly
    """
    df_filtered = df[df["Eisenhower"].notna()]
    if df_filtered.empty:
        st.info("No hay subactividades etiquetadas con la matriz de Eisenhower.")
        return

    # Procesar datos
    summary = (
        df_filtered
        .groupby(["Eisenhower", "Subactivity"])["Duration"]
        .sum()
        .div(60)  # segundos → minutos
        .reset_index()
        .sort_values("Duration", ascending=False)
    )

    # Configuración de cuadrantes
    quad_config = {
        "I: Urgente & Importante": {
            "x": 0, "y": 1, 
            "color": "#EF4444", 
            "icon": "🔥", 
            "title": "Urgente & Importante"
        },
        "II: No urgente pero Importante": {
            "x": 1, "y": 1, 
            "color": "#F59E0B", 
            "icon": "⭐", 
            "title": "Importante"
        },
        "III: Urgente pero No importante": {
            "x": 0, "y": 0, 
            "color": "#3B82F6", 
            "icon": "⚡", 
            "title": "Urgente"
        },
        "IV: No urgente & No importante": {
            "x": 1, "y": 0, 
            "color": "#10B981", 
            "icon": "🌱", 
            "title": "Opcional"
        }
    }

    # Crear figura
    fig = go.Figure()

    # Agregar cuadrantes como rectángulos
    for quad, config in quad_config.items():
        # Rectángulo del cuadrante
        fig.add_shape(
            type="rect",
            x0=config["x"], y0=config["y"], 
            x1=config["x"] + 0.95, y1=config["y"] + 0.95,
            fillcolor=config["color"],
            opacity=0.15,
            line=dict(color=config["color"], width=2),
            layer="below"
        )

        # Título del cuadrante
        fig.add_annotation(
            x=config["x"] + 0.475,
            y=config["y"] + 0.88,
            text=f"<b>{config['icon']} {config['title']}</b>",
            showarrow=False,
            font=dict(size=14, color=config["color"]),
            bgcolor="white",
            bordercolor=config["color"],
            borderwidth=1,
            borderpad=8
        )

        # Obtener actividades del cuadrante
        quad_items = summary[summary["Eisenhower"] == quad].head(max_items)
        
        if quad_items.empty:
            activities_text = "Sin actividades"
            total_time = 0
        else:
            activities_list = []
            for i, (_, row) in enumerate(quad_items.iterrows(), 1):
                name = truncate(str(row.Subactivity), 30)
                time = f"{row.Duration:.0f}min"
                activities_list.append(f"{i}. {name} ({time})")
            
            activities_text = "<br>".join(activities_list)
            total_time = summary[summary["Eisenhower"] == quad]["Duration"].sum()

        # Lista de actividades
        fig.add_annotation(
            x=config["x"] + 0.05,
            y=config["y"] + 0.70,
            text=activities_text,
            showarrow=False,
            font=dict(size=10, color="#374151"),
            xanchor="left",
            yanchor="top",
            align="left"
        )

        # Tiempo total
        total_display = f"{total_time/60:.1f}h" if total_time >= 60 else f"{total_time:.0f}min"
        fig.add_annotation(
            x=config["x"] + 0.475,
            y=config["y"] + 0.12,
            text=f"<b>Total: {total_display}</b>",
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor=config["color"],
            borderpad=6
        )

    # Líneas divisorias minimalistas
    fig.add_hline(y=1, line_dash="dot", line_color="#D1D5DB", line_width=1)
    fig.add_vline(x=1, line_dash="dot", line_color="#D1D5DB", line_width=1)

    # Etiquetas de ejes
    fig.add_annotation(
        x=1, y=-0.1,
        text="<b>URGENCIA →</b>",
        showarrow=False,
        font=dict(size=12, color="#6B7280"),
        yanchor="middle"
    )
    
    fig.add_annotation(
        x=-0.1, y=1,
        text="<b>IMPORTANCIA ↑</b>",
        showarrow=False,
        font=dict(size=12, color="#6B7280"),
        textangle=90,
        yanchor="middle"
    )

    # Configuración del layout
    fig.update_layout(
        title=dict(
            text="<b>Matriz de Eisenhower</b>",
            x=0.45,
            y=0.96,
            font=dict(size=20, color="#1F2937")
        ),
        xaxis=dict(
            range=[-0.15, 2.05],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-0.15, 2.05],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        plot_bgcolor="white",
        paper_bgcolor="#FAFAFA",
        width=800,
        height=700,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Mostrar en Streamlit
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "staticPlot": True
        }
    )

# Versión aún más minimalista con subplots
def plot_eisenhower_matrix_minimal(df, max_items=8):
    """
    Versión súper minimalista con cards limpias
    """
    df_filtered = df[df["Eisenhower"].notna()]
    if df_filtered.empty:
        st.info("No hay subactividades etiquetadas con la matriz de Eisenhower.")
        return

    summary = (
        df_filtered
        .groupby(["Eisenhower", "Subactivity"])["Duration"]
        .sum()
        .div(60)
        .reset_index()
        .sort_values("Duration", ascending=False)
    )

    # Configuración minimalista
    quads = [
        {"name": "I: Urgente & Importante", "color": "#EF4444", "icon": "🔥"},
        {"name": "II: No urgente pero Importante", "color": "#F59E0B", "icon": "⭐"},
        {"name": "III: Urgente pero No importante", "color": "#3B82F6", "icon": "⚡"},
        {"name": "IV: No urgente & No importante", "color": "#10B981", "icon": "🌱"}
    ]

    # Crear grid 2x2
    col1, col2 = st.columns(2)
    
    with col1:
        # Cuadrante I
        quad = quads[0]
        items = summary[summary["Eisenhower"] == quad["name"]].head(max_items)
        total = summary[summary["Eisenhower"] == quad["name"]]["Duration"].sum()
        
        with st.container():
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {quad['color']}15, {quad['color']}08);
                border-left: 4px solid {quad['color']};
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                min-height: 300px;
            ">
                <h4 style="color: {quad['color']}; margin-bottom: 15px;">
                    {quad['icon']} Urgente & Importante
                </h4>
                <div style="color: #374151; font-size: 14px; line-height: 1.6;">
            """, unsafe_allow_html=True)
            
            if items.empty:
                st.markdown("*Sin actividades registradas*")
            else:
                for _, row in items.iterrows():
                    st.markdown(f"• {truncate(str(row.Subactivity), 35)} `{row.Duration:.0f}min`")
            
            time_display = f"{total/60:.1f}h" if total >= 60 else f"{total:.0f}min"
            st.markdown(f"""
                </div>
                <div style="
                    margin-top: 15px; 
                    padding: 8px 15px; 
                    background: {quad['color']}; 
                    color: white; 
                    border-radius: 20px; 
                    text-align: center;
                    font-weight: 600;
                ">
                    Total: {time_display}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Cuadrante III
        quad = quads[2]
        items = summary[summary["Eisenhower"] == quad["name"]].head(max_items)
        total = summary[summary["Eisenhower"] == quad["name"]]["Duration"].sum()
        
        with st.container():
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {quad['color']}15, {quad['color']}08);
                border-left: 4px solid {quad['color']};
                border-radius: 12px;
                padding: 20px;
                min-height: 300px;
            ">
                <h4 style="color: {quad['color']}; margin-bottom: 15px;">
                    {quad['icon']} Urgente pero No importante
                </h4>
                <div style="color: #374151; font-size: 14px; line-height: 1.6;">
            """, unsafe_allow_html=True)
            
            if items.empty:
                st.markdown("*Sin actividades registradas*")
            else:
                for _, row in items.iterrows():
                    st.markdown(f"• {truncate(str(row.Subactivity), 35)} `{row.Duration:.0f}min`")
            
            time_display = f"{total/60:.1f}h" if total >= 60 else f"{total:.0f}min"
            st.markdown(f"""
                </div>
                <div style="
                    margin-top: 15px; 
                    padding: 8px 15px; 
                    background: {quad['color']}; 
                    color: white; 
                    border-radius: 20px; 
                    text-align: center;
                    font-weight: 600;
                ">
                    Total: {time_display}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Cuadrante II
        quad = quads[1]
        items = summary[summary["Eisenhower"] == quad["name"]].head(max_items)
        total = summary[summary["Eisenhower"] == quad["name"]]["Duration"].sum()
        
        with st.container():
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {quad['color']}15, {quad['color']}08);
                border-left: 4px solid {quad['color']};
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                min-height: 300px;
            ">
                <h4 style="color: {quad['color']}; margin-bottom: 15px;">
                    {quad['icon']} No urgente pero Importante
                </h4>
                <div style="color: #374151; font-size: 14px; line-height: 1.6;">
            """, unsafe_allow_html=True)
            
            if items.empty:
                st.markdown("*Sin actividades registradas*")
            else:
                for _, row in items.iterrows():
                    st.markdown(f"• {truncate(str(row.Subactivity), 35)} `{row.Duration:.0f}min`")
            
            time_display = f"{total/60:.1f}h" if total >= 60 else f"{total:.0f}min"
            st.markdown(f"""
                </div>
                <div style="
                    margin-top: 15px; 
                    padding: 8px 15px; 
                    background: {quad['color']}; 
                    color: white; 
                    border-radius: 20px; 
                    text-align: center;
                    font-weight: 600;
                ">
                    Total: {time_display}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Cuadrante IV
        quad = quads[3]
        items = summary[summary["Eisenhower"] == quad["name"]].head(max_items)
        total = summary[summary["Eisenhower"] == quad["name"]]["Duration"].sum()
        
        with st.container():
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {quad['color']}15, {quad['color']}08);
                border-left: 4px solid {quad['color']};
                border-radius: 12px;
                padding: 20px;
                min-height: 300px;
            ">
                <h4 style="color: {quad['color']}; margin-bottom: 15px;">
                    {quad['icon']} No urgente & No importante
                </h4>
                <div style="color: #374151; font-size: 14px; line-height: 1.6;">
            """, unsafe_allow_html=True)
            
            if items.empty:
                st.markdown("*Sin actividades registradas*")
            else:
                for _, row in items.iterrows():
                    st.markdown(f"• {truncate(str(row.Subactivity), 35)} `{row.Duration:.0f}min`")
            
            time_display = f"{total/60:.1f}h" if total >= 60 else f"{total:.0f}min"
            st.markdown(f"""
                </div>
                <div style="
                    margin-top: 15px; 
                    padding: 8px 15px; 
                    background: {quad['color']}; 
                    color: white; 
                    border-radius: 20px; 
                    text-align: center;
                    font-weight: 600;
                ">
                    Total: {time_display}
                </div>
            </div>
            """, unsafe_allow_html=True)