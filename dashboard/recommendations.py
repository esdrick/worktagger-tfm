import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.constants import EISEN_OPTIONS
from utils.calculations import calculate_productive_time

def show_productivity_recommendations():
    df = st.session_state.df_original.copy()
    df['Duration_min'] = df['Duration'] / 60
    
    # 🎯 ANÁLISIS INTELIGENTE DE PRODUCTIVIDAD
    st.markdown("### 🎯 Análisis Inteligente de Productividad")
    
    # Métricas principales en cards minimalistas
    tiempo_cuadrante = (
        df[df['Eisenhower'].notna()]
        .groupby('Eisenhower')['Duration_min']
        .sum()
    )
    
    total_tiempo = tiempo_cuadrante.sum()
    tiempo_productivo = calculate_productive_time(st.session_state.df_original)
    tiempo_improductivo = tiempo_cuadrante.get(EISEN_OPTIONS[3], 0)
    eficiencia = (tiempo_productivo / total_tiempo * 100) if total_tiempo > 0 else 0
    
    # Cards de métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #F54927;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; font-weight: 700; color: #F54927;">
                {eficiencia:.0f}%
            </div>
            <div style="color: #666; font-size: 14px; margin-top: 4px;">
                Eficiencia
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #10B981;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; font-weight: 700; color: #10B981;">
                {tiempo_productivo:.0f}min
            </div>
            <div style="color: #666; font-size: 14px; margin-top: 4px;">
                Tiempo productivo
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #EF4444;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; font-weight: 700; color: #EF4444;">
                {tiempo_improductivo:.0f}min
            </div>
            <div style="color: #666; font-size: 14px; margin-top: 4px;">
                Tiempo perdido
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calcular focus score basado en ratio productivo/improductivo
        if tiempo_improductivo > 0:
            focus_score = max(0, 100 - (tiempo_improductivo / tiempo_productivo * 50))
        else:
            focus_score = 100
            
        st.markdown(f"""
        <div style="
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3B82F6;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; font-weight: 700; color: #3B82F6;">
                {focus_score:.0f}/100
            </div>
            <div style="color: #666; font-size: 14px; margin-top: 4px;">
                Índice de foco
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 📊 GRÁFICO DE DISTRIBUCIÓN INTERACTIVO
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### 📊 Distribución del tiempo")
        
        if not tiempo_cuadrante.empty:
            # Crear gráfico de dona
            fig = go.Figure(data=[go.Pie(
                labels=[q.split(':')[1].strip() for q in tiempo_cuadrante.index],
                values=tiempo_cuadrante.values,
                hole=0.6,
                marker_colors=['#EF4444', '#F59E0B', '#3B82F6', '#10B981']
            )])
            
            fig.update_layout(
                title_text="",
                height=300,
                margin=dict(t=0, b=0, l=0, r=0),
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("#### 🎯 Recomendación IA")
        
        # Generar recomendación inteligente basada en datos
        if eficiencia >= 80:
            recomendacion = f"🌟 <strong>Excelente productividad!</strong> Mantén este ritmo y considera compartir tus estrategias."
            color = "#10B981"
        elif eficiencia >= 60:
            recomendacion = f"📈 <strong>Buen trabajo.</strong> Reduce {tiempo_improductivo:.0f}min de distracciones para mejorar."
            color = "#F59E0B"
        else:
            recomendacion = f"🚨 <strong>Foco necesario.</strong> Elimina {tiempo_improductivo:.0f}min improductivos y planifica mejor."
            color = "#EF4444"
        
        st.markdown(f"""
        <div style="
            background: {color}15;
            border: 1px solid {color}40;
            border-radius: 8px;
            padding: 16px;
            margin-top: 20px;
        ">
            <div style="color: {color}; font-weight: 600; margin-bottom: 8px;">
                Análisis Personalizado
            </div>
            <div style="color: #333; font-size: 14px; line-height: 1.5;">
                {recomendacion}
            </div>
        </div>
        """, unsafe_allow_html=True)

# 🎮 CONFIGURADOR DE OBJETIVOS INTERACTIVO
    st.markdown("### 🎮 Configurar Objetivos")
    
    with st.container(border=True):
        # Agregar espaciado interno
        st.markdown("<div style='padding: 16px;'>", unsafe_allow_html=True)
        
        # Calcular límites dinámicos basados en los datos actuales
        max_tiempo_total = int(total_tiempo * 1.5)  # 150% del tiempo actual como máximo
        max_improductivo = max(180, int(tiempo_improductivo * 2))  # Mínimo 3h o el doble del actual
        
        # Valores RECOMENDADOS basados en estudios de productividad
        if total_tiempo > 2000:  # Más de 33 horas = análisis semanal/mensual
            # Basado en semana laboral de 40h (2400 min)
            tiempo_recomendado_productivo = int(total_tiempo * 0.65)  # 65% productivo (estándar)
            tiempo_recomendado_improductivo = int(total_tiempo * 0.15)  # 15% improductivo máximo
            periodo_texto = "para este periodo"
        else:  # Análisis diario
            # Día laboral de 8h = 480 min
            tiempo_recomendado_productivo = min(480, int(total_tiempo * 0.65))
            tiempo_recomendado_improductivo = min(60, int(total_tiempo * 0.15))
            periodo_texto = "diarios"
        
        col1, col2 = st.columns(2)
        
        with col1:
            objetivo_productivo = st.slider(
                f"🎯 Tiempo productivo objetivo {periodo_texto} (minutos)",
                min_value=60, 
                max_value=max_tiempo_total, 
                value=tiempo_recomendado_productivo,  # Usar valor recomendado
                step=30,
                help="Combina cuadrantes I y II (Importante)"
            )
            # Mostrar recomendación específica debajo del slider
            st.caption(f"💡 **Recomendado**: {tiempo_recomendado_productivo} min ({tiempo_recomendado_productivo/60:.1f}h) - Basado en estándares laborales")
            
        with col2:
            limite_improductivo = st.slider(
                f"🚫 Límite de tiempo improductivo {periodo_texto} (minutos)",
                min_value=0, 
                max_value=max_improductivo, 
                value=tiempo_recomendado_improductivo,  # Usar valor recomendado
                step=15,
                help="Máximo tiempo en cuadrante IV (distracciones)"
            )
            # Mostrar recomendación específica debajo del slider
            st.caption(f"💡 **Recomendado**: {tiempo_recomendado_improductivo} min ({tiempo_recomendado_improductivo/60:.1f}h) - Máximo 15% del tiempo total")
        
        # Espaciado entre secciones
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Progreso hacia objetivos - CORREGIDO
        progreso_productivo = min(100, (tiempo_productivo / objetivo_productivo) * 100) if objetivo_productivo > 0 else 0
        progreso_limite = min(100, (tiempo_improductivo / limite_improductivo) * 100) if limite_improductivo > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            color_productivo = "#10B981" if progreso_productivo >= 100 else "#F59E0B" if progreso_productivo >= 80 else "#EF4444"
            st.markdown(f"""
            **Progreso productivo: {progreso_productivo:.0f}%** ({tiempo_productivo:.0f}/{objetivo_productivo} min)
            <div style="background: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden; margin-top: 8px;">
                <div style="background: {color_productivo}; height: 100%; width: {min(100, progreso_productivo)}%; transition: width 0.3s;"></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            color_limite = "#EF4444" if progreso_limite > 100 else "#F59E0B" if progreso_limite > 80 else "#10B981"
            st.markdown(f"""
            **Uso improductivo: {progreso_limite:.0f}%** ({tiempo_improductivo:.0f}/{limite_improductivo} min)
            <div style="background: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden; margin-top: 8px;">
                <div style="background: {color_limite}; height: 100%; width: {min(100, progreso_limite)}%; transition: width 0.3s;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Cerrar el div de padding
        st.markdown("</div>", unsafe_allow_html=True)

    # 🚫 APPS PROBLEMÁTICAS CON MEJOR DISEÑO
    st.markdown("### 🚫 Apps que Reducir")
    
    df_apps = df[df["Eisenhower"] == EISEN_OPTIONS[3]].groupby("App")["Duration_min"].sum().sort_values(ascending=False)
    
    if not df_apps.empty:
        for i, (app, tiempo) in enumerate(df_apps.head(5).items()):
            if tiempo >= 10:  # Solo mostrar apps con uso significativo
                urgencia = "🔴 Crítico" if tiempo >= 60 else "🟡 Moderado" if tiempo >= 30 else "🟢 Leve"
                
                st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #e1e5e9;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 8px 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <strong>{app}</strong><br>
                        <span style="color: #666; font-size: 14px;">{tiempo:.0f} minutos desperdiciados</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 12px;">{urgencia}</span><br>
                        <span style="color: #F54927; font-weight: 600;">-{tiempo:.0f}min</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 ¡No hay apps problemáticas detectadas!")

def generate_smart_insights(df):
    """Genera insights inteligentes basados en patrones de uso"""
    insights = []
    
    # Análisis por hora del día
    if 'Begin' in df.columns:
        df['Hour'] = pd.to_datetime(df['Begin']).dt.hour
        peak_hours = df.groupby('Hour')['Duration'].sum().sort_values(ascending=False).head(3)
        insights.append(f"🕐 Tus horas más activas: {', '.join([f'{h}:00' for h in peak_hours.index])}")
    
    # Análisis de apps más usadas
    top_apps = df.groupby('App')['Duration'].sum().sort_values(ascending=False).head(3)
    insights.append(f"📱 Apps principales: {', '.join(top_apps.index)}")
    
    return insights