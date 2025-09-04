import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.constants import EISEN_OPTIONS

def show_productivity_recommendations():
    df = st.session_state.df_original.copy()
    df['Duration_min'] = df['Duration'] / 60
    
    # Verificar que existen datos clasificados
    df_classified = df[df['Eisenhower'].notna()]
    if df_classified.empty:
        st.warning("No activities classified with Eisenhower matrix yet. Please classify some activities first.")
        return
    
    # 🎯 INTELLIGENT PRODUCTIVITY ANALYSIS
    st.markdown("### 🎯 Intelligent Productivity Analysis")
    
    # Main metrics in minimalist cards
    tiempo_cuadrante = (
        df_classified
        .groupby('Eisenhower')['Duration_min']
        .sum()
    )
    
    # 🔧 FIX: Cálculo directo usando EISEN_OPTIONS
    total_tiempo = tiempo_cuadrante.sum()
    tiempo_productivo = (
        tiempo_cuadrante.get(EISEN_OPTIONS[0], 0) +  # "I: Urgent & Important"
        tiempo_cuadrante.get(EISEN_OPTIONS[1], 0)    # "II: Not urgent but Important"
    )
    tiempo_improductivo = tiempo_cuadrante.get(EISEN_OPTIONS[3], 0)  # "IV: Not urgent & Not important"
    eficiencia = (tiempo_productivo / total_tiempo * 100) if total_tiempo > 0 else 0
    
    # Main metrics cards
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
                Efficiency
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
                Productive time
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
                Time wasted
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calculate focus score based on productive/unproductive ratio
        if tiempo_improductivo > 0 and tiempo_productivo > 0:
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
                Focus index
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 📊 INTERACTIVE DISTRIBUTION CHART
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### 📊 Time distribution")
        
        if not tiempo_cuadrante.empty:
            # Create donut chart
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
        st.markdown("#### 🎯 AI Recommendation")
        
        # Generate intelligent recommendation based on data
        if eficiencia >= 80:
            recomendacion = f"🌟 <strong>Excellent productivity!</strong> Keep this pace and consider sharing your strategies."
            color = "#10B981"
        elif eficiencia >= 60:
            recomendacion = f"📈 <strong>Good work.</strong> Reduce {tiempo_improductivo:.0f}min of distractions to improve."
            color = "#F59E0B"
        else:
            recomendacion = f"🚨 <strong>Focus needed.</strong> Eliminate {tiempo_improductivo:.0f}min of unproductive time and plan better."
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
                Personalized Analysis
            </div>
            <div style="color: #333; font-size: 14px; line-height: 1.5;">
                {recomendacion}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 🎮 INTERACTIVE GOAL CONFIGURATOR
    st.markdown("### 🎮 Set Goals")
    
    with st.container(border=True):
        # Add internal spacing
        st.markdown("<div style='padding: 16px;'>", unsafe_allow_html=True)
        
        # Calculate dynamic limits based on current data
        max_tiempo_total = int(total_tiempo * 1.5) if total_tiempo > 0 else 480  # 150% of current time as maximum
        max_improductivo = max(180, int(tiempo_improductivo * 2))  # Minimum 3h or double current
        
        # RECOMMENDED values based on productivity studies
        if total_tiempo > 2000:  # More than 33 hours = weekly/monthly analysis
            # Based on 40h work week (2400 min)
            tiempo_recomendado_productivo = int(total_tiempo * 0.65)  # 65% productive (standard)
            tiempo_recomendado_improductivo = int(total_tiempo * 0.15)  # 15% unproductive maximum
            periodo_texto = "for this period"
        else:  # Daily analysis
            # 8h work day = 480 min
            tiempo_recomendado_productivo = min(480, int(total_tiempo * 0.65)) if total_tiempo > 0 else 320
            tiempo_recomendado_improductivo = min(60, int(total_tiempo * 0.15)) if total_tiempo > 0 else 60
            periodo_texto = "daily"
        
        col1, col2 = st.columns(2)
        
        with col1:
            objetivo_productivo = st.slider(
                f"🎯 Target productive time {periodo_texto} (minutes)",
                min_value=60, 
                max_value=max_tiempo_total, 
                value=tiempo_recomendado_productivo,  # Use recommended value
                step=30,
                help="Combines quadrants I and II (Important)"
            )
            # Show specific recommendation below slider
            st.caption(f"💡 **Recommended**: {tiempo_recomendado_productivo} min ({tiempo_recomendado_productivo/60:.1f}h) - Based on work standards")
            
        with col2:
            limite_improductivo = st.slider(
                f"🚫 Unproductive time limit {periodo_texto} (minutes)",
                min_value=0, 
                max_value=max_improductivo, 
                value=tiempo_recomendado_improductivo,  # Use recommended value
                step=15,
                help="Maximum time in quadrant IV (distractions)"
            )
            # Show specific recommendation below slider
            st.caption(f"💡 **Recommended**: {tiempo_recomendado_improductivo} min ({tiempo_recomendado_improductivo/60:.1f}h) - Maximum 15% of total time")
        
        # Spacing between sections
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Progress towards goals - CORRECTED
        progreso_productivo = min(100, (tiempo_productivo / objetivo_productivo) * 100) if objetivo_productivo > 0 else 0
        progreso_limite = min(100, (tiempo_improductivo / limite_improductivo) * 100) if limite_improductivo > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            color_productivo = "#10B981" if progreso_productivo >= 100 else "#F59E0B" if progreso_productivo >= 80 else "#EF4444"
            st.markdown(f"""
            **Productive progress: {progreso_productivo:.0f}%** ({tiempo_productivo:.0f}/{objetivo_productivo} min)
            <div style="background: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden; margin-top: 8px;">
                <div style="background: {color_productivo}; height: 100%; width: {min(100, progreso_productivo)}%; transition: width 0.3s;"></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            color_limite = "#EF4444" if progreso_limite > 100 else "#F59E0B" if progreso_limite > 80 else "#10B981"
            st.markdown(f"""
            **Unproductive usage: {progreso_limite:.0f}%** ({tiempo_improductivo:.0f}/{limite_improductivo} min)
            <div style="background: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden; margin-top: 8px;">
                <div style="background: {color_limite}; height: 100%; width: {min(100, progreso_limite)}%; transition: width 0.3s;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Close padding div
        st.markdown("</div>", unsafe_allow_html=True)

    # 🚫 PROBLEMATIC APPS WITH BETTER DESIGN
    st.markdown("### 🚫 Apps to Reduce")
    
    df_apps = df[df["Eisenhower"] == EISEN_OPTIONS[3]].groupby("App")["Duration_min"].sum().sort_values(ascending=False)
    
    if not df_apps.empty:
        for i, (app, tiempo) in enumerate(df_apps.head(5).items()):
            if tiempo >= 10:  # Only show apps with significant usage
                urgencia = "🔴 Critical" if tiempo >= 60 else "🟡 Moderate" if tiempo >= 30 else "🟢 Minor"
                
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
                        <span style="color: #666; font-size: 14px;">{tiempo:.0f} minutes wasted</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 12px;">{urgencia}</span><br>
                        <span style="color: #F54927; font-weight: 600;">-{tiempo:.0f}min</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 No problematic apps detected!")

def generate_smart_insights(df):
    """Generate intelligent insights based on usage patterns"""
    insights = []
    
    # Analysis by hour of day
    if 'Begin' in df.columns:
        df['Hour'] = pd.to_datetime(df['Begin']).dt.hour
        peak_hours = df.groupby('Hour')['Duration'].sum().sort_values(ascending=False).head(3)
        insights.append(f"🕐 Your most active hours: {', '.join([f'{h}:00' for h in peak_hours.index])}")
    
    # Most used apps analysis
    top_apps = df.groupby('App')['Duration'].sum().sort_values(ascending=False).head(3)
    insights.append(f"📱 Main apps: {', '.join(top_apps.index)}")
    
    return insights