import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go

from utils.calculations import calculate_productive_time

def show_productivity_chatbot():
    st.markdown("### 🤖 Asistente de Productividad Inteligente")
    st.markdown("Hazle preguntas sobre tu tiempo, productividad o pide recomendaciones personalizadas.")
    
    # Inicializar sistema de objetivos si no existe
    if "productivity_goals" not in st.session_state:
        st.session_state.productivity_goals = {
            "active_goal": None,
            "goal_history": [],
            "weekly_targets": {},
            "focus_mode": False
        }
    
    # Función auxiliar para asegurar que df_graph tenga todas las columnas necesarias
    def ensure_df_graph_columns(df):
        """Asegura que df_graph tenga todas las columnas necesarias"""
        if df is None or df.empty:
            return df
        
        # Crear copia si no existe
        df_copy = df.copy()
        
        # Verificar que Begin existe y es datetime
        if 'Begin' in df_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_copy['Begin']):
                try:
                    df_copy['Begin'] = pd.to_datetime(df_copy['Begin'])
                except:
                    st.error("Error: No se puede convertir la columna 'Begin' a datetime")
                    return None
        else:
            st.error("Error: No se encontró la columna 'Begin' en los datos")
            return None
        
        # Añadir columnas derivadas solo si no existen
        if 'Date' not in df_copy.columns:
            df_copy['Date'] = df_copy['Begin'].dt.date
        
        if 'Week' not in df_copy.columns:
            df_copy['Week'] = df_copy['Begin'].dt.isocalendar().week
        
        if 'WeekYear' not in df_copy.columns:
            df_copy['WeekYear'] = df_copy['Begin'].dt.year.astype(str) + "-W" + df_copy['Begin'].dt.isocalendar().week.astype(str).str.zfill(2)
        
        return df_copy
    
    # Verificar y procesar datos
    df_graph = None
    if "df_original" in st.session_state and st.session_state.df_original is not None:
        df_graph = ensure_df_graph_columns(st.session_state.df_original)
        if df_graph is not None:
            st.session_state["df_graph"] = df_graph
    else:
        # Si no hay datos originales, verificar si existe df_graph procesado
        if "df_graph" in st.session_state:
            df_graph = st.session_state["df_graph"]
            # Verificar que tiene las columnas necesarias
            if df_graph is not None and 'WeekYear' not in df_graph.columns:
                df_graph = ensure_df_graph_columns(df_graph)
                if df_graph is not None:
                    st.session_state["df_graph"] = df_graph

    # Detectar cambio de archivo y resetear chat
    current_file_info = None
    if df_graph is not None and not df_graph.empty:
        try:
            current_file_info = f"{len(df_graph)}_{df_graph.iloc[0]['Begin']}_{df_graph.iloc[-1]['End'] if 'End' in df_graph.columns else 'no_end'}"
        except:
            current_file_info = f"{len(df_graph)}_current_data"
    
    # Verificar si cambió el archivo
    if st.session_state.get("last_file_info") != current_file_info:
        st.session_state["last_file_info"] = current_file_info
        # Limpiar chat pero mantener df_graph si ya está procesado correctamente
        if "chat_history" in st.session_state:
            del st.session_state["chat_history"]
        
        # Solo recrear df_graph si es necesario
        if df_graph is None and "df_original" in st.session_state:
            df_graph = ensure_df_graph_columns(st.session_state.df_original)
            if df_graph is not None:
                st.session_state["df_graph"] = df_graph
        
        st.rerun()

    # Inicializar historial del chat con mensajes más inteligentes
    if "chat_history" not in st.session_state:
        # Generar mensaje de bienvenida personalizado basado en datos
        welcome_msg = _generate_welcome_message(df_graph)
        st.session_state.chat_history = [
            {"role": "assistant", "content": welcome_msg}
        ]
    
    # Panel de estadísticas mejorado con comparativas - solo si df_graph es válido
    if df_graph is not None and not df_graph.empty and 'WeekYear' in df_graph.columns:
        _show_enhanced_stats_panel(df_graph)
    elif df_graph is not None and not df_graph.empty:
        st.warning("Los datos están cargados pero faltan algunas columnas necesarias. Intentando reprocessar...")
        df_graph = ensure_df_graph_columns(df_graph)
        if df_graph is not None:
            st.session_state["df_graph"] = df_graph
            st.rerun()
    
    # Sistema de objetivos y modo foco
    _show_goals_section(df_graph)
    
    # Preguntas rápidas mejoradas
    quick_questions = [
        "📊 Comparativa semanal",
        "🎯 ¿Cómo voy con mis metas?", 
        "⚠️ ¿Qué me distrae más?",
        "💡 Sugerencias personalizadas",
        "🔥 Activar modo foco"
    ]

    # Preguntas rápidas y botón de reset
    col_questions, col_reset = st.columns([4, 1])
    
    with col_questions:
        cols = st.columns(len(quick_questions))
        for i, question in enumerate(quick_questions):
            with cols[i]:
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    
                    # Procesar respuesta con nuevo sistema inteligente
                    context = _generate_enhanced_context(df_graph, question)
                    
                    with st.spinner("🤔 Analizando tus patrones de productividad..."):
                        response = _get_enhanced_ai_response(question, context, df_graph)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
    
    with col_reset:
        if st.button("🔄", use_container_width=True, help="Reinicia la conversación"):
            welcome_msg = _generate_welcome_message(df_graph)
            st.session_state.chat_history = [
                {"role": "assistant", "content": welcome_msg}
            ]
            st.rerun()

    # Contenedor del chat
    with st.container(border=True):
        st.markdown("💬 **Conversación**")
        
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="
                    background: #F54927; 
                    color: white; 
                    padding: 10px 16px; 
                    border-radius: 16px 16px 4px 16px; 
                    margin: 8px 0 8px 60px;
                    font-size: 14px;
                ">
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: #f8f8f8; 
                    color: #333; 
                    padding: 12px 16px; 
                    border-radius: 16px 16px 16px 4px; 
                    margin: 8px 60px 8px 0;
                    font-size: 14px; 
                    line-height: 1.5;
                ">
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)

    # Input del chat
    prompt = st.chat_input("💭 Pregúntame sobre tu productividad, pide consejos o define nuevos objetivos...")
    
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Sistema de procesamiento mejorado
        context = _generate_enhanced_context(df_graph, prompt)
        
        with st.spinner("🤔 Analizando tus patrones y generando recomendaciones..."):
            response = _get_enhanced_ai_response(prompt, context, df_graph)
            
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

def _generate_welcome_message(df_graph):
    """Genera mensaje de bienvenida personalizado basado en los datos del usuario"""
    if df_graph is None or df_graph.empty:
        return "¡Hola! 👋 Soy tu asistente de productividad.\n\n**¿En qué puedo ayudarte?**\n\n🔍 *Sube tus datos de RescueTime para comenzar el análisis personalizado.*"
    
    # Verificar que tenemos las columnas necesarias
    required_columns = ['Duration', 'App']
    if not all(col in df_graph.columns for col in required_columns):
        return "¡Hola! 👋 Soy tu asistente de productividad.\n\n⚠️ *Los datos cargados no tienen el formato esperado. Verifica que contengan las columnas Duration y App.*"
    
    try:
        # Análisis rápido para personalizar bienvenida
        total_time = df_graph['Duration'].sum() / 60
        days_tracked = df_graph['Date'].nunique() if 'Date' in df_graph.columns else 1
        top_app = df_graph.groupby('App')['Duration'].sum().idxmax()
        
        # Determinar periodo de tiempo
        if days_tracked == 1:
            period = "hoy"
        elif days_tracked <= 7:
            period = f"los últimos {days_tracked} días"
        else:
            period = f"las últimas {days_tracked//7} semanas"
        
        return f"""¡Hola! 👋 He analizado tu actividad de {period}.

**📊 Resumen rápido:**
• **{total_time:.0f} minutos** de actividad registrada
• **{top_app}** es tu aplicación principal
• **{days_tracked} días** de datos disponibles

**🚀 ¿Qué te gustaría explorar?**

🎯 *Puedo ayudarte con:*
• Análisis de patrones y tendencias
• Comparativas semanales
• Sugerencias personalizadas de mejora
• Definir y seguir objetivos de productividad
• Identificar distracciones y optimizar tiempo

*¡Pregúntame lo que quieras saber sobre tu productividad!*"""
    
    except Exception as e:
        return f"¡Hola! 👋 Soy tu asistente de productividad.\n\n⚠️ *Hay un problema procesando los datos: {str(e)[:100]}...*\n\n*Intenta recargar o verifica el formato de los datos.*"

def _show_enhanced_stats_panel(df_graph):
    """Panel de estadísticas mejorado con comparativas semanales"""
    
    # Verificar que tenemos las columnas necesarias
    if df_graph is None or df_graph.empty or 'WeekYear' not in df_graph.columns:
        st.warning("No se pueden mostrar estadísticas: faltan datos o columnas necesarias")
        return
    
    try:
        # Calcular métricas actuales
        total_time = df_graph['Duration'].sum() / 60
        top_app = df_graph.groupby('App')['Duration'].sum().idxmax()
        
        # Calcular tiempo productivo
        if 'Eisenhower' in df_graph.columns:
            productive_time = calculate_productive_time(st.session_state.df_original)
            productive_display = f"{productive_time:.0f} min"
            productivity_pct = (productive_time / total_time * 100) if total_time > 0 else 0
        else:
            productive_display = "Sin clasificar"
            productivity_pct = 0
        
        # Comparativa semanal si hay suficientes datos
        weeks = df_graph['WeekYear'].unique()
        trend_display = ""
        
        if len(weeks) >= 2:
            # Comparar última semana vs anterior
            last_weeks = sorted(weeks)[-2:]
            current_week_data = df_graph[df_graph['WeekYear'] == last_weeks[-1]]
            prev_week_data = df_graph[df_graph['WeekYear'] == last_weeks[-2]]
            
            current_week_time = current_week_data['Duration'].sum() / 60
            prev_week_time = prev_week_data['Duration'].sum() / 60
            
            if prev_week_time > 0:
                change_pct = ((current_week_time - prev_week_time) / prev_week_time) * 100
                if abs(change_pct) > 5:  # Solo mostrar cambios significativos
                    trend_icon = "📈" if change_pct > 0 else "📉"
                    trend_display = f"{trend_icon} {change_pct:+.0f}% vs semana anterior"
        
        # Panel visual mejorado
        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #e8e8e8;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="
                display: grid; 
                grid-template-columns: 1fr 1fr 1fr 1fr; 
                gap: 20px;
                text-align: center;
            ">
                <div>
                    <div style="color: #F54927; font-size: 32px; font-weight: 700; margin-bottom: 4px;">
                        {total_time:.0f}
                    </div>
                    <div style="color: #666; font-size: 12px; font-weight: 500; margin-bottom: 2px;">
                        MINUTOS TOTALES
                    </div>
                    <div style="color: #999; font-size: 10px;">
                        {trend_display}
                    </div>
                </div>
                <div>
                    <div style="color: #333; font-size: 16px; font-weight: 600; margin-bottom: 4px;">
                        {top_app[:15]}{"..." if len(top_app) > 15 else ""}
                    </div>
                    <div style="color: #666; font-size: 12px; font-weight: 500;">
                        APP PRINCIPAL
                    </div>
                </div>
                <div>
                    <div style="color: #28a745; font-size: 20px; font-weight: 600; margin-bottom: 4px;">
                        {productive_display}
                    </div>
                    <div style="color: #666; font-size: 12px; font-weight: 500; margin-bottom: 2px;">
                        TIEMPO PRODUCTIVO
                    </div>
                    <div style="color: #999; font-size: 10px;">
                        {productivity_pct:.0f}% del total
                    </div>
                </div>
                <div>
                    <div style="color: #6c757d; font-size: 20px; font-weight: 600; margin-bottom: 4px;">
                        {len(weeks)}
                    </div>
                    <div style="color: #666; font-size: 12px; font-weight: 500;">
                        SEMANAS DE DATOS
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error mostrando estadísticas: {str(e)}")

def _show_goals_section(df_graph):
    """Sección de objetivos y modo foco"""
    
    goals = st.session_state.productivity_goals
    
    # Mostrar objetivo activo si existe
    if goals["active_goal"]:
        goal = goals["active_goal"]
        
        # Calcular progreso del objetivo actual
        if df_graph is not None and not df_graph.empty:
            progress = _calculate_goal_progress(goal, df_graph)
            
            # Mostrar barra de progreso del objetivo
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px;
                border-radius: 8px;
                margin-bottom: 16px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; font-size: 14px;">🎯 {goal['name']}</div>
                        <div style="font-size: 12px; opacity: 0.9;">{goal['description']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 18px; font-weight: 700;">{progress['percentage']:.0f}%</div>
                        <div style="font-size: 11px; opacity: 0.8;">{progress['current']}/{progress['target']} {progress['unit']}</div>
                    </div>
                </div>
                <div style="
                    background: rgba(255,255,255,0.3);
                    height: 6px;
                    border-radius: 3px;
                    margin-top: 8px;
                    overflow: hidden;
                ">
                    <div style="
                        background: white;
                        height: 100%;
                        width: {min(progress['percentage'], 100)}%;
                        border-radius: 3px;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def _generate_enhanced_context(df_graph, user_question):
    """Genera contexto inteligente mejorado con análisis avanzados"""
    if df_graph is None or df_graph.empty:
        return "No hay datos disponibles para analizar."
    
    context = f"📊 **Análisis Avanzado de Productividad - {datetime.now().strftime('%d/%m/%Y')}**\n\n"
    
    question_lower = user_question.lower()
    
    try:
        # Análisis por patrones de pregunta mejorados
        if any(word in question_lower for word in ['comparativa', 'semanal', 'tendencia', 'progreso']):
            context += _generate_weekly_comparison_context(df_graph)
        
        elif any(word in question_lower for word in ['meta', 'objetivo', 'goal']):
            context += _generate_goals_context(df_graph)
        
        elif any(word in question_lower for word in ['distrae', 'distraccion', 'interrupc']):
            context += _generate_distraction_analysis(df_graph)
        
        elif any(word in question_lower for word in ['sugerencia', 'mejora', 'consejo', 'recomendacion']):
            context += _generate_personalized_suggestions(df_graph)
        
        elif any(word in question_lower for word in ['foco', 'concentr', 'focus']):
            context += _generate_focus_analysis(df_graph)
        
        else:
            # Análisis general mejorado
            context += _generate_comprehensive_summary(df_graph)
    
    except Exception as e:
        context += f"⚠️ Error procesando análisis: {str(e)[:100]}...\n\n"
        context += _generate_basic_summary(df_graph)
    
    return context

def _generate_basic_summary(df_graph):
    """Genera un resumen básico cuando fallan otros análisis"""
    try:
        total_time = df_graph['Duration'].sum() / 60 if 'Duration' in df_graph.columns else 0
        apps_count = df_graph['App'].nunique() if 'App' in df_graph.columns else 0
        
        return f"""📋 **Resumen Básico**
        
• Tiempo total: {total_time:.0f} minutos
• Aplicaciones únicas: {apps_count}
• Registros: {len(df_graph)}

*Algunos análisis avanzados no están disponibles debido a limitaciones en los datos.*
"""
    except:
        return "📋 **Resumen**: Datos básicos disponibles para análisis simple."

def _generate_weekly_comparison_context(df_graph):
    """Genera análisis comparativo semanal detallado"""
    context = "📈 **ANÁLISIS COMPARATIVO SEMANAL**\n\n"
    
    # Verificar que las columnas de semana existan
    if 'WeekYear' not in df_graph.columns:
        context += "ℹ️ *Para comparativas semanales, necesito que los datos tengan información temporal completa.*\n\n"
        return context
    
    weeks = sorted(df_graph['WeekYear'].unique())
    
    if len(weeks) < 2:
        context += "ℹ️ *Necesitas al menos 2 semanas de datos para comparativas.*\n\n"
        return context
    
    try:
        # Comparar las dos últimas semanas completas
        current_week = weeks[-1]
        prev_week = weeks[-2]
        
        current_data = df_graph[df_graph['WeekYear'] == current_week]
        prev_data = df_graph[df_graph['WeekYear'] == prev_week]
        
        # Métricas de comparación
        metrics = {
            'Tiempo total': (current_data['Duration'].sum()/60, prev_data['Duration'].sum()/60, 'min'),
            'Sesiones': (len(current_data), len(prev_data), 'sesiones'),
            'Apps únicas': (current_data['App'].nunique(), prev_data['App'].nunique(), 'apps')
        }
        
        if 'Eisenhower' in df_graph.columns:
            current_productive = current_data[current_data['Eisenhower'].isin(['I: Urgente & Importante', 'II: No urgente pero Importante'])]['Duration'].sum()/60
            prev_productive = prev_data[prev_data['Eisenhower'].isin(['I: Urgente & Importante', 'II: No urgente pero Importante'])]['Duration'].sum()/60
            metrics['Tiempo productivo'] = (current_productive, prev_productive, 'min')
        
        for metric_name, (current, previous, unit) in metrics.items():
            if previous > 0:
                change = ((current - previous) / previous) * 100
                trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                context += f"{trend} **{metric_name}**: {current:.0f} {unit} ({change:+.0f}%)\n"
            else:
                context += f"• **{metric_name}**: {current:.0f} {unit} (nuevo)\n"
        
        # Top apps comparison
        current_top = current_data.groupby('App')['Duration'].sum().nlargest(3)
        
        context += f"\n**🔥 Apps más usadas esta semana:**\n"
        for app, duration in current_top.items():
            context += f"• {app}: {duration/60:.0f} min\n"
    
    except Exception as e:
        context += f"⚠️ Error en análisis semanal: {str(e)[:100]}...\n"
    
    return context + "\n"

def _generate_goals_context(df_graph):
    """Genera contexto sobre objetivos y metas"""
    context = "🎯 **ANÁLISIS DE OBJETIVOS**\n\n"
    
    goals = st.session_state.productivity_goals
    
    if goals["active_goal"]:
        goal = goals["active_goal"]
        progress = _calculate_goal_progress(goal, df_graph)
        
        context += f"**Objetivo Activo**: {goal['name']}\n"
        context += f"**Progreso**: {progress['current']}/{progress['target']} {progress['unit']} ({progress['percentage']:.0f}%)\n"
        context += f"**Estado**: {'¡Objetivo alcanzado! 🎉' if progress['percentage'] >= 100 else 'En progreso 💪'}\n\n"
    else:
        context += "No tienes objetivos activos definidos.\n\n"
    
    # Sugerir objetivos basados en datos
    try:
        if 'Eisenhower' in df_graph.columns and 'Duration' in df_graph.columns:
            total_time = df_graph['Duration'].sum() / 60
            productive_time = df_graph[df_graph['Eisenhower'].isin(['I: Urgente & Importante', 'II: No urgente pero Importante'])]['Duration'].sum() / 60
            productivity_rate = (productive_time / total_time * 100) if total_time > 0 else 0
            
            context += f"**💡 Sugerencias de objetivos basadas en tus datos:**\n"
            if productivity_rate < 60:
                context += f"• Aumentar tiempo productivo a 70% (actual: {productivity_rate:.0f}%)\n"
            if total_time > 480:  # más de 8 horas
                context += f"• Optimizar tiempo total de pantalla a 6-7 horas diarias\n"
    except:
        context += "**💡 Para sugerencias personalizadas, necesito más datos clasificados.**\n"
    
    return context + "\n"

def _generate_personalized_suggestions(df_graph):
    """Genera sugerencias personalizadas basadas en patrones"""
    context = "💡 **SUGERENCIAS PERSONALIZADAS**\n\n"
    
    try:
        total_time = df_graph['Duration'].sum() / 60
        top_apps = df_graph.groupby('App')['Duration'].sum().nlargest(5)
        
        suggestions = []
        
        # Análisis de patrones temporales si tenemos la información
        if 'Begin' in df_graph.columns:
            df_graph['Hour'] = pd.to_datetime(df_graph['Begin']).dt.hour
            hourly_usage = df_graph.groupby('Hour')['Duration'].sum()
            peak_hour = hourly_usage.idxmax()
            
            suggestions.append(f"🕒 **Horario pico**: Tu mayor actividad es a las {peak_hour}:00h. Considera programar tareas importantes en este horario.")
        
        # Análisis de aplicaciones
        if 'Eisenhower' in df_graph.columns:
            distractions = df_graph[df_graph['Eisenhower'] == 'IV: No urgente & No importante']
            if not distractions.empty:
                distraction_time = distractions['Duration'].sum() / 60
                distraction_pct = (distraction_time / total_time) * 100
                if distraction_pct > 20:
                    main_distraction = distractions.groupby('App')['Duration'].sum().idxmax()
                    suggestions.append(f"⚠️ **Reducir distracciones**: {distraction_pct:.0f}% de tu tiempo son distracciones. Enfócate en limitar {main_distraction}.")
        
        # Sugerencias basadas en duración de sesiones
        avg_session = df_graph['Duration'].mean()
        if avg_session < 15:  # sesiones muy cortas
            suggestions.append(f"🔄 **Sesiones fragmentadas**: Tus sesiones promedio duran {avg_session:.0f} min. Intenta bloques más largos para mayor concentración.")
        elif avg_session > 120:  # sesiones muy largas
            suggestions.append(f"⏰ **Descansos necesarios**: Tus sesiones son largas ({avg_session:.0f} min promedio). Considera la técnica Pomodoro.")
        
        # Análisis de diversidad de aplicaciones
        if 'Date' in df_graph.columns:
            apps_per_day = df_graph.groupby('Date')['App'].nunique().mean()
            if apps_per_day > 15:
                suggestions.append(f"🎯 **Reducir dispersión**: Usas {apps_per_day:.0f} apps por día en promedio. Intenta enfocarte en menos herramientas.")
        
        for suggestion in suggestions[:4]:  # Mostrar máximo 4 sugerencias
            context += f"{suggestion}\n\n"
    
    except Exception as e:
        context += f"⚠️ Error generando sugerencias: {str(e)[:100]}...\n\n"
        context += "💡 **Sugerencia general**: Clasifica tus actividades para obtener recomendaciones más específicas.\n\n"
    
    return context

def _calculate_goal_progress(goal, df_graph):
    """Calcula el progreso de un objetivo específico"""
    try:
        if goal['type'] == 'productive_time':
            if 'Eisenhower' in df_graph.columns:
                current = df_graph[df_graph['Eisenhower'].isin(['I: Urgente & Importante', 'II: No urgente pero Importante'])]['Duration'].sum() / 60
            else:
                current = 0
            target = goal['target']
            unit = 'min/día'
        elif goal['type'] == 'reduce_distractions':
            if 'Eisenhower' in df_graph.columns:
                current = df_graph[df_graph['Eisenhower'] == 'IV: No urgente & No importante']['Duration'].sum() / 60
                # Para objetivos de reducción, invertimos el cálculo
                target = goal['target']
                current = max(0, target - current)  # Progreso = cuánto hemos reducido
            else:
                current = 0
                target = goal['target']
            unit = 'min reducidos'
        else:
            current = 0
            target = goal.get('target', 100)
            unit = 'unidades'
        
        percentage = (current / target * 100) if target > 0 else 0
        
        return {
            'current': current,
            'target': target,
            'percentage': percentage,
            'unit': unit
        }
    except Exception as e:
        return {
            'current': 0,
            'target': 100,
            'percentage': 0,
            'unit': 'unidades'
        }

def _generate_distraction_analysis(df_graph):
    """Análisis detallado de distracciones"""
    context = "⚠️ **ANÁLISIS DE DISTRACCIONES**\n\n"
    
    if 'Eisenhower' not in df_graph.columns:
        context += "Para analizar distracciones detalladamente, primero clasifica tus actividades con la Matriz de Eisenhower.\n\n"
        return context
    
    try:
        distractions = df_graph[df_graph['Eisenhower'] == 'IV: No urgente & No importante']
        total_time = df_graph['Duration'].sum() / 60
        
        if distractions.empty:
            context += "¡Excelente! No se detectaron actividades clasificadas como distracciones.\n\n"
            return context
        
        distraction_time = distractions['Duration'].sum() / 60
        distraction_pct = (distraction_time / total_time) * 100
        
        context += f"📊 **Tiempo total en distracciones**: {distraction_time:.0f} min ({distraction_pct:.0f}% del total)\n\n"
        
        # Top distractores
        top_distractors = distractions.groupby('App')['Duration'].sum().nlargest(5)
        context += "**🚫 Principales distractores:**\n"
        for app, duration in top_distractors.items():
            pct = (duration/60 / total_time) * 100
            context += f"• {app}: {duration/60:.0f} min ({pct:.0f}%)\n"
        
        # Análisis temporal de distracciones
        if 'Begin' in df_graph.columns:
            distractions['Hour'] = pd.to_datetime(distractions['Begin']).dt.hour
            distraction_hours = distractions.groupby('Hour')['Duration'].sum().nlargest(3)
            
            context += f"\n**⏰ Horarios más propensos a distracciones:**\n"
            for hour, duration in distraction_hours.items():
                context += f"• {hour}:00h - {duration/60:.0f} min\n"
    
    except Exception as e:
        context += f"⚠️ Error en análisis de distracciones: {str(e)[:100]}...\n"
    
    return context + "\n"

def _generate_focus_analysis(df_graph):
    """Análisis de patrones de concentración"""
    context = "🔍 **ANÁLISIS DE CONCENTRACIÓN**\n\n"
    
    try:
        # Análisis de sesiones continuas
        df_sessions = df_graph.copy()
        df_sessions['SessionLength'] = df_sessions['Duration']
        
        # Categorizar sesiones por duración
        short_sessions = df_sessions[df_sessions['SessionLength'] < 15].shape[0]
        medium_sessions = df_sessions[(df_sessions['SessionLength'] >= 15) & (df_sessions['SessionLength'] < 45)].shape[0]
        long_sessions = df_sessions[df_sessions['SessionLength'] >= 45].shape[0]
        
        total_sessions = len(df_sessions)
        
        context += f"**📊 Distribución de sesiones:**\n"
        context += f"• Sesiones cortas (<15 min): {short_sessions} ({short_sessions/total_sessions*100:.0f}%)\n"
        context += f"• Sesiones medianas (15-45 min): {medium_sessions} ({medium_sessions/total_sessions*100:.0f}%)\n"
        context += f"• Sesiones largas (>45 min): {long_sessions} ({long_sessions/total_sessions*100:.0f}%)\n\n"
        
        # Recomendaciones de concentración
        if short_sessions / total_sessions > 0.6:
            context += "💡 **Recomendación**: Tienes muchas sesiones fragmentadas. Intenta:\n"
            context += "• Técnica Pomodoro (25 min concentrado + 5 min descanso)\n"
            context += "• Bloquear notificaciones durante trabajo importante\n"
            context += "• Definir bloques específicos para tareas profundas\n\n"
        elif long_sessions / total_sessions > 0.4:
            context += "💡 **Recomendación**: Tienes sesiones muy largas. Considera:\n"
            context += "• Descansos regulares cada 45-60 minutos\n"
            context += "• Alternar entre tareas para mantener energía\n"
            context += "• Usar recordatorios para pausas activas\n\n"
        else:
            context += "✅ **¡Excelente balance!** Tienes una buena distribución de sesiones.\n\n"
        
        # Apps que favorecen la concentración
        if 'Eisenhower' in df_graph.columns:
            focused_work = df_graph[df_graph['Eisenhower'].isin(['I: Urgente & Importante', 'II: No urgente pero Importante'])]
            if not focused_work.empty:
                focus_apps = focused_work.groupby('App')['Duration'].sum().nlargest(3)
                context += "**🎯 Apps que favorecen tu concentración:**\n"
                for app, duration in focus_apps.items():
                    context += f"• {app}: {duration/60:.0f} min de trabajo concentrado\n"
    
    except Exception as e:
        context += f"⚠️ Error en análisis de concentración: {str(e)[:100]}...\n"
    
    return context + "\n"

def _generate_comprehensive_summary(df_graph):
    """Genera resumen comprehensivo para preguntas generales"""
    context = "📋 **RESUMEN COMPREHENSIVO**\n\n"
    
    try:
        total_time = df_graph['Duration'].sum() / 60
        days_tracked = df_graph['Date'].nunique() if 'Date' in df_graph.columns else 1
        avg_daily = total_time / days_tracked if days_tracked > 0 else 0
        
        context += f"**📊 Métricas generales:**\n"
        context += f"• Tiempo total registrado: {total_time:.0f} minutos\n"
        context += f"• Días con datos: {days_tracked}\n"
        context += f"• Promedio diario: {avg_daily:.0f} minutos\n\n"
        
        # Top aplicaciones
        top_apps = df_graph.groupby('App')['Duration'].sum().nlargest(5)
        context += "**🔥 Top 5 aplicaciones:**\n"
        for i, (app, duration) in enumerate(top_apps.items(), 1):
            pct = (duration/60 / total_time) * 100
            context += f"{i}. {app}: {duration/60:.0f} min ({pct:.0f}%)\n"
        
        # Análisis de productividad si está disponible
        if 'Eisenhower' in df_graph.columns:
            eisenhower_summary = df_graph[df_graph['Eisenhower'].notna()].groupby('Eisenhower')['Duration'].sum()
            productive_time = calculate_productive_time(st.session_state.df_original)
            
            context += f"\n**🎯 Distribución por importancia:**\n"
            for quadrant, duration in eisenhower_summary.items():
                pct = (duration/60 / total_time) * 100
                context += f"• {quadrant}: {duration/60:.0f} min ({pct:.0f}%)\n"

            context += f"\n**⚠️ IMPORTANTE: Solo las categorías I y II son productivas. Tiempo productivo total: {productive_time:.0f} min**\n"

    except Exception as e:
        context += f"⚠️ Error en resumen: {str(e)[:100]}...\n"

    return context + "\n"

def _get_enhanced_ai_response(user_prompt, context, df_graph):
    """Sistema de IA mejorado con procesamiento de objetivos y recomendaciones"""
    
    # Detectar si el usuario quiere definir un objetivo
    if any(word in user_prompt.lower() for word in ['objetivo', 'meta', 'quiero', 'propósito', 'foco']):
        if any(word in user_prompt.lower() for word in ['activar', 'modo', 'empezar', 'comenzar']):
            return _activate_focus_mode(user_prompt, df_graph)
        elif any(word in user_prompt.lower() for word in ['definir', 'crear', 'nuevo', 'establecer']):
            return _suggest_goal_creation(user_prompt, df_graph)
    
    try:
        openrouter_key = st.secrets["openrouter"]["key"]
    except KeyError:
        return "🔐 **Error de configuración**: No se encontró la clave de API. Contacta al administrador para configurar las credenciales."
    
    # Sistema de prompts mejorado
    system_prompt = """Eres un coach de productividad experto y analista de datos. Tu trabajo es:

1. **Analizar patrones de comportamiento** del usuario basándote en sus datos reales
2. **Dar consejos específicos y accionables** basados en evidencia
3. **Ser motivador pero realista** en tus recomendaciones
4. **Identificar oportunidades de mejora** de forma constructiva
5. **Sugerir técnicas y estrategias** probadas de productividad
6. **Hacer seguimiento del progreso** hacia objetivos

**Estilo de comunicación:**
- Usa emojis relevantes para hacer la conversación más visual
- Estructura tus respuestas con encabezados claros
- Proporciona números y métricas específicas cuando sea posible
- Incluye llamadas a la acción concretas
- Mantén un tono profesional pero amigable

**Especialidades:**
- Análisis de tiempo y patrones de uso
- Técnicas de concentración y flujo
- Gestión de distracciones
- Establecimiento y seguimiento de objetivos
- Optimización de rutinas de trabajo

Responde siempre de forma que el usuario pueda tomar acción inmediata."""

    # Añadir información sobre objetivos activos al contexto
    goals_info = ""
    if st.session_state.productivity_goals["active_goal"]:
        goal = st.session_state.productivity_goals["active_goal"]
        goals_info = f"\n\n**OBJETIVO ACTIVO DEL USUARIO:**\n{goal['name']}: {goal['description']}\nTipo: {goal['type']}, Meta: {goal['target']}"
    
    enhanced_context = context + goals_info

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{enhanced_context}\n\n**Pregunta del usuario:** {user_prompt}"}
    ]
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-app.streamlit.app"
            },
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": messages,
                "temperature": 0.4,  # Ligeramente más creativo para mejores sugerencias
                "max_tokens": 1200   # Más tokens para respuestas más completas
            },
            timeout=30
        )
        
        if response.status_code == 200:
            ai_response = response.json()["choices"][0]["message"]["content"]
            
            # Post-procesar la respuesta para añadir funcionalidades específicas
            return _post_process_response(ai_response, user_prompt, df_graph)
        else:
            return f"⚠️ **Error de API** ({response.status_code}): No pude procesar tu consulta en este momento. Intenta de nuevo más tarde."
            
    except requests.exceptions.Timeout:
        return "⏱️ **Tiempo agotado**: La consulta está tardando demasiado. Intenta con una pregunta más específica."
    except Exception as e:
        return f"⚠️ **Error inesperado**: Ocurrió un problema al procesar tu consulta. Detalles: {str(e)[:100]}..."

def _post_process_response(ai_response, user_prompt, df_graph):
    """Post-procesa la respuesta de IA para añadir funcionalidades específicas"""
    
    # Añadir botones de acción contextual al final de la respuesta
    action_buttons = ""
    
    if any(word in user_prompt.lower() for word in ['comparativa', 'semanal']):
        action_buttons += "\n\n**🎯 Acciones sugeridas:**\n• Pregúntame '¿Cómo puedo mejorar la próxima semana?'\n• Di 'Define un objetivo semanal' para establecer metas"
    
    elif any(word in user_prompt.lower() for word in ['distrae', 'distraccion']):
        action_buttons += "\n\n**🎯 Acciones sugeridas:**\n• Pregunta 'Activar modo foco' para concentrarte\n• Di 'Cómo reducir distracciones' para un plan específico"
    
    elif any(word in user_prompt.lower() for word in ['product', 'eficien']):
        action_buttons += "\n\n**🎯 Acciones sugeridas:**\n• Pregunta 'Define objetivo de productividad'\n• Di 'Análisis de concentración' para optimizar sesiones"
    
    # Añadir recordatorios proactivos si es relevante
    proactive_suggestions = _generate_proactive_suggestions(df_graph)
    
    return ai_response + action_buttons + proactive_suggestions

def _generate_proactive_suggestions(df_graph):
    """Genera sugerencias proactivas basadas en patrones detectados"""
    suggestions = ""
    
    if df_graph is None or df_graph.empty:
        return suggestions
    
    try:
        # Detectar patrones que requieren atención
        total_time = df_graph['Duration'].sum() / 60
        
        # Sugerencia si hay mucho tiempo de pantalla
        if total_time > 480:  # más de 8 horas diarias en promedio
            days = df_graph['Date'].nunique() if 'Date' in df_graph.columns else 1
            avg_daily = total_time / days
            if avg_daily > 480:
                suggestions += f"\n\n💡 **Sugerencia proactiva**: Detecté {avg_daily:.0f} min promedio de pantalla diarios. ¿Te gustaría que te ayude a optimizar este tiempo?"
        
        # Sugerencia si no hay objetivos activos
        if not st.session_state.productivity_goals["active_goal"]:
            if 'Eisenhower' in df_graph.columns:
                suggestions += f"\n\n🎯 **Sugerencia**: Tienes datos clasificados perfectos para definir objetivos. ¿Quieres que te ayude a crear un objetivo personalizado?"
    
    except:
        pass
    
    return suggestions

def _activate_focus_mode(user_prompt, df_graph):
    """Activa el modo foco con recomendaciones específicas"""
    
    # Marcar modo foco como activo
    st.session_state.productivity_goals["focus_mode"] = True
    
    # Analizar datos para dar recomendaciones de foco personalizadas
    if df_graph is not None and not df_graph.empty:
        
        try:
            # Identificar mejores apps para concentración
            if 'Eisenhower' in df_graph.columns:
                focus_apps = df_graph[df_graph['Eisenhower'].isin(['I: Urgente & Importante', 'II: No urgente pero Importante'])]
                if not focus_apps.empty:
                    top_focus_app = focus_apps.groupby('App')['Duration'].sum().idxmax()
                    focus_time = focus_apps.groupby('App')['Duration'].sum().max() / 60
                else:
                    top_focus_app = "tu aplicación principal de trabajo"
                    focus_time = 0
            else:
                top_focus_app = df_graph.groupby('App')['Duration'].sum().idxmax()
                focus_time = df_graph.groupby('App')['Duration'].sum().max() / 60
            
            # Identificar distracciones principales para bloquear
            distractions_to_avoid = []
            if 'Eisenhower' in df_graph.columns:
                distractions = df_graph[df_graph['Eisenhower'] == 'IV: No urgente & No importante']
                if not distractions.empty:
                    distractions_to_avoid = distractions.groupby('App')['Duration'].sum().nlargest(3).index.tolist()
            
            # Determinar mejor horario para concentración
            if 'Begin' in df_graph.columns:
                df_graph['Hour'] = pd.to_datetime(df_graph['Begin']).dt.hour
                if 'Eisenhower' in df_graph.columns:
                    productive_hours = df_graph[df_graph['Eisenhower'].isin(['I: Urgente & Importante', 'II: No urgente pero Importante'])]
                    if not productive_hours.empty:
                        best_hour = productive_hours.groupby('Hour')['Duration'].sum().idxmax()
                    else:
                        best_hour = df_graph.groupby('Hour')['Duration'].sum().idxmax()
                else:
                    best_hour = df_graph.groupby('Hour')['Duration'].sum().idxmax()
            else:
                best_hour = 10
        
        except:
            top_focus_app = "tu aplicación de trabajo principal"
            focus_time = 60
            distractions_to_avoid = ["redes sociales", "mensajería", "entretenimiento"]
            best_hour = 10
    
    else:
        top_focus_app = "tu aplicación de trabajo principal"
        focus_time = 60
        distractions_to_avoid = ["redes sociales", "mensajería", "entretenimiento"]
        best_hour = 10
    
    response = f"""🔥 **MODO FOCO ACTIVADO** 🔥

¡Perfecto! He analizado tus patrones y tengo un plan personalizado para maximizar tu concentración:

## 🎯 **Tu Plan de Concentración Personalizado**

**📱 Aplicación recomendada para trabajo profundo:**
• **{top_focus_app}** - Donde eres más productivo ({focus_time:.0f} min promedio)

**⏰ Horario óptimo para concentración:**
• **{best_hour}:00h** - Tu hora pico de productividad detectada

**🚫 Distracciones a evitar:**"""
    
    for distraction in distractions_to_avoid[:3]:
        response += f"\n• {distraction}"
    
    response += f"""

## 🧠 **Técnica Recomendada: Pomodoro Personalizado**

**Basándome en tus datos, te sugiero:**
1. **Sesiones de 45 minutos** de trabajo concentrado
2. **Descansos de 10 minutos** entre sesiones
3. **Usar {top_focus_app}** como herramienta principal
4. **Bloquer notificaciones** durante las sesiones

## ✅ **Plan de Acción Inmediato**

**Los próximos 90 minutos:**
1. 🔕 Silencia notificaciones
2. 📱 Abre {top_focus_app}
3. ⏲️ Pon timer para 45 minutos
4. 🎯 Enfócate en UNA tarea importante
5. 🎉 ¡Celebra cuando termines!

**💪 ¿Estás listo para comenzar?**

*Pregúntame "¿Cómo voy con mi sesión de foco?" después de tu primera sesión para hacer seguimiento.*"""

    return response

def _suggest_goal_creation(user_prompt, df_graph):
    """Sugiere la creación de objetivos basados en los datos del usuario"""
    
    if df_graph is None or df_graph.empty:
        return """🎯 **CREACIÓN DE OBJETIVOS**

Me encanta que quieras establecer objetivos, pero necesito datos de tu actividad para sugerir metas personalizadas.

**Para empezar:**
1. Sube tus datos de RescueTime
2. Clasifica algunas actividades con la Matriz de Eisenhower
3. Vuelve a preguntarme y te daré objetivos específicos basados en tus patrones

¡Una vez que tenga tus datos, podré sugerir metas súper específicas y alcanzables!"""
    
    try:
        # Analizar datos para sugerir objetivos realistas
        total_time = df_graph['Duration'].sum() / 60
        days = df_graph['Date'].nunique() if 'Date' in df_graph.columns else 1
        avg_daily = total_time / days if days > 0 else 0
        
        suggested_goals = []
        
        # Sugerencias basadas en tiempo productivo
        if 'Eisenhower' in df_graph.columns:
            productive_time = df_graph[df_graph['Eisenhower'].isin(['I: Urgente & Importante', 'II: No urgente pero Importante'])]['Duration'].sum() / 60
            productivity_rate = (productive_time / total_time * 100) if total_time > 0 else 0
            avg_productive_daily = productive_time / days if days > 0 else 0
            
            if productivity_rate < 50:
                target_increase = min(avg_productive_daily + 60, avg_daily * 0.7)  # Aumentar 1h o 70% del tiempo total
                suggested_goals.append({
                    "name": "Aumentar Tiempo Productivo",
                    "description": f"Llegar a {target_increase:.0f} minutos diarios de trabajo importante",
                    "type": "productive_time",
                    "target": target_increase,
                    "current": avg_productive_daily
                })
            
            # Sugerencias para reducir distracciones
            distractions = df_graph[df_graph['Eisenhower'] == 'IV: No urgente & No importante']
            if not distractions.empty:
                distraction_time = distractions['Duration'].sum() / 60
                avg_distraction_daily = distraction_time / days
                if avg_distraction_daily > 30:  # más de 30 min diarios de distracciones
                    target_reduction = max(avg_distraction_daily * 0.5, 15)  # Reducir a la mitad o máximo 15 min
                    suggested_goals.append({
                        "name": "Reducir Distracciones",
                        "description": f"Limitar distracciones a máximo {target_reduction:.0f} minutos diarios",
                        "type": "reduce_distractions",
                        "target": avg_distraction_daily - target_reduction,
                        "current": avg_distraction_daily
                    })
        
        # Sugerencias basadas en sesiones de concentración
        avg_session = df_graph['Duration'].mean()
        if avg_session < 20:
            suggested_goals.append({
                "name": "Mejorar Concentración",
                "description": "Aumentar sesiones promedio a 30+ minutos",
                "type": "focus_sessions",
                "target": 30,
                "current": avg_session
            })
        
        response = f"""🎯 **OBJETIVOS PERSONALIZADOS SUGERIDOS**

Basándome en tu actividad de los últimos {days} días, he identificado estas oportunidades de mejora:

"""
        
        for i, goal in enumerate(suggested_goals[:3], 1):  # Máximo 3 sugerencias
            progress_needed = goal['target'] - goal['current']
            response += f"""**{i}. {goal['name']} 📈**
• **Meta**: {goal['description']}
• **Situación actual**: {goal['current']:.0f} min/día
• **Objetivo**: {goal['target']:.0f} min/día
• **Mejora necesaria**: +{progress_needed:.0f} min/día

"""
        
        response += """## 🚀 **¿Cómo definir tu objetivo?**

**Responde con el número del objetivo que te interesa** (ej: "Objetivo 1") y te ayudaré a:
1. ✅ Activarlo en tu sistema de seguimiento
2. 📊 Configurar métricas específicas
3. 📅 Establecer plazos realistas
4. 💡 Darte estrategias específicas para lograrlo

**💪 ¿Cuál objetivo te motiva más para empezar?**

*También puedes decir "objetivo personalizado" si quieres crear uno diferente.*"""
        
        return response
    
    except Exception as e:
        return f"""🎯 **CREACIÓN DE OBJETIVOS**

He detectado algunos datos pero hay problemas para analizarlos completamente.

**Sugerencias generales que puedes empezar:**
1. 📊 **Objetivo de tiempo productivo**: Dedica 2-3 horas diarias a trabajo importante
2. ⚠️ **Reducir distracciones**: Limita redes sociales a 30 min/día
3. 🎯 **Mejorar concentración**: Practica sesiones de 25-45 minutos sin interrupciones

**Para objetivos más específicos:**
- Asegúrate de que tus datos tengan las columnas correctas
- Clasifica actividades con la Matriz de Eisenhower
- Vuelve a preguntarme para análisis personalizado

¿Te gustaría empezar con alguno de estos objetivos generales?"""