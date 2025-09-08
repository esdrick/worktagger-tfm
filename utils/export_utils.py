# export_utils.py
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import pandas as pd
from datetime import datetime
import streamlit as st

def generate_pdf_report():
    """Genera PDF completo del análisis"""
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Añadir contenido
    story = add_title_page(story, styles)
    story = add_summary_section(story, styles)
    story = add_eisenhower_section(story, styles)
    story = add_activity_dashboard_section(story, styles)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def add_title_page(story, styles):
    """Página de título"""
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#003366'),
        alignment=1  # Center
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#666666'),
        alignment=1,
        spaceAfter=20
    )
    
    story.append(Paragraph("Productivity Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
    story.append(Spacer(1, 40))
    
    return story

def add_summary_section(story, styles):
    """Sección de resumen ejecutivo"""
    df = st.session_state.df_original
    
    # Título de sección
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#003366'),
        spaceAfter=20
    )
    
    story.append(Paragraph("Executive Summary", section_title))
    
    # Calcular métricas básicas
    total_activities = len(df)
    classified_activities = df['Eisenhower'].notna().sum() if 'Eisenhower' in df.columns else 0
    total_time_hours = df['Duration'].sum() / 3600 if 'Duration' in df.columns else 0
    total_time_minutes = df['Duration'].sum() / 60 if 'Duration' in df.columns else 0
    classification_rate = (classified_activities/total_activities*100) if total_activities > 0 else 0
    
    # Análisis de fechas
    if 'Begin' in df.columns:
        df['Date'] = pd.to_datetime(df['Begin']).dt.date
        date_range = f"{df['Date'].min()} to {df['Date'].max()}"
        unique_days = df['Date'].nunique()
    else:
        date_range = "Not available"
        unique_days = "N/A"
    
    # Tabla de métricas principales
    data = [
        ['Metric', 'Value'],
        ['Analysis Period', date_range],
        ['Days Analyzed', str(unique_days)],
        ['Total Activities', str(total_activities)],
        ['Classified Activities', str(classified_activities)],
        ['Classification Rate', f"{classification_rate:.1f}%"],
        ['Total Time Analyzed', f"{total_time_hours:.1f} hours ({total_time_minutes:.0f} minutes)"],
    ]
    
    table = Table(data, colWidths=[3*inch, 2.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 30))
    
    return story

def add_eisenhower_section(story, styles):
    """Sección de análisis de matriz de Eisenhower"""
    df = st.session_state.df_original
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#003366'),
        spaceAfter=20
    )
    
    story.append(Paragraph("Eisenhower Matrix Analysis", section_title))
    
    if 'Eisenhower' not in df.columns or df['Eisenhower'].notna().sum() == 0:
        story.append(Paragraph("No activities have been classified using the Eisenhower Matrix yet.", styles['Normal']))
        story.append(Spacer(1, 20))
        return story
    
    # Calcular distribución por cuadrante
    df_classified = df[df['Eisenhower'].notna()]
    time_by_quadrant = df_classified.groupby('Eisenhower')['Duration'].sum() / 60  # minutos
    count_by_quadrant = df_classified.groupby('Eisenhower').size()
    
    # POR ESTE MAPEO CORRECTO:
    quadrant_descriptions = {
        'I: Urgent & Important': 'Q1: Urgent & Important (DO)',
        'II: Not urgent but Important': 'Q2: Not Urgent & Important (SCHEDULE)',
        'III: Urgent but Not important': 'Q3: Urgent & Not Important (DELEGATE)',
        'IV: Not urgent & Not important': 'Q4: Not Urgent & Not Important (ELIMINATE)'
    }
    
    # Tabla de distribución
    quadrant_data = [['Quadrant', 'Activities', 'Time (min)', 'Time (hours)', 'Percentage']]
    total_time = time_by_quadrant.sum()
    
    # Y cambiar el bucle por:
    for quadrant in ['I: Urgent & Important', 'II: Not urgent but Important', 'III: Urgent but Not important', 'IV: Not urgent & Not important']:
        time_min = time_by_quadrant.get(quadrant, 0)
        time_hours = time_min / 60
        count = count_by_quadrant.get(quadrant, 0)
        percentage = (time_min / total_time * 100) if total_time > 0 else 0
        
        quadrant_data.append([
            quadrant_descriptions.get(quadrant, quadrant),
            str(count),
            f"{time_min:.0f}",
            f"{time_hours:.1f}",
            f"{percentage:.1f}%"
        ])
    
    # Fila de totales
    quadrant_data.append([
        'TOTAL',
        str(count_by_quadrant.sum()),
        f"{total_time:.0f}",
        f"{total_time/60:.1f}",
        "100.0%"
    ])
    
    table = Table(quadrant_data, colWidths=[3.2*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor('#F8F9FA')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#E9ECEF')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Análisis de productividad
    if total_time > 0:
        productive_time = time_by_quadrant.get('I: Urgent & Important', 0) + time_by_quadrant.get('II: Not urgent but Important', 0)
        unproductive_time = time_by_quadrant.get('IV: Not urgent & Not important', 0)
        efficiency = (productive_time / total_time * 100) if total_time > 0 else 0  # AÑADIR ESTA LÍNEA
        
        productivity_text = f"""
        <b>Productivity Analysis:</b><br/>
        • Productive time (Q1 + Q2): {productive_time:.0f} minutes ({productive_time/60:.1f} hours)<br/>
        • Unproductive time (Q4): {unproductive_time:.0f} minutes ({unproductive_time/60:.1f} hours)<br/>
        • Efficiency rate: {efficiency:.1f}%<br/>
        """
        
        story.append(Paragraph(productivity_text, styles['Normal']))
    
    story.append(Spacer(1, 30))
    
    return story

def add_activity_dashboard_section(story, styles):
    """Sección del dashboard de actividades"""
    df = st.session_state.df_original
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#003366'),
        spaceAfter=20
    )
    
    story.append(Paragraph("Activity Dashboard", section_title))
    
    # Verificar si hay datos de subactividades
    if 'Subactivity' not in df.columns or df['Subactivity'].notna().sum() == 0:
        story.append(Paragraph("No subactivities have been classified yet.", styles['Normal']))
        story.append(Spacer(1, 20))
        return story
    
    # Análisis de subactividades
    df_subact = df[df['Subactivity'].notna()]
    subact_summary = df_subact.groupby('Subactivity').agg({
        'Duration': 'sum',
        'Activity': 'count'
    }).reset_index()
    
    subact_summary['Duration_minutes'] = subact_summary['Duration'] / 60
    subact_summary['Duration_hours'] = subact_summary['Duration'] / 3600
    subact_summary = subact_summary.sort_values('Duration_minutes', ascending=False)
    
    # Top 10 subactividades
    top_subactivities = subact_summary.head(10)
    
    # Tabla de top subactividades
    subact_data = [['Subactivity', 'Occurrences', 'Time (min)', 'Time (hours)', 'Percentage']]
    total_subact_time = subact_summary['Duration_minutes'].sum()
    
    for _, row in top_subactivities.iterrows():
        percentage = (row['Duration_minutes'] / total_subact_time * 100) if total_subact_time > 0 else 0
        subact_data.append([
            str(row['Subactivity'])[:30] + "..." if len(str(row['Subactivity'])) > 30 else str(row['Subactivity']),
            str(row['Activity']),
            f"{row['Duration_minutes']:.0f}",
            f"{row['Duration_hours']:.1f}",
            f"{percentage:.1f}%"
        ])
    
    table = Table(subact_data, colWidths=[2.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Align subactivity names to left
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(Paragraph("Top Subactivities by Time", styles['Heading3']))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Insights adicionales
    if 'Begin' in df.columns:
        df_subact['Hour'] = pd.to_datetime(df_subact['Begin']).dt.hour
        peak_hour = df_subact.groupby('Hour')['Duration'].sum().idxmax()
        
        insights_text = f"""
        <b>Key Insights:</b><br/>
        • Total subactivities tracked: {len(subact_summary)}<br/>
        • Most active hour: {peak_hour}:00<br/>
        • Top subactivity: {top_subactivities.iloc[0]['Subactivity']} ({top_subactivities.iloc[0]['Duration_hours']:.1f}h)<br/>
        • Average time per subactivity: {total_subact_time/len(subact_summary):.0f} minutes
        """
        
        story.append(Paragraph(insights_text, styles['Normal']))
    
    story.append(Spacer(1, 30))
    
    return story

def export_chatbot_conversation():
    """Exporta conversación del chatbot"""
    if 'chat_history' not in st.session_state or not st.session_state.chat_history:
        return None
    
    conversation = "# Productivity Assistant Conversation\n\n"
    conversation += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    conversation += "---\n\n"
    
    for i, msg in enumerate(st.session_state.chat_history, 1):
        if msg["role"] == "user":
            conversation += f"## Message {i} - You\n\n"
            conversation += f"{msg['content']}\n\n"
        else:
            conversation += f"## Message {i} - Assistant\n\n"
            conversation += f"{msg['content']}\n\n"
        
        conversation += "---\n\n"
    
    conversation += f"\n\n*Export generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}*"
    
    return conversation