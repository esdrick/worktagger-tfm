import streamlit as st
import pandas as pd
from config.constants import EISEN_OPTIONS

def show_productivity_recommendations():
    # --- Análisis de hábitos improductivos ---
    st.divider()
    st.markdown("### 📉 Análisis de hábitos improductivos")
    df = st.session_state.df_original
    tiempo_iv = df[df["Eisenhower"] == EISEN_OPTIONS[3]]["Duration"].sum() / 60  # en minutos

    if tiempo_iv >= 60:
        st.warning(f"⛔ Has dedicado {tiempo_iv:.0f} minutos a actividades improductivas en el periodo analizado. Considera reducirlas para mejorar tu foco.")
    elif tiempo_iv > 0:
        st.info(f"🕒 Tiempo en actividades poco importantes: {tiempo_iv:.0f} minutos. ¡Bien! Pero aún se puede optimizar.")
    else:
        st.success("🎯 No se han registrado actividades improductivas en el periodo analizado. ¡Excelente trabajo!")

    st.divider()
    st.markdown("### 🎯 Objetivo y análisis por cuadrante")

    comentarios = {
        EISEN_OPTIONS[0]: "🔴 Estas tareas son críticas. Bien si cumpliste tu meta, pero cuida no sobrecargarte.",
        EISEN_OPTIONS[1]: "🟠 Son las que previenen crisis futuras. Si dedicaste poco tiempo, intenta planificarlas mejor.",
        EISEN_OPTIONS[2]: "🔵 Tareas que podrías delegar. Evalúa si puedes automatizarlas o pedir apoyo.",
        EISEN_OPTIONS[3]: "🟢 No aportan valor. Intenta reducir o eliminar estas actividades.",
    }

    # Inputs de objetivos por cuadrante
    objetivos_por_cuadrante = {}
    col_I, col_II, col_III, col_IV = st.columns(4)
    with col_I:
        objetivos_por_cuadrante[EISEN_OPTIONS[0]] = st.number_input("🔴 Urgente e Importante (min)", min_value=0, step=10, value=120, key="obj_I")
    with col_II:
        objetivos_por_cuadrante[EISEN_OPTIONS[1]] = st.number_input("🟠 Importante no urgente (min)", min_value=0, step=10, value=120, key="obj_II")
    with col_III:
        objetivos_por_cuadrante[EISEN_OPTIONS[2]] = st.number_input("🔵 Urgente no importante (min)", min_value=0, step=10, value=30, key="obj_III")
    with col_IV:
        objetivos_por_cuadrante[EISEN_OPTIONS[3]] = st.number_input("🟢 No urgente ni importante (min)", min_value=0, step=10, value=0, key="obj_IV")

    # Cálculo de duración real por cuadrante (totales globales)
    df = st.session_state.df_original.copy()
    df['Duration_min'] = df['Duration'] / 60
    tiempo_cuadrante = (
        df[df['Eisenhower'].notna()]
        .groupby('Eisenhower')['Duration_min']
        .sum()
        .to_dict()
    )

    # Feedback unificado por cuadrante
    st.markdown("#### 🧠 Resultado por cuadrante")
    for cuadrante, objetivo in objetivos_por_cuadrante.items():
        tiempo = round(tiempo_cuadrante.get(cuadrante, 0), 1)
        comentario = comentarios.get(cuadrante, "")
        
        st.markdown(f"**{cuadrante}**")
        st.markdown(f"- ⏱️ Objetivo: **{objetivo} min** | Real: **{tiempo:.0f} min**")

        if objetivo == 0:
            if tiempo > 0:
                st.error(f"❌ Has dedicado tiempo a {cuadrante} cuando el objetivo era 0.")
            else:
                st.success(f"✅ No has dedicado tiempo a {cuadrante}.")
        else:
            if tiempo >= objetivo:
                st.success(f"✅ Has cumplido tu meta para {cuadrante} ({tiempo:.0f} / {objetivo} min).")
            elif tiempo >= 0.8 * objetivo:
                st.info(f"🟡 Casi logras tu meta para {cuadrante} ({tiempo:.0f} / {objetivo} min).")
            else:
                st.warning(f"⚠️ Aún puedes dedicar más tiempo a {cuadrante} ({tiempo:.0f} / {objetivo} min).")

        st.markdown(f"- 💬 {comentario}")
        st.markdown("---")

    # --- Sugerencia de bloqueo temporal de apps improductivas ---
    st.divider()
    st.markdown("### 🚫 Sugerencia de bloqueo temporal de apps improductivas")

    df = st.session_state.df_original.copy()
    df['Duration_min'] = df['Duration'] / 60

    # Solo cuadrantes III y IV
    df_valid = df[df["Eisenhower"].isin([EISEN_OPTIONS[2], EISEN_OPTIONS[3]])]
    apps_info = (
        df_valid.groupby(["App", "Eisenhower"])["Duration_min"]
        .sum()
        .reset_index()
        .sort_values("Duration_min", ascending=False)
    )

    if apps_info.empty:
        st.success("🎯 No se detectaron apps improductivas en este periodo.")
    else:
        st.markdown("""
<style>
.app-card {
    padding: 0.6em 1em;
    margin-bottom: 0.5em;
    border-radius: 8px;
    border: 1px solid #ddd;
    background-color: #f9f9f9;
}
.app-card strong {
    font-size: 1.05em;
}
.app-card .cuadrante {
    float: right;
    font-size: 0.85em;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

        for _, row in apps_info.iterrows():
            app = row["App"]
            dur = row["Duration_min"]
            cuadrante = row["Eisenhower"]

            cuadrante_tag = f"<span class='cuadrante'>🧭 {cuadrante}</span>"

            if cuadrante == EISEN_OPTIONS[3]:
                if dur >= 30:
                    color = "#fff3cd"
                    recomendacion = f"⛔ <strong>{app}</strong> — {dur:.0f} min. {cuadrante_tag}<br>Considera bloquearla temporalmente para evitar distracciones."
                elif dur >= 10:
                    color = "#e2e3e5"
                    recomendacion = f"🔕 <strong>{app}</strong> — {dur:.0f} min. {cuadrante_tag}<br>Puedes reducir su uso si no es esencial."
                else:
                    color = "#d1e7dd"
                    recomendacion = f"✅ <strong>{app}</strong> — {dur:.0f} min. {cuadrante_tag}<br>Uso mínimo. No se sugiere bloqueo."
            elif cuadrante == EISEN_OPTIONS[2]:
                if dur >= 30:
                    color = "#fff3cd"
                    recomendacion = f"⚠️ <strong>{app}</strong> — {dur:.0f} min. {cuadrante_tag}<br>Evalúa si estas tareas pueden delegarse o automatizarse."
                elif dur >= 10:
                    color = "#e2e3e5"
                    recomendacion = f"📌 <strong>{app}</strong> — {dur:.0f} min. {cuadrante_tag}<br>Puedes replantear su necesidad o frecuencia."
                else:
                    color = "#d1e7dd"
                    recomendacion = f"✅ <strong>{app}</strong> — {dur:.0f} min. {cuadrante_tag}<br>Sin alerta de uso."
            else:
                continue

            st.markdown(f"<div class='app-card' style='background-color:{color}'>{recomendacion}</div>", unsafe_allow_html=True)