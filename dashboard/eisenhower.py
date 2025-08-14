import matplotlib.pyplot as plt
import streamlit as st
from config.constants import EISEN_OPTIONS

def truncate(text, max_len=40):
    return text if len(text) <= max_len else text[:max_len - 3] + '...'

def plot_eisenhower_matrix(df, max_items=15):
    """
    Pretty Eisenhower matrix with coloured quadrants and a bullet list of the
    most relevantes sub‑activities (and total minutes) inside each quadrant.
    """
    df_filtered = df[df["Eisenhower"].notna()]
    if df_filtered.empty:
        st.info("No hay subactividades etiquetadas con la matriz de Eisenhower.")
        return
    # minutes per sub‑activity & quadrant
    summary = (
        df_filtered
        .groupby(["Eisenhower", "Subactivity"])["Duration"]
        .sum()
        .div(60)                # seconds → minutes
        .reset_index()
        .sort_values("Duration", ascending=False)
    )
    quad_order = [
        "I: Urgente & Importante",
        "II: No urgente pero Importante",
        "III: Urgente pero No importante",
        "IV: No urgente & No importante",
    ]
    quad_pos = {
        quad_order[0]: (0, 1),
        quad_order[1]: (1, 1),
        quad_order[2]: (0, 0),
        quad_order[3]: (1, 0),
    }
    quad_color = {
        quad_order[0]: "#cf4d4dff",  # red
        quad_order[1]: "#f37100ff",  # orange
        quad_order[2]: "#0083e0ff",  # blue
        quad_order[3]: "#00ba00ff",  # green
    }
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    for quad, (x, y) in quad_pos.items():
        # coloured rectangle
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=quad_color[quad], alpha=1.0))
        # title
        ax.text(
            x + 0.02,
            y + 0.98,
            quad,
            fontsize=8,
            weight="bold",
            color="white",
            ha="left",
            va="top",
            wrap=True,
        )
        # bullet list of sub‑activities
        items = summary[summary["Eisenhower"] == quad].head(max_items)
        if items.empty:
            body = "—"
        else:
            body = "\n".join(
                f"• {truncate(str(row.Subactivity), 32)} ({row.Duration:.0f} min)"
                for _, row in items.iterrows()
            )
        ax.text(
            x + 0.02,
            y + 0.85,
            body,
            fontsize=6,
            weight="light",
            color="white",
            ha="left",
            va="top",
            wrap=True
        )
        # Tiempo total dedicado al cuadrante
        total_min = summary[summary["Eisenhower"] == quad]["Duration"].sum()
        ax.text(
            x + 0.02,
            y + 0.05,
            f"Total: {total_min:.0f} min",
            weight="bold",
            fontsize=8,
            color="white",
            ha="left",
            va="bottom",
        )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.set_title(" ⊹ Matriz de Eisenhower ", fontsize=10,weight="bold", pad=10, color="#003860ff")
    st.pyplot(fig)
