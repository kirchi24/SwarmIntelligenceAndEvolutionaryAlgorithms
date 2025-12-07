import streamlit as st

# ========================================================
# Streamlit Page Setup
# ========================================================

st.set_page_config(
    page_title="CNN Hyperparameter Tuning",
    layout="wide"
)

st.title("CNN Hyperparameter Tuning")


# ========================================================
# Tabs
# ========================================================

tabs = st.tabs(["Introduction", "Methods", "Results", "Discussion"])


# ========================================================
# TAB 1 - INTRODUCTION
# ========================================================
with tabs[0]:
    st.markdown("## Einführung")

    st.markdown(
        r"""
        empty page for introduction
        """
    )


# ========================================================
# TAB 2 - METHODS
# ========================================================
with tabs[1]:
    st.markdown("## Methoden")

    st.markdown(
        r"""
        Empty page for methods description
        """
)

# ========================================================
# TAB 3 - RESULTS - RUN DE + Animation
# ========================================================
with tabs[2]:

    st.markdown("## Differential Evolution - Simulation")
    st.sidebar.markdown("### Parameter konfigurieren")

    st.sidebar.markdown("#### DE-Parameter")
    K = st.sidebar.slider("K (Radienanzahl)", 10, 100, 20)

    if st.button("Diff Evol starten"):

        st.info("Optimierung läuft… das kann dauern.")

        # Fortschrittsanzeige
        progress = st.progress(0)
        status = st.empty()

        def progress_callback(current_iter, total_iters, best_fitness):
            progress.progress(current_iter / total_iters)
            status.text(
                f"Iteration {current_iter}/{total_iters} - aktueller bester Score: {best_fitness:.4f}"
            )


# ========================================================
# TAB 4 - DISCUSSION
# ========================================================
with tabs[3]:

    st.markdown("## Diskussion")

    st.markdown(
        r"""
        empty page for discussion
        """
    )
