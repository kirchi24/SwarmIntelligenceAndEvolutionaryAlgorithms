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
        In dieser Arbeit wird die Optimierung der Hyperparameter eines Convolutional Neural Networks (CNN) für das Fashion-MNIST-Datenset durchgeführt. Für die Suche nach optimalen Architekturen und Parametern wird eine Kombination aus Differential Evolution (DE) und Hill Climbing (HC) verwendet.

        **Warum DE + HC?**
        - Differential Evolution ist ein robuster, einfach zu implementierender globaler Optimierer, der sowohl mit diskreten als auch kontinuierlichen Parametern umgehen kann. Er ist weniger empfindlich gegenüber Skalierung und benötigt keine Gradienteninformationen.
        - Hill Climbing ist ein schneller lokaler Optimierer, der gefundene Lösungen effizient weiter verbessern kann.
        - Die Kombination ermöglicht eine gute Balance zwischen globaler Exploration und lokaler Exploitation, was besonders bei komplexen Suchräumen wie CNN-Architekturen wichtig ist.

        **Warum nicht andere Algorithmen?**
        - Genetische Algorithmen (GA) sind zwar ebenfalls global, benötigen aber oft mehr Feintuning und sind langsamer.
        - Simulated Annealing (SA) ist gut für das Entkommen aus lokalen Minima, aber meist ineffizient in hochdimensionalen Räumen.
        - Ant Colony Optimization (ACO) ist für kombinatorische Probleme wie TSP konzipiert und weniger geeignet für gemischte Parameter.
        - Particle Swarm Optimization (PSO) funktioniert gut für kontinuierliche Parameter, hat aber Schwierigkeiten mit diskreten und stark eingeschränkten Suchräumen.

        Insgesamt bietet DE + HC eine praktische und effektive Lösung für die Hyperparameteroptimierung von CNNs in diesem Kontext.
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
