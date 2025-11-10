import streamlit as st

# Streamlit-Konfiguration
st.set_page_config(
    page_title="Ant Colony Optimization (ACO) - Nurse Scheduling",
    layout="wide",
)

st.title("Ant Colony Optimization (ACO) für das Nurse Scheduling Problem")

# Tabs erstellen
tabs = st.tabs(["Introduction", "Methods", "Results", "Discussion"])

# ---------------------------
# Tab 1: Introduction
# ---------------------------
with tabs[0]:
    st.markdown("## Einführung")
    st.markdown(
    """
    Ant Colony Optimization (ACO) ist eine metaheuristische Methode, inspiriert vom Nahrungssucheverhalten realer Ameisen. 
    Sie eignet sich besonders für kombinatorische Optimierungsprobleme, z. B. das Travelling Salesman Problem (TSP) 
    oder das Nurse Scheduling Problem (NSP).

    **Stärken:**
    - Kann komplexe, multi-constraint Probleme lösen  
    - Flexibel bei harten und weichen Einschränkungen  

    **Schwächen:**
    - Parameterwahl (ρ, Q, Anzahl Ameisen) kann die Lösung stark beeinflussen  
    - Berechnungsintensiv bei großen Problemgrößen
    """
    )

# ---------------------------
# Tab 2: Methods
# ---------------------------
with tabs[1]:
    st.markdown("## Methoden / Implementierung")
    st.markdown("### Pheromon-Aktualisierung")
    st.markdown(
        "Für jede Ameise k wird die Pheromonmatrix $\\tau_{n,d,s}$ wie folgt aktualisiert:"
    )

    st.latex(r"\tau_{n,d,s} \leftarrow (1 - \rho)\,\tau_{n,d,s} + \sum_k \Delta\tau_{n,d,s}^{(k)}")

    st.markdown("Dabei gilt:")
    st.latex(
        r"""
    \Delta\tau_{n,d,s}^{(k)} =
    \begin{cases}
    \dfrac{Q}{1 + L_k}, & \text{wenn Ameise k die Zuweisung gesetzt hat} \\
    0, & \text{sonst}
    \end{cases}
    """
    )

    st.markdown("**Parameter:**")
    st.markdown(
        """
    - ρ — Verdunstungsrate (z. B. 0,1 → 10 % Verlust pro Iteration)  
    - Q — Verstärkungsfaktor für abgelegtes Pheromon  
    - Lₖ — Kosten / Score der Ameise k (je kleiner, desto besser)
    """
    )

    st.markdown("**Intuition:**")
    st.markdown(
    """
    - Schlechte Lösungen verlieren Gewicht: $ \\tau \leftarrow (1-\\rho)\, \\tau $
    - Gute Lösungen verstärken Pfade: $\\tau \leftarrow \\tau + \dfrac{Q}{1 + L_k}$  
    - Gleichgewicht zwischen Vergessen und Lernen fokussiert den Algorithmus auf gute Lösungen
    """
    )

# ---------------------------
# Tab 3: Results
# ---------------------------
with tabs[2]:
    st.markdown("## Ergebnisse")
    st.markdown(
        """
    Hier können die Ergebnisse der ACO-Optimierung angezeigt werden, z. B.:
    - Optimaler Dienstplan für 10 Krankenschwestern, 7 Tage, 3 Schichten pro Tag  
    - Visualisierung der Pheromonmatrix  
    - Vergleich unterschiedlicher Parameter (ρ, Q, Anzahl Ameisen)
    """
    )

# ---------------------------
# Tab 4: Discussion
# ---------------------------
with tabs[3]:
    st.markdown("## Diskussion")
    st.markdown(
        """
    - Analyse der Lösungsqualität und Effizienz  
    - Vergleich erwarteter vs. unerwarteter Ergebnisse  
    - Limitationen der Implementierung  
    - Mögliche Verbesserungen (z. B. dynamische Heuristik, adaptive Parameterwahl)  
    - Komplexität bei größeren Problemgrößen
"""
    )


