import streamlit as st

# Streamlit config
st.set_page_config(
    page_title="Ant Colony Optimization - Pheromone Update",
    layout="wide",
)

st.title("Pheromon-Aktualisierung in der Ameisenkolonieoptimierung")

# Abschnitt: Mathematische Formulierung
st.markdown("### 🧠 Mathematische Formulierung")
st.markdown("Für jede Ameise k gilt:")

st.latex(r"\tau_{n,d,s} \leftarrow (1 - \rho)\,\tau_{n,d,s} + \sum_k \Delta\tau_{n,d,s}^{(k)}")

st.markdown("mit:")

st.latex(
    r"""
\Delta\tau_{n,d,s}^{(k)} =
\begin{cases}
\dfrac{Q}{1 + L_k}, & \text{wenn Ameise k dort eine 1 (Zuweisung) gesetzt hat} \\
0, & \text{sonst}
\end{cases}
"""
)

# Abschnitt: Parameter
st.markdown("---")
st.markdown("**Parameter:**")
st.markdown(
    """
- ρ — *Verdunstungsrate* (z. B. 0,1 → 10 % Verlust pro Iteration)  
- Q — *Verstärkungsfaktor*, skaliert die Menge des abgelegten Pheromons  
- Lₖ — *Kosten / Score* der Ameise k (je kleiner, desto besser)
"""
)

# Abschnitt: Intuition
st.markdown("---")
st.markdown("**Intuition:**")
st.markdown(
    """
- Schlechte Lösungen verdunsten mit der Zeit: τ ← (1-ρ)·τ  
- Gute Lösungen verstärken ihre Pfade: τ ← τ + Q / (1 + Lₖ)  
- Dadurch stellt sich ein Gleichgewicht zwischen *Vergessen* und *Lernen* ein.
"""
)
