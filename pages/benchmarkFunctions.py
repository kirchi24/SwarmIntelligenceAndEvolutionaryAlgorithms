import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.benchmark import (
    quadratic,
    sinusoidal,
    ackley,
    rosenbrock,
    rastrigin,
    visualize_1d_input,
    visualize_2d_input_surface,
)

st.set_page_config(page_title="Function Explorer", layout="wide")

st.title("Function Explorer")
st.markdown(
    """
Explore benchmark functions and their properties.  
All functions are **vectorized** for batch evaluation.
"""
)

st.sidebar.header("Select Functions")

one_d_options = {
    "Quadratic": quadratic,
    "Sinusoidal": sinusoidal,
}
one_d_choice = st.sidebar.selectbox(
    "1D Function (input dimension = 1)", list(one_d_options.keys())
)
one_d_func = one_d_options[one_d_choice]

two_d_options = {
    "Ackley": ackley,
    "Rosenbrock": rosenbrock,
    "Rastrigin": rastrigin,
}
two_d_choice = st.sidebar.selectbox(
    "2D Function (input dimension = 2)", list(two_d_options.keys())
)
two_d_func = two_d_options[two_d_choice]

st.subheader(f"1D Function (input dimension = 1): {one_d_choice}")
fig_1d = visualize_1d_input(one_d_func, name=one_d_choice, xlim=(-5, 5))
st.plotly_chart(fig_1d, use_container_width=True)

st.subheader(f"2D Function (visualized as 3D surface): {two_d_choice}")
fig_3d = visualize_2d_input_surface(
    two_d_func, name=two_d_choice, xlim=(-5, 5), ylim=(-5, 5), points=150
)
st.plotly_chart(fig_3d, use_container_width=True)

st.subheader("Implementation Highlights")
st.markdown(
    """
- Functions are **vectorized** and support batch evaluation.  
- 1D input functions are plotted interactively using Plotly lines.  
- 2D input functions are visualized as interactive 3D surfaces.  
- Functions include common benchmark functions: Quadratic, Sinusoidal, Ackley, Rosenbrock, Rastrigin.
"""
)

st.markdown("---")
st.markdown(
    "<small>Note: This documentation was generated with the assistance of ChatGPT.</small>",
    unsafe_allow_html=True,
)
