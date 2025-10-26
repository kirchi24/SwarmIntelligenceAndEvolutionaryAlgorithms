import streamlit as st
import numpy as np
from PIL import Image
import os
from io import BytesIO

from src.GeneticAlgorithmVariations.chromosome import ImageChromosome
from src.GeneticAlgorithmVariations.population import Population
from src.GeneticAlgorithmVariations.image_utils import (
    load_and_resize_image,
    fitness_factory,
)

# -------------------------
# Streamlit Konfiguration
# -------------------------
st.set_page_config(
    page_title="Genetic Algorithm Variations",
    layout="wide",
)

st.title("Genetic Algorithm Variations - Image Reconstruction")

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs(["Introduction", "Methods", "Results", "Discussion"])

# =====================================================
# TAB 0: INTRODUCTION
# =====================================================
with tabs[0]:
    st.header("Introduction")
    st.markdown(
        """
        Diese App demonstriert einen einfachen genetischen Algorithmus zur **Bildrekonstruktion**.
        Ziel ist es, ein Population von kleinen Bildern (z.B. 16×16) so zu entwickeln, dass
        eines der Bilder möglichst nah an einem Zielbild liegt.
        """
    )

    # Zielbild anzeigen (ca. 400x400 px)
    default_target_path = os.path.join("src", "GeneticAlgorithmVariations", "data", "example_image.png")
    if os.path.exists(default_target_path):
        target_preview = Image.open(default_target_path).convert("L").resize((400, 400), Image.LANCZOS)
        st.image(target_preview, caption="Target image (preview ~400×400 px)", width=400)
    else:
        st.info(f"Default target image not found at '{default_target_path}'. Upload a file in Results tab to use a target.")

# =====================================================
# TAB 1: METHODS  (ausführliche Code-Erklärung)
# =====================================================
with tabs[1]:
    st.header("Methods — Klassen & Konzepte (ausführlich)")
    st.markdown(
        """
        Nachfolgend findest du eine ausführliche, aber konzentrierte Erklärung der wichtigsten Komponenten,
        so wie sie in deinem Projekt implementiert sind.

        ---
        ### 1) `ImageChromosome`
        - Repräsentiert ein *Individuum* / Chromosom: ein 2D-Array von Pixelwerten im Bereich `[0, 1]`.
        - **Kernattribute:**
          - `genes`: 2D `np.ndarray` mit Float-Werten ∈ [0,1] (das Bild).
          - `fitness`: numerischer Fitness-Wert (höher = besser).
          - `fitness_fn`: callable, das ein `genes`-Array bewertet.
          - `mutation_method`, `crossover_method`, `mutation_rate`, `mutation_width`, `alpha`.
        - **Initialisierung:**
          - `random`: vollständiges Rauschen (Uniform[0,1]).
          - `expert_knowledge`: glatteres Startbild (weiß mit dunklerem Zentrum + Rauschen).
        - **Mutationen:**
          - `uniform_local`: uniformer Störterm in einem Intervall um das aktuelle Gen.
          - `gaussian_adaptive`: normalverteilte Störung, deren Standardabweichung adaptiv skaliert wird anhand der Fitness (schwächere Individuen mutieren stärker).
        - **Crossover:**
          - `arithmetic`: gewichtete Mittelung der Elternpixel (`alpha` steuert Gewichtung).
          - `global_uniform`: für jedes Pixel Zufallsauswahl: Eltern A oder B.
        - **Methoden:**
          - `evaluate()`: ruft `fitness_fn(genes)` auf und speichert Ergebnis.
          - `mutate(max_fitness)`: ruft die gewählte Mutationsstrategie auf.
          - `crossover(other)`: erzeugt zwei Kinder-`ImageChromosome`.
          - `copy()`: tiefe Kopie (unabhängig).

        ---
        ### 2) `Population`
        - Verwaltet eine Liste von `ImageChromosome`-Objekten und führt die GA-Schritte aus:
          - **Evaluation** aller Individuen,
          - **Parent Selection** (z. B. `tournament` oder `rank`),
          - **Crossover** und **Mutation** zur Erzeugung von Nachkommen,
          - **Survivor Selection** (z. B. nach `fitness` oder `age`).
        - **Wichtige Punkte:**
          - `evaluate()` ruft `ind.evaluate()` für alle Individuen auf (Fehlerbehandlung setzt Fitness auf 0).
          - `_select_parents()` bietet Turnier- und Rang-basiertes Selektionsschema.
          - `_select_survivors()` ermöglicht Elimination/Überleben basierend auf Fitness oder Alter.
          - `evolve()` führt genau eine Generation aus (Eltern auswählen, Nachkommen erzeugen, mutieren, evaluieren, Survivors wählen).
        - **Tipps/Erweiterungen:**
          - Elitismus: Top-k direkt übernehmen, um Verlust guter Lösungen zu verhindern.
          - Adaptive Mutation: Mutation-Rate adaptiv verändern (z. B. je nach Varianz der Population).
          - Multi-Island: Mehrere Subpopulationen mit Migration.

        ---
        ### 3) `fitness_utils` (load & fitness_factory)
        - `load_and_resize_image(path, size)`: lädt Bild, konvertiert zu Graustufen, skaliert auf `size` und normalisiert auf [0,1].
        - `fitness_factory(target, method)`: liefert eine Fitness-Funktion:
          - `manhattan`: `1 - mean(|individual - target|)` (1.0 = perfekter Treffer)
          - `euclidean`: `1 - sqrt(mean((individual - target)^2))` (1.0 = perfekter Treffer)

        ---
        Diese Architektur trennt Verantwortlichkeiten sauber:
        - `ImageChromosome` kennt nur das einzelne Individuum,
        - `Population` orchestriert Evolution,
        - `fitness_utils` liefert Ziel & Bewertungsmaß.
        """
    )

# =====================================================
# TAB 2: RESULTS
# =====================================================
with tabs[2]:
    st.header("Results — Run & Visualize")

    # ----------------------------
    # Konfiguration (Parameter)
    # ----------------------------
    st.subheader("Configuration")
    col1, col2 = st.columns(2)

    with col1:
        pop_size = st.slider("Population size", 10, 100, 30, step=5)
        generations = st.slider("Generations", 10, 500, 100, step=10)
        mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.3, 0.05)
        mutation_width = st.slider("Mutation width", 0.0, 1.0, 0.2, 0.05)

    with col2:
        crossover_method = st.selectbox("Crossover method", ["arithmetic", "global_uniform"])
        mutation_method = st.selectbox("Mutation method", ["uniform_local", "gaussian_adaptive"])
        parent_selection = st.selectbox("Parent selection", ["tournament", "rank"])
        survivor_method = st.selectbox("Survivor selection", ["fitness", "age"])

    st.divider()

    # ----------------------------
    # Zielbild Upload / Default
    # ----------------------------
    uploaded_file = st.file_uploader("Upload your target image (optional)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("L")
        preview = img.resize((400, 400), Image.LANCZOS)
        st.image(preview, caption="Uploaded target (preview ~400×400 px)", width=400)
        # Für die GA: 16x16 normalisieren
        target = np.array(img.resize((16, 16), Image.LANCZOS), dtype=np.float32) / 255.0
    else:
        default_path = os.path.join("src", "GeneticAlgorithmVariations", "data", "example_image.png")
        if os.path.exists(default_path):
            img = Image.open(default_path).convert("L")
            preview = img.resize((400, 400), Image.LANCZOS)
            st.image(preview, caption="Default target (preview ~400×400 px)", width=400)
            target = load_and_resize_image(default_path, (16, 16))
        else:
            st.warning("Kein Standard-Target gefunden. Bitte lade ein Zielbild hoch.")
            target = np.ones((16, 16), dtype=np.float32) * 0.5

    fitness_method = st.radio("Fitness method", ["manhattan", "euclidean"], horizontal=True)
    fitness_fn = fitness_factory(target, method=fitness_method)

    # ----------------------------
    # Run GA Button
    # ----------------------------
    if st.button("Run Genetic Algorithm"):
        with st.spinner("Evolving population... please wait"):
            # Initial population
            pop = Population(
                size=pop_size,
                shape=(16, 16),
                initialization_method="random",
                parent_selection=parent_selection,
                survivor_method=survivor_method,
                fitness_fn=fitness_fn,
                mutation_method=mutation_method,
                mutation_rate=mutation_rate,
                mutation_width=mutation_width,
                crossover_method=crossover_method,
                alpha=0.5,
            )

            fitness_history = []
            best_images = []

            for gen in range(generations):
                pop.evolve()
                best = pop.best()
                fitness_history.append(best.fitness)
                # snapshots (z. B. alle 10 Generationen)
                if gen % 10 == 0 or gen == generations - 1:
                    best_images.append((gen, best.genes.copy()))

            st.session_state["fitness_history"] = fitness_history
            st.session_state["best_images"] = best_images
            st.success("Evolution complete!")

    # ----------------------------
    # Ergebnisse anzeigen (falls vorhanden)
    # ----------------------------
    if "fitness_history" in st.session_state:
        st.subheader("Fitness progression")
        fitness_history = st.session_state["fitness_history"]
        # Line chart – Streamlit akzeptiert eine Liste/Numpy-Array
        st.line_chart(fitness_history)

        st.subheader("Evolution snapshots")
        best_images = st.session_state["best_images"]
        if best_images:
            n = len(best_images)
            cols = st.columns(min(6, n))
            for i, (gen, genes) in enumerate(best_images):
                img_arr = (genes * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_arr)
                # kleine Vorschau für Snapshots (z.B. 150 px breit)
                cols[i % len(cols)].image(img_pil, caption=f"Gen {gen}", width=150)

        # Final best image / Download
        final_gen, final_genes = best_images[-1]
        final_img = Image.fromarray((final_genes * 255).astype(np.uint8))
        buf = BytesIO()
        final_img.save(buf, format="PNG")
        st.download_button("⬇️ Download final image", buf.getvalue(), file_name="best_evolved.png", mime="image/png")
    else:
        st.info("Führe zuerst den Algorithmus im oberen Bereich dieses Tabs aus (Run Genetic Algorithm).")

# =====================================================
# TAB 3: DISCUSSION
# =====================================================
with tabs[3]:
    st.header("Discussion")
    st.markdown(
        """
        Hier kannst du Ergebnisse interpretieren:
        - Wie schnell konvergiert die Fitness?
        - Was bewirken unterschiedliche Mutationsstrategien?
        - Welche Einstellungen produzieren stabilere Resultate?

        Ideen zur Erweiterung:
        - RGB-Unterstützung (3-Kanal-Chromosomen)
        - Elitismus hinzufügen
        - Adaptive Mutation/Selektion
        - Mehrere Inseln (Multi-population) mit Migration
        """
    )

