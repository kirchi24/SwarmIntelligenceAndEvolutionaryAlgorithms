import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
from io import BytesIO
import altair as alt

from src.GeneticAlgorithmVariations.chromosome import ImageChromosome
from src.GeneticAlgorithmVariations.population import Population
from src.GeneticAlgorithmVariations.image_utils import (
    load_and_resize_image,
    fitness_factory,
)

# Streamlit config
st.set_page_config(
    page_title="Genetic Algorithm - Image Reconstruction",
    layout="wide",
)

# ---------------------------
# Sprachumschaltung (EN / DE)
# ---------------------------
if "lang" not in st.session_state:
    st.session_state["lang"] = "DE"

col_lang1, col_lang2 = st.columns([1, 6])
with col_lang1:
    if st.button("EN English" if st.session_state["lang"] == "DE" else "DE Deutsch"):
        st.session_state["lang"] = "EN" if st.session_state["lang"] == "DE" else "DE"

lang = st.session_state["lang"]

# Textdictionary 
T = {
    "DE": {
        "title": "Genetischer Algorithmus - Bildrekonstruktion",
        "intro": """
        Diese App demonstriert einen **genetischen Algorithmus zur Bildrekonstruktion**.
        Ziel ist es, eine Population kleiner Graustufenbilder (z. B. 16x16) so zu entwickeln,
        dass eines möglichst gut einem Zielbild entspricht.
        """,
        "tabs": ["Einführung", "Methoden", "Ergebnisse", "Diskussion"],
        "config": "Konfiguration",
        "upload": "Lade ein Zielbild hoch (optional)",
        "run": " Algorithmus starten",
        "fitprog": "Fitness-Verlauf",
        "evo": "Evolution-Schnappschüsse",
        "download": "Bestes Bild herunterladen",
        "init_method": "Initialisierungsmethode",
        "stopcrit": "Abbruchkriterium",
        "generations": "Anzahl Generationen",
        "fitness": "Nach Fitness-Verbesserung",
        "intro_warn": "Standardbild nicht gefunden. Bitte lade ein Zielbild hoch.",
        "pop_size": "Populationsgröße",
        "crossover_alpha": "Crossover-Alpha",
        "mutation_rate": "Mutationsrate",
        "mutation_width": "Mutationsbreite",
        "shape": "Form",
        "crossover_method": "Crossover-Methode",
        "mutation_method": "Mutationsmethode",
        "parent_selection": "Elternauswahl",
        "survivor_method": "Überlebensmethode",
        "fitness_method": "Fitnessmethode",
    },
    "EN": {
        "title": "Genetic Algorithm - Image Reconstruction",
        "intro": """
        This app demonstrates a **genetic algorithm for image reconstruction**.
        The goal is to evolve a population of small grayscale images (e.g., 16x16)
        so that one of them closely matches a given target image.
        """,
        "tabs": ["Introduction", "Methods", "Results", "Discussion"],
        "config": "Configuration",
        "upload": "Upload a target image (optional)",
        "run": "Run Genetic Algorithm",
        "fitprog": "Fitness progression",
        "evo": "Evolution snapshots",
        "download": " Download best image",
        "init_method": "Initialization method",
        "stopcrit": "Stopping criterion",
        "generations": "By generation count",
        "fitness": "By fitness improvement",
        "intro_warn": "Default image not found. Please upload a target image.",
        "pop_size": "Population size",
        "crossover_alpha": "Crossover alpha",
        "mutation_rate": "Mutation rate",
        "mutation_width": "Mutation width",
        "shape": "Shape",
        "crossover_method": "Crossover method",
        "mutation_method": "Mutation method",
        "parent_selection": "Parent selection",
        "survivor_method": "Survivor selection",
        "fitness_method": "Fitness method",
    },
}

st.title(T[lang]["title"])

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs(T[lang]["tabs"])

# =====================================================
# TAB 0: INTRODUCTION
# =====================================================
with tabs[0]:
    st.markdown(T[lang]["intro"])

    default_target_path = os.path.join("src", "GeneticAlgorithmVariations", "data", "example_image.png")
    if os.path.exists(default_target_path):
        target_preview = Image.open(default_target_path).convert("L").resize((400, 400), Image.LANCZOS)
        st.image(target_preview, caption="Target image (~400x400 px)", width=400)
    else:
        st.warning(T[lang]["intro_warn"])

# =====================================================
# TAB 1: METHODS
# =====================================================
with tabs[1]:
    if lang == "DE":
        st.header("Methods — Klassen & Konzepte")
        st.markdown(
            """
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
            ### 3) `image_utils`
            - `load_and_resize_image(path, size)`: lädt Bild, konvertiert zu Graustufen, skaliert auf `size` und normalisiert auf [0,1].
            - `fitness_factory(target, method)`: liefert eine Fitness-Funktion:
            - `manhattan`: `1 - mean(|individual - target|)` (1.0 = perfekter Treffer)
            - `euclidean`: `1 - sqrt(mean((individual - target)^2))` (1.0 = perfekter Treffer)

            """
        )
    else:
        st.header("Methods — Classes & Concepts")
        st.markdown(
            """
             ---
            ### 1) `ImageChromosome`
            - Represents a single *individual* / chromosome: a 2D array of pixel values in `[0, 1]`.
            - **Core attributes:**
            - `genes`: 2D `np.ndarray` of floats ∈ [0,1] (the image).
            - `fitness`: numerical fitness value (higher = better).
            - `fitness_fn`: callable that evaluates the `genes` array.
            - `mutation_method`, `crossover_method`, `mutation_rate`, `mutation_width`, `alpha`.
            - **Initialization:**
            - `random`: full uniform noise in [0,1].
            - `expert_knowledge`: smoother starting image (mostly white with darker center + noise).
            - **Mutations:**
            - `uniform_local`: uniform perturbation around current gene values.
            - `gaussian_adaptive`: Gaussian perturbation scaled adaptively by fitness (weaker individuals mutate more).
            - **Crossover:**
            - `arithmetic`: weighted average of parent pixels (`alpha` controls weighting).
            - `global_uniform`: each pixel randomly taken from one of the parents.
            - **Methods:**
            - `evaluate()`: calls `fitness_fn(genes)` and stores the result.
            - `mutate(max_fitness)`: applies the chosen mutation strategy.
            - `crossover(other)`: produces two child `ImageChromosome` instances.
            - `copy()`: deep, independent copy.

            ---
            ### 2) `Population`
            - Manages a list of `ImageChromosome` objects and executes GA steps:
            - **Evaluation** of all individuals,
            - **Parent selection** (e.g., `tournament` or `rank`),
            - **Crossover** and **mutation** to generate offspring,
            - **Survivor selection** (e.g., by `fitness` or `age`).
            - **Key points:**
            - `evaluate()` calls `ind.evaluate()` for all individuals (error handling sets fitness to 0 on failure).
            - `_select_parents()` provides tournament or rank-based selection.
            - `_select_survivors()` allows elimination/survival based on fitness or age.
            - `evolve()` executes exactly one generation (select parents, produce offspring, mutate, evaluate, select survivors).
            - **Tips / Extensions:**
            - Elitism: directly retain top-k to prevent loss of good solutions.
            - Adaptive mutation: adjust mutation rate based on population variance.
            - Multi-Island: multiple subpopulations with migration.

            ---
            ### 3) `image_utils`
            - `load_and_resize_image(path, size)`: loads image, converts to grayscale, resizes to `size`, normalizes to [0,1].
            - `fitness_factory(target, method)`: returns a fitness function:
            - `manhattan`: `1 - mean(|individual - target|)` (1.0 = perfect match)
            - `euclidean`: `1 - sqrt(mean((individual - target)^2))` (1.0 = perfect match)
           """
        )    

# =====================================================
# TAB 2: RESULTS
# =====================================================
with tabs[2]:
    st.subheader(T[lang]["config"])
    col1, col2 = st.columns(2)

    survivor_options = ["fitness", "age"]
    survivor_labels = {
        "fitness": {"DE": "Nach Fitness", "EN": "By fitness"},
        "age": {"DE": "Nach Alter", "EN": "By age"}
    }

    init_options = ["random", "expert_knowledge"]
    init_labels = {
        "random": {"DE": "Zufällig", "EN": "Random"},
        "expert_knowledge": {"DE": "Expertenwissen", "EN": "Expert"}
    }

    with col1:
        pop_size = st.slider(T[lang]["pop_size"], 10, 200, 50, step=5)
        generations = st.slider(T[lang]["generations"], 10, 5000, 1000, step=10)
        crossover_alpha = st.slider(T[lang]["crossover_alpha"], 0.0, 1.0, 0.5, 0.05)
        mutation_rate = st.slider(T[lang]["mutation_rate"], 0.0, 1.0, 0.5, 0.05)
        mutation_width = st.slider(T[lang]["mutation_width"], 0.0, 1.0, 0.1, 0.05)
        shape = st.slider(T[lang]["shape"], 0, 256, 16, 16)

    with col2:
        crossover_method = st.selectbox(T[lang]["crossover_method"], ["arithmetic", "global_uniform"])
        mutation_method = st.selectbox(T[lang]["mutation_method"], ["uniform_local", "gaussian_adaptive"])
        parent_selection = st.selectbox(T[lang]["parent_selection"], ["tournament", "rank"])
        survivor_method = st.selectbox(
            T[lang]["survivor_method"],
            options=survivor_options,
            format_func=lambda x: survivor_labels[x][lang]
        )
        initialization_method = st.selectbox(
            T[lang]["init_method"],
            options=init_options,
            format_func=lambda x: init_labels[x][lang]
        )
        fitness_method = st.selectbox(T[lang]["fitness_method"], ["manhattan", "euclidean"])
 
    st.divider()

    #image upload
    uploaded_file = st.file_uploader("Upload your target image (optional)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("L")
        preview = img.resize((400, 400), Image.LANCZOS)
        st.image(preview, caption="Uploaded target (preview ~400x400 px)", width=400)
        # Für die GA: 16x16 normalisieren
        target = np.array(img.resize((shape, shape), Image.LANCZOS), dtype=np.float32) / 255.0
    else:
        default_path = os.path.join("src", "GeneticAlgorithmVariations", "data", "example_image.png")
        if os.path.exists(default_path):
            img = Image.open(default_path).convert("L")
            preview = img.resize((400, 400), Image.LANCZOS)
            st.image(preview, caption="Default target (preview ~400x400 px)", width=400)
            target = load_and_resize_image(default_path, (shape, shape))
        else:
            st.warning("Kein Standard-Target gefunden. Bitte lade ein Zielbild hoch.")
            target = np.ones((shape, shape), dtype=np.float32) * 0.5

    fitness_fn = fitness_factory(target, method=fitness_method)

    if st.button(T[lang]["run"]):
        with st.spinner("Evolving population... please wait"):
            st.session_state["survivor_method"] = survivor_method

            pop = Population(
                size=pop_size,
                shape=(shape, shape),
                initialization_method=initialization_method,
                parent_selection=parent_selection,
                survivor_method="fitness",
                fitness_fn=fitness_fn,
                mutation_method=mutation_method,
                mutation_rate=mutation_rate,
                mutation_width=mutation_width,
                crossover_method=crossover_method,
                alpha=crossover_alpha,
            )

            fitness_history = []
            best_images = []

            for gen in range(generations):
                pop.evolve()
                best = pop.best()
                fitness_history.append(best.fitness)
                if gen % 25 == 0 or gen == generations - 1:
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
        
        df = pd.DataFrame({
            "Generation": list(range(1, len(fitness_history)+1)),
            "Fitness": fitness_history
        })
        
        chart = alt.Chart(df).mark_line().encode(
            x="Generation",
            y="Fitness"
        ).interactive(bind_x=False, bind_y=False)  # Deaktiviert Zoom/Pan
        
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Evolution snapshots")
        best_images = st.session_state["best_images"]
        if best_images:
            n = len(best_images)
            cols = st.columns(min(6, n))
            for i, (gen, genes) in enumerate(best_images):
                img_arr = (genes * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_arr)
                cols[i % len(cols)].image(img_pil, caption=f"Gen {gen}", width=150)
                
        # Final best image / Download
        final_gen, final_genes = best_images[-1]
        final_img = Image.fromarray((final_genes * 255).astype(np.uint8))
        buf = BytesIO()
        final_img.save(buf, format="PNG")
        st.download_button(T[lang]["download"], buf.getvalue(), "best_evolved.png", mime="image/png")
    else:
        st.info("Führe zuerst den Algorithmus im oberen Bereich dieses Tabs aus (Run Genetic Algorithm).")

# =====================================================
# TAB 3: DISCUSSION
# =====================================================
with tabs[3]:
    if lang == "DE":
        st.markdown(
            """
            **Diskussion**

            - **Mutationsmethoden:**  
              - `uniform_local`: gleichmäßige kleine Änderungen  
              - `gaussian_adaptive`: schwächere Individuen mutieren stärker → schnellere Erkundung

            - **Eltern- und Survivor-Auswahl:**  
              - Turnier vs. Rang-basiert → Selektionsdruck vs. Diversität  
              - Survivor nach Fitness → aggressive Eliminierung  
              - Survivor nach Alter → Diversität + Stabilität

            - **Initialisierung:**  
              - `random` → reines Rauschen  
              - `expert_knowledge` → grobes Muster → schnellere Konvergenz, Risiko lokaler Optima

            - **Fitness-Funktion:**  
              - Manhattan vs. Euclidean → Bewertung der Nachkommen  
              - Normalisierung wichtig für Vergleichbarkeit

            - **Abbruchkriterium:**  
              - Feste Generationenanzahl oder minimale Fitness-Verbesserung  
              - Adaptive Kriterien sparen Berechnungen

            - **Mögliche Erweiterungen:**
                - Oft bleibt die Lösung mit typischen Fehlern (Rauschen, Unschärfe) hinter dem Zielbild zurück. Eine mögliche Verbesserung wäre es, eine Bildnachbearbeitung (z.B. Rauschunterdrückung) anzuwenden, um solche Artefakte zu minimieren.
                - Achtung die Bildbearbeitung sollte am besten NACH dem GA sein, da das Glätten die Mutationen unterdrückt!
            """
        )
    else:
        st.markdown(
            """
            **Discussion**

            - **Mutation Methods:**  
              - `uniform_local`: uniform small changes  
              - `gaussian_adaptive`: weaker individuals mutate more → faster exploration

            - **Parent & Survivor Selection:**  
              - Tournament vs. rank-based → selection pressure vs. diversity  
              - Survivor by fitness → aggressive elimination  
              - Survivor by age → diversity + stability

            - **Initialization:**  
              - `random` → pure noise  
              - `expert_knowledge` → rough pattern → faster convergence, risk of local optima

            - **Fitness Function:**  
              - Manhattan vs. Euclidean → offspring evaluation  
              - Normalization important for comparability

            - **Termination Criterion:**  
              - Fixed generations or minimal fitness improvement  
              - Adaptive criteria save computations

            - **Possible Extensions:**
                - Often the solution lags behind the target image with typical artifacts (noise, blur). A possible improvement would be to apply image post-processing (e.g., noise reduction) to minimize such artifacts.
                - Note that the image processing should preferably be done AFTER the GA, as smoothing suppresses mutations!
            """
        )
