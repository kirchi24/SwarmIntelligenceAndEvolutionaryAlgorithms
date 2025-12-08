import streamlit as st
import os
import torch
import random

from src.CnnHyperparamTuning.fitness_objectives import (
    objective_f1,
    penalty_l2_regularization,
)
from src.CnnHyperparamTuning.main import (
    differential_evolution,
    hill_climbing,
    evaluate_individual,
    build_model,
    get_data_loaders,
    visualize_predictions,
    SEARCH_SPACE,
)

import ast


def safe_eval_list(v):
    """Konvertiert '[8,8,8]' sauber zu [8,8,8]."""
    if isinstance(v, list):
        return v
    return ast.literal_eval(v)


def build_individual_from_space(space):
    """Erstellt ein zufälliges Individuum aus dem UI-Suchraum."""
    return {
        "num_conv_layers": random.choice(space["num_conv_layers"]),
        "filters_per_layer": random.choice(space["filters_per_layer"]),
        "kernel_sizes": random.choice(space["kernel_sizes"]),
        "pool_types": random.choice(space["pool_types"]),
        "use_dropout": random.choice(space["use_dropout"]),
        "dropout_rates": random.choice(space["dropout_rates"]),
        "fc_neurons": random.choice(space["fc_neurons"]),
    }


# ========================================================
# Streamlit Page Setup
# ========================================================

st.set_page_config(page_title="CNN Hyperparameter Tuning", layout="wide")

st.title("CNN Hyperparameter Tuning")


# ========================================================
# Tabs
# ========================================================

tabs = st.tabs(["Introduction", "Methods", "Results", "Discussion"])


# ========================================================
# TAB 1 - INTRODUCTION
# ========================================================
with tabs[0]:
    st.markdown(
        r"""
    ## Einleitung

    Ziel dieser Arbeit ist die **Optimierung der Hyperparameter einer Convolutional Neural Network (CNN) Architektur** zur Klassifikation des **Fashion-MNIST** Datensatzes.  
    Hierfür soll ein Optimierungsalgorithmus - oder eine Kombination aus globalen und lokalen Suchverfahren - ausgewählt, implementiert und begründet werden.

    In diesem Projekt wurde eine hybride Strategie gewählt, bestehend aus:

    - **Differential Evolution (DE)** als globaler Optimierer  
    - **Hill Climbing (HC)** als lokaler Feinschliff  

    Diese Kombination ermöglicht eine effektive Balance aus breiter Exploration des Suchraums und gezielter lokaler Verbesserung vielversprechender Lösungen.

    ---

    ## Aufgabenstellung

    Die wesentlichen Teilaufgaben lauten:

    1. **Wähle einen Optimierungsalgorithmus**  
    (z. B. HC, GA, SA, ACO, PSO, DE - oder eine Kombination daraus)  
    und **begründe deine Wahl** basierend auf Problemstruktur und Rechenbudget.

    2. **Verwende einen sinnvollen Subsample des Fashion-MNIST Datensatzes**  
    Der komplette Datensatz besteht aus **70 000 Graustufen-Bildern**  
    (60 000 Training, 10 000 Test) in 10 Modekategorien.

    3. **Optimiere eine CNN-Architektur**  
    Ziel: **möglichst gute Klassifikationsperformance** auf dem Fashion-MNIST-Datensatz.  
    Dazu gehören Entscheidungen über:
    - Anzahl Convolution-Layer  
    - Filteranzahl & Kernelgrößen  
    - Pooling-Methoden  
    - Dropout-Einsatz  
    - Fully-Connected Layer Größe  
    - etc.

    ---

    ## Der Fashion-MNIST Datensatz

    Der Fashion-MNIST Datensatz ist ein weit verbreitetes Benchmark-Dataset, das als moderner Ersatz für das klassische MNIST-Zifferndataset dient.  
    Er besteht aus **28x28 Pixel großen Graustufenbildern** von echten Modeartikeln aus dem Zalando-Sortiment.

    Er umfasst folgende 10 Klassen:

    0 - T-shirt/top  
    1 - Trouser  
    2 - Pullover  
    3 - Dress  
    4 - Coat  
    5 - Sandal  
    6 - Shirt  
    7 - Sneaker  
    8 - Bag  
    9 - Ankle boot  

    ### Beispielbilder aus Fashion-MNIST:
    """
    )

    # Beispielbilder laden (einmalig)
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    sample_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )

    fig, axes = plt.subplots(1, 10, figsize=(12, 2))
    for i in range(10):
        img, label = sample_data[i]
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(str(label))
        axes[i].axis("off")

    st.pyplot(fig)

    st.markdown(
        r"""
    Fashion-MNIST ist bewusst herausfordernder als das klassische MNIST-Zifferndataset:  
    Die Klassen unterscheiden sich visuell subtiler, und viele Kleidungsstücke weisen  
    ähnliche Formen oder Texturen auf. Dadurch eignet sich Fashion-MNIST ideal als Testumgebung  
    für Optimierungsalgorithmen, die robuste und flexible Modellarchitekturen hervorbringen sollen.
    """
    )


# ========================================================
# TAB 2 - METHODS
# ========================================================
with tabs[1]:
    st.header("Methoden")

    st.markdown(
        """
    ### 1. Konfigurierbare CNN-Architektur (`ConfigurableCNN`)

    - **Ziel:** Ermöglichen, beliebige CNN-Varianten (innerhalb definierter Grenzen) automatisch zu bauen.
    - **Parameter (konfigurierbar):**
    - `num_conv_layers` (1-3)
    - `filters_per_layer` (z. B. `[8,16,32]`)
    - `kernel_sizes` (nur `3` oder `5`)
    - `pool_types` (`"max"` oder `"avg"` pro Layer)
    - `use_dropout` und `dropout_rates` pro Layer
    - `fc_neurons` (Anzahl Neuronen im Fully-Connected-Layer, ≤ 128)
    - **Aufbau pro Conv-Block:** `Conv2d → ReLU → Pool → Dropout/Identity`
    - **Flatten-Größe:** Wird automatisch über einen Dummy-Forward (Tensor `1x1x28x28`) berechnet, damit die FC-Layer-Dimension korrekt gesetzt wird.
    - **Forward-Pfad:**  
    `Input → (Conv → ReLU → Pool → Dropout)xN → Flatten → FC → Output`

    ---

    ### 2. Fitness-Berechnung (Multi-Objective)

    - **Primäres Ziel:** Maximierung des **macro F1-Scores** auf dem Testset (`objective_f1`).
    - **Sekundäre Ziele / Penalties:** z. B. L2-Gewichtsstrafe (`penalty_l2_regularization`) oder Modellgrößenstrafe (`penalty_large_model`).
    - **Skalierung:** Jedes Objective wird mittels globaler Min-/Max-Werte auf `[0,1]` skaliert (siehe `GLOBAL_OBJECTIVE_MINS/MAXS`).
    - **Kombination:** Gewichtete Summe der skalierten Objectives:
    $$
    \\text{Fitness} =  \\sum_i w_i \\cdot \\text{scaled\\_objective}_i
    $$
    Standard-Gewichtung: `weights = [1.0, -0.01]` (F1 hoch, L2 niedrig).

    ---

    ### 3. Differential Evolution

    - Ziel der DE-Phase: Breite, globale Exploration des Suchraums (Architektur-Parameter).
    - Initialisierung: Population von Konfigurationen wird erzeugt. In der Implementierung werden Optionen aus dem UI-/Search-Space zufällig kombiniert (je Parameter eine Auswahl).
    - Evaluation: Jedes Individuum → CNN bauen → kurz trainieren (parameter `num_epochs`, evtl. `quick_run`) → Fitness berechnen.
    - Selektion / Speicherung: Nur die Konfigurationen mit der besten Fitness werden in der neuen Generation übernommen. Diskrete Parameter (z.B. num_conv_layers) werden immer auf die nächsten erlaubten Werte gerundet.
    - Mutation / Crossover: Für jedes Individuum werden drei weitere Kandidaten ausgewählt, um den Mutanten zu erzeugen (Faktor F). Anschließend wird durch Crossover entschieden, welche Werte aus dem Mutanten übernommen werden.
    - New Generation: Die nächste Generation besteht aus den Individuen, die sich nach der Selektion durchgesetzt haben (bessere Fitness).
    - **Compute-Kosten: Laufzeit ∝ Population x Generations x Trainingsdauer pro Individuum.


    ---

    ### 4. Hill Climbing (lokale Verfeinerung)

    - **Ziel der HC-Phase:** Lokale Verbesserung der besten von DE gefundenen Lösung.
    - **Nachbarerzeugung:** Alle Kandidaten, die entstehen, wenn man **einen** Parameter des aktuellen Best-Params zu einer anderen erlaubten Option ändert.
    - **Strategie:** First-ascent Hill Climbing:
    - Iteriere über Nachbarn; nimm die **erste** gefundene Verbesserung an.
    - Falls keine Verbesserung gefunden wird → Stop.
    - **Vorteil:** Sehr effizient, um nahegelegene bessere Architekturen schnell zu entdecken.

    ---

    ### 5. Gesamt-Workflow (End-to-end)

    1. **UI / Search Space:** Benutzer wählt per Streamlit die erlaubten Werte für jeden Parameter (mehrere Optionen pro Parameter sind möglich).
    2. **DE-Phase:**  
    - Erzeuge initiale Population (Kombinationen aus den ausgewählten Optionen).  
    - Trainiere/werte Individuen aus (kurze Trainingsläufe für Exploration).  
    - Speichere das beste Individuum.
    3. **HC-Phase:**  
    - Erzeuge Nachbarn des besten Individuums (eine Änderung pro Nachbar).  
    - Evaluieren & Akzeptieren der ersten Verbesserung; iterativ wiederholen.
    4. **Finales Training & Visualisierung:**  
    - Baue das finale Modell aus den besten Parametern.  
    - Trainiere es (ein Batch / oder kompletter Epochendurchlauf wie in `main.py`).  
    - Zeige Vorhersagen (Bilder mit True `T:` und Predicted `P:` Labels) in der Streamlit-Oberfläche.

    ---

    ### 6. Implementierungsdetails / wichtige Hinweise

    - **Datentransfers:** Modelle und Daten werden auf `device` (CPU oder GPU) verschoben (`.to(device)`).
    - **Stabilität:** Vor dem Erstellen eines Modells werden Assertions in `ConfigurableCNN` ausgeführt (z. B. `1 <= num_conv_layers <= 3`) — falsche Typen (Liste statt int) führen zu Fehlern.
    - **Performance-Tradeoffs:** Für schnelle Iterationen wird `quick_run=True` verwendet (nur ein Batch pro Epoch), für finale Evaluierung sollten mehrere Epochen / kompletter Trainingslauf gewählt werden.
    - **Reproduzierbarkeit:** Durch UI-geführten Search-Space und deterministische Auswahl/Neighbour-Generierung lässt sich das Experiment gut reproduzieren.
    - **Erweiterbarkeit:** Die Architektur- und Fitness-Module sind modular; neue Objectives (z. B. validation loss, inference latency) lassen sich leicht hinzufügen und in die gewichtete Fitness einfließen.
    """
    )
# ========================================================
# TAB 3 - RESULTS - RUN DE + Animation
# ========================================================
with tabs[2]:
    st.header("Hyperparameter konfigurieren & Optimierung starten")

    st.sidebar.subheader("CNN Search Space anpassen")

    # --- Editable Search Space (with multiselect) ---
    num_conv_layers = st.sidebar.multiselect(
        "Anzahl Convolution-Layer",
        options=[1, 2, 3],
        default=[1, 2, 3],
    )

    filters = st.sidebar.multiselect(
        "Filters per Layer (Sets)",
        options=[str(f) for f in SEARCH_SPACE["filters_per_layer"]],
        default=[str(f) for f in SEARCH_SPACE["filters_per_layer"]],
    )

    kernels = st.sidebar.multiselect(
        "Kernel Sizes (Sets)",
        options=[str(k) for k in SEARCH_SPACE["kernel_sizes"]],
        default=[str(k) for k in SEARCH_SPACE["kernel_sizes"]],
    )

    pools = st.sidebar.multiselect(
        "Pooling Types",
        options=[str(p) for p in SEARCH_SPACE["pool_types"]],
        default=[str(p) for p in SEARCH_SPACE["pool_types"]],
    )

    dropouts = st.sidebar.multiselect(
        "Use Dropout",
        options=[str(d) for d in SEARCH_SPACE["use_dropout"]],
        default=[str(d) for d in SEARCH_SPACE["use_dropout"]],
    )

    dropout_rates = st.sidebar.multiselect(
        "Dropout Rates",
        options=[str(d) for d in SEARCH_SPACE["dropout_rates"]],
        default=[str(d) for d in SEARCH_SPACE["dropout_rates"]],
    )

    fc_neurons = st.sidebar.multiselect(
        "FC Neurons",
        options=SEARCH_SPACE["fc_neurons"],
        default=SEARCH_SPACE["fc_neurons"],
    )

    # --- Mode Selection: only 1 or full optimization ---
    mode = st.radio(
        "Was möchtest du ausführen?",
        ["Nur ein Individuum testen", "DE + Hill Climbing Optimierung"],
    )

    st.subheader("Optimierungsparameter (DE + HC)")
    de_gens = st.slider("DE Generationen", min_value=1, max_value=20, value=5)
    pop_size = st.slider("Population Size", 5, 30, 10)
    hc_steps = st.slider("Hill Climbing Steps", 1, 30, 10)
    num_epochs = st.slider("Training Epochs per Individuum", 1, 5, 2)
    quick_evaluation = st.checkbox("Quick Evaluation (nur 1 Batch pro Epoche)", value=True)

    # Prepare UI Search Space
    LOCAL_SEARCH_SPACE = {
        "num_conv_layers": num_conv_layers,
        "filters_per_layer": [safe_eval_list(f) for f in filters],
        "kernel_sizes": [safe_eval_list(k) for k in kernels],
        "pool_types": [safe_eval_list(p) for p in pools],
        "use_dropout": [safe_eval_list(d) for d in dropouts],
        "dropout_rates": [safe_eval_list(dr) for dr in dropout_rates],
        "fc_neurons": fc_neurons,
    }

    st.write("### Aktiver Search Space")
    st.table({k: [str(v)] for k, v in LOCAL_SEARCH_SPACE.items()})

    # ========================= RUN ===============================
    if st.button("Starten"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"Device: {device}")
        data_dir = os.path.join("src", "CnnHyperparamTuning", "data")
        os.makedirs(data_dir, exist_ok=True)
        train_loader, test_loader = get_data_loaders(data_dir)

        # =======================================================
        # MODE 1: ONLY ONE INDIVIDUAL
        # =======================================================
        if mode == "Nur ein Individuum testen":
            st.info("Ein Individuum wird getestet...")

            indiv = build_individual_from_space(LOCAL_SEARCH_SPACE)

            score = evaluate_individual(
                indiv, num_epochs, train_loader, test_loader, device, quick_run=True
            )

            st.success(f"Fitness Score: {score:.4f}")

            st.subheader("Parameter dieses Individuums")
            st.table({k: [v] for k, v in indiv.items()})

            # Visualization
            st.subheader("Vorhersagen des Modells")
            model = build_model(indiv, device)
            fig = visualize_predictions(model, test_loader, device)
            st.pyplot(fig)
            st.markdown(
                """
            ### Legende

            **T:** True Label (echtes Label)  
            **P:** Predicted Label (vom Modell vorhergesagt)

            **Fashion-MNIST Klassen:**
            - **0** - T-shirt/top  
            - **1** - Trouser  
            - **2** - Pullover  
            - **3** - Dress  
            - **4** - Coat  
            - **5** - Sandal  
            - **6** - Shirt  
            - **7** - Sneaker  
            - **8** - Bag  
            - **9** - Ankle boot  
            """
            )

        # =======================================================
        # MODE 2: FULL DE + HILL CLIMBING
        # =======================================================
        else:
            st.info("DE + HC Optimierung läuft …")

            # --- Parameters für Fitness & Gewichtung ---
            fitness_objectives = [objective_f1, penalty_l2_regularization]
            weights = [1.0, -0.01]

            # --- Differential Evolution ---
            best_params, best_score = differential_evolution(
                pop_size=pop_size,
                de_gens=de_gens,
                num_epochs=num_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                fitness_objectives=fitness_objectives,
                weights=weights,
                quick_run=quick_evaluation,
                local_search_space=LOCAL_SEARCH_SPACE,
            )
            st.info(f"DE abgeschlossen. Bester Score: {best_score:.4f}")

            # --- Hill Climbing ---
            st.info("Starte Hill Climbing...")
            best_params = hill_climbing(
                best_params=best_params,
                hc_steps=hc_steps,
                num_epochs=num_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                fitness_objectives=fitness_objectives,
                weights=weights,
                quick_run=quick_evaluation,
                local_search_space=LOCAL_SEARCH_SPACE,
            )

            # --- Best Score nach HC ---
            best_score = evaluate_individual(
                best_params,
                num_epochs=num_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                quick_run=quick_evaluation,
                fitness_objectives=fitness_objectives,
                weights=weights,
            )

            st.info(f"Optimierung abgeschlossen! Bester Score: {best_score:.4f}")

            st.subheader("Beste gefundene Parameter")
            st.table({k: [v] for k, v in best_params.items()})

            st.subheader(f"Bester Fitness Score: {best_score:.4f}")

            # Final model visualization
            st.subheader("Vorhersagen des finalen Modells")
            model = build_model(best_params, device)
            fig = visualize_predictions(model, test_loader, device)
            st.pyplot(fig)

            st.markdown(
                """
            ### Legende

            **T:** True Label (echtes Label)  
            **P:** Predicted Label (vom Modell vorhergesagt)

            **Fashion-MNIST Klassen:**
            - **0** - T-shirt/top  
            - **1** - Trouser  
            - **2** - Pullover  
            - **3** - Dress  
            - **4** - Coat  
            - **5** - Sandal  
            - **6** - Shirt  
            - **7** - Sneaker  
            - **8** - Bag  
            - **9** - Ankle boot  
            """
            )

# ========================================================
# TAB 4 - DISCUSSION
# ========================================================
with tabs[3]:

    st.markdown("## Diskussion")

    st.markdown(
        """
        ### Warum Differential Evolution (DE) + Hill Climbing (HC)?

        Die Kombination aus **Differential Evolution** und **Hill Climbing** ist besonders attraktiv für die Optimierung von Convolutional Neural Networks (CNNs), da sie die Stärken beider Verfahren vereint:

        1. **Differential Evolution (DE)**  
        - Ist ein robuster, globaler Optimierer, der ohne Gradienteninformationen arbeitet.  
        - Funktioniert gut in *gemischt-diskreten* und *nicht-konvexen* Suchräumen - typisch für CNN-Architekturen.  
        - Benötigt im Vergleich zu anderen evolutionären Verfahren weniger Hyperparameter (Mutation, Crossover).  
        - Kann komplexe Architekturentscheidungen wie Kernelgrößen, Dropout-Konfigurationen oder Anzahl der Convolution-Layer explorieren.

        2. **Hill Climbing (HC)**  
        - Ein schneller lokaler Optimierer, der die von DE gefundene Lösung effizient weiter verbessert.  
        - Arbeitet deterministisch: Änderungen im Suchraum werden strukturiert evaluiert.  
        - Ideal, um nahegelegene Varianten der Architektur zu testen und die globale Lösung zu verfeinern.

        Die Kombination ermöglicht damit **globale Exploration (DE)** und **lokale Exploitation (HC)** - ein entscheidender Vorteil in hochdimensionalen Architektursuchräumen.

        ---

        ### Warum nicht andere Algorithmen?

        | Algorithmus | Warum weniger geeignet? |
        |-------------|--------------------------|
        | **Genetische Algorithmen (GA)** | Höherer Konfigurationsaufwand (Selektion, Mutation, Crossover), oft langsamer, empfindlich gegenüber Designentscheidungen. |
        | **Simulated Annealing (SA)** | Gut für das Entkommen aus lokalen Minima, aber ineffizient in hochdimensionalen, kombinierten Parameterlandschaften. |
        | **Ant Colony Optimization (ACO)** | Ursprünglich für kombinatorische Wegeprobleme; schwer anzupassen für CNN-Architekturen mit multiplen Parameterarten. |
        | **Particle Swarm Optimization (PSO)** | Funktioniert exzellent für kontinuierliche Räume, aber schwach bei diskreten, kategorischen Parametern. |

        Gerade beim Design von CNNs, wo Parameter wie  
        - Anzahl Convolution-Layer  
        - Pooling-Typ  
        - Kernel-Größen  
        - Dropout-Konfigurationen  
        diskret und stark eingeschränkt sind, spielt DE seine Vorteile voll aus.

        ---

        ### Rechenzeit und Compute-Betrachtung

        Ein wesentlicher Faktor in der Wahl der Optimierungsstrategie ist die **Rechenzeit**, denn:

        - Jedes Individuum entspricht einem **Trainingslauf eines CNNs** - selbst „quick runs“ sind teuer.  
        - Die Gesamtzeit eines DE-Durchlaufs ist:  

        $$
        T_{\\text{total}} = \\text{Population Size} \\times \\text{Generations} \\times T_{\\text{Training}}
        $$

        - Hill Climbing führt zusätzliche Modelltrainings auf Nachbararchitekturen aus.  
        - Schon kleine Anpassungen der Parameter (z. B. mehr Generationen oder volle statt schnelle Trainingsläufe) erhöhen die Laufzeit signifikant.

        In typischen Hardware-Umgebungen (CPU oder einfache GPU) kann dies schnell mehrere Minuten bis Stunden dauern.  
        Die Implementierung im Streamlit-Interface bietet daher:

        - **Einstellbare Generationen und Populationsgrößen**, um Compute zu steuern  
        - **Option, nur ein einzelnes Individuum auszuwerten**, was die Laufzeit auf Sekunden reduziert  
        - **quick_run = True**, um die Trainingszeit pro Individuum stark zu verkürzen  

        Dadurch lässt sich der Optimierungsprozess auch unter begrenzten Ressourcen sinnvoll durchführen.

        ---

        ### Interpretation der Ergebnisse

        Die DE + HC Optimierung zeigt:

        - Die gefundenen Parameter spiegeln oft eine Balance zwischen **Modellkomplexität** und **Generalierungsleistung** wider.  
        - Hill Climbing verbessert häufig subtil einzelne Aspekte der Netzwerkstruktur, was zu kleinen, aber signifikanten Leistungsgewinnen führt.  
        - Visualisierte Vorhersagen geben unmittelbare Einblicke in Stärken und Schwächen des finalen Modells.

        Insgesamt bietet der Ansatz eine **effektive, gut kontrollierbare und rechenschonende Methode**, um CNN-Hyperparameter zu optimieren - besonders in einem Lern- oder Forschungsumfeld, wo Interpretierbarkeit und reproduzierbare Optimierungsschritte essentiell sind.

        ---"""
    )
