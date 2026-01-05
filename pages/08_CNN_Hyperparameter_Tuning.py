import streamlit as st
import os
import torch
import random

from src.CnnHyperparamTuning.fitness_objectives import (
    objective_f1,
    penalty_l2_regularization,
)
from src.CnnHyperparamTuning.main import (
    hill_climbing,
    evaluate_individual,
    build_model,
    genetic_algorithm,
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

    - **Genetischer Algorithmus (GA)** als globaler Optimierer  
    - **Hill Climbing (HC)** als lokaler Feinschliff  

    Diese Kombination ermöglicht eine effektive Balance aus breiter Exploration des Suchraums und gezielter lokaler Verbesserung vielversprechender Lösungen.

    **Warum ein Genetischer Algorithmus?**  
    Die CNN-Hyperparameter sind überwiegend **diskret** (z.B. Anzahl Layer: 1, 2 oder 3; Kernel-Größen: 3 oder 5; Pooling-Typ: max oder avg).  
    GAs sind besonders gut geeignet für solche diskreten Suchräume, da sie natürlich mit kategorischen Variablen umgehen können.

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

    ### 3. Genetischer Algorithmus (globale Suche)

    - **Ziel der GA-Phase:** Breite, globale Exploration des diskreten Suchraums (Architektur-Parameter).
    - **Warum GA statt DE?** Der Suchraum besteht fast ausschließlich aus **diskreten/kategorischen Variablen**:
      - `num_conv_layers`: {1, 2, 3}
      - `kernel_sizes`: {[3,3,3], [5,5,5]}
      - `pool_types`: {max, avg}
      - `use_dropout`: {True, False} pro Layer
      - etc.
      
      GAs sind für solche diskreten Räume besser geeignet als DE, da Crossover und Mutation direkt auf den kategorischen Werten operieren.
    
    - **Initialisierung:** Zufällige Population aus dem UI-Search-Space.
    - **Selektion:** Tournament Selection (2 Kandidaten, besserer gewinnt).
    - **Crossover:** Uniform Crossover - für jeden Parameter wird zufällig entschieden, von welchem Elternteil er stammt.
    - **Mutation:** Mit Wahrscheinlichkeit `mutation_rate` wird ein Parameter durch einen zufälligen Wert aus dem Search-Space ersetzt.
    - **Elitismus:** Die besten `elite_size` Individuen werden direkt in die nächste Generation übernommen.

    #### Fitness-Caching (Hashable Elitism)
    
    Um Rechenzeit zu sparen, werden bereits evaluierte Individuen **gecacht**:
    - Jedes Individuum wird in ein hashbares Tupel konvertiert (`make_hashable`).
    - Ein `fitness_cache` Dictionary speichert bereits berechnete Fitness-Werte.
    - **Vorteil:** Eliten und wiederholte Konfigurationen müssen nicht erneut trainiert werden.


    ---

    ### 4. Hill Climbing (lokale Verfeinerung)

    - **Ziel der HC-Phase:** Lokale Verbesserung der besten vom GA gefundenen Lösung.
    - **Warum HC nach GA?**  
      Der GA exploriert den Suchraum breit, kann aber lokale Optima überspringen.  
      HC verfeinert die beste gefundene Lösung durch systematisches Testen aller Nachbarn.
    - **Nachbarerzeugung:** Alle Kandidaten, die entstehen, wenn man **einen** Parameter des aktuellen Best-Params zu einer anderen erlaubten Option ändert.
    - **Strategie:** First-ascent Hill Climbing:
      - Iteriere über Nachbarn; nimm die **erste** gefundene Verbesserung an.
      - Falls keine Verbesserung gefunden wird → Stop (lokales Optimum erreicht).
    - **Vorteil:** Sehr effizient, um nahegelegene bessere Architekturen schnell zu entdecken.
    - **Caching:** Auch HC nutzt den `fitness_cache`, um bereits evaluierte Nachbarn nicht erneut zu trainieren.

    ---

    ### 5. Gesamt-Workflow (End-to-end)

    1. **UI / Search Space:** Benutzer wählt per Streamlit die erlaubten Werte für jeden Parameter (mehrere Optionen pro Parameter sind möglich).
    2. **GA-Phase:**  
       - Erzeuge initiale Population (Kombinationen aus den ausgewählten Optionen).  
       - Tournament Selection, Crossover, Mutation über mehrere Generationen.  
       - Elitismus bewahrt die besten Individuen.  
       - Fitness-Caching vermeidet redundante Evaluierungen.
    3. **HC-Phase (optional):**  
       - Erzeuge Nachbarn des besten Individuums (eine Änderung pro Nachbar).  
       - Evaluieren & Akzeptieren der ersten Verbesserung; iterativ wiederholen.
    4. **Finales Training & Visualisierung:**  
       - Baue das finale Modell aus den besten Parametern.  
       - Trainiere es (ein Batch / oder kompletter Epochendurchlauf).  
       - Zeige Vorhersagen (Bilder mit True `T:` und Predicted `P:` Labels) in der Streamlit-Oberfläche.

    ---

    ### Wichtiger Hinweis: Stochastische Fitness-Funktion

    Die Fitness-Funktion ist **nicht deterministisch**, da:
    - Das CNN-Training stochastisch ist (zufällige Gewichtsinitialisierung, Batch-Shuffling, Dropout).
    - Bei `quick_run=True` wird nur ein zufälliger Batch pro Epoche verwendet.
    
    **Konsequenzen:**
    - Dasselbe Individuum kann bei erneuter Evaluierung einen **anderen Fitness-Score** erhalten.
    - Ein Individuum kann nach Re-Evaluierung sogar einen **niedrigeren** Score haben als zuvor.
    - Der Fitness-Cache hilft hier nur teilweise: Er verhindert Re-Evaluierung desselben Individuums, aber neue Individuen mit gleichen Parametern werden neu trainiert.
    
    **Empfehlung:**
    - Für robustere Ergebnisse: `quick_run=False` und mehr Epochen verwenden.
    - Bei stochastischer Fitness: Mehrere Läufe durchführen und Ergebnisse mitteln.

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
# TAB 3 - RESULTS - RUN GA + Animation
# ========================================================
with tabs[2]:
    st.header("Hyperparameter konfigurieren & Optimierung starten")

    # =========================================================
    # SIDEBAR - SEARCH SPACE
    # =========================================================
    st.sidebar.subheader("CNN Search Space anpassen")

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

    # =========================================================
    # MODE SELECTION
    # =========================================================
    st.subheader("Optimierungsmodus")

    mode = st.radio(
        "Was möchtest du ausführen?",
        [
            "Nur ein Individuum testen",
            "Genetischer Algorithmus (GA)",
            "GA + Hill Climbing",
        ],
    )

    # =========================================================
    # GA PARAMETER
    # =========================================================
    st.subheader("GA Parameter")

    pop_size = st.slider("Population Size", 5, 30, 10)
    ga_gens = st.slider("GA Generationen", 1, 30, 10)
    mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.2)
    crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
    elite_size = st.slider("Elite Size", 1, 5, 2)

    # =========================================================
    # HC PARAMETER (nur relevant für GA+HC)
    # =========================================================
    if mode == "GA + Hill Climbing":
        st.subheader("Hill Climbing Parameter")
        hc_steps = st.slider("Hill Climbing Steps", 1, 30, 10)

    # =========================================================
    # TRAINING / SYSTEM
    # =========================================================
    st.subheader("Training & System")

    num_epochs = st.slider("Training Epochs pro Individuum", 1, 5, 2)
    quick_evaluation = st.checkbox(
        "Quick Evaluation (nur 1 Batch pro Epoche)", value=True
    )
    little_dataset = st.checkbox("Kleiner Datensatz (10%)", value=True)

    # =========================================================
    # SEARCH SPACE FINAL
    # =========================================================
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

    # =========================================================
    # RUN
    # =========================================================
    if st.button("Starten"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"Device: {device}")

        data_dir = os.path.join("src", "CnnHyperparamTuning", "data")
        os.makedirs(data_dir, exist_ok=True)

        train_loader, test_loader = get_data_loaders(
            data_dir, little_dataset=little_dataset
        )

        fitness_objectives = [objective_f1, penalty_l2_regularization]
        weights = [1.0, -0.01]

        # =====================================================
        # MODE 1 - SINGLE INDIVIDUAL
        # =====================================================
        if mode == "Nur ein Individuum testen":
            st.info("Ein zufälliges Individuum wird getestet")

            indiv = build_individual_from_space(LOCAL_SEARCH_SPACE)

            score = evaluate_individual(
                indiv,
                num_epochs,
                train_loader,
                test_loader,
                device,
                quick_run=True,
                fitness_objectives=fitness_objectives,
                weights=weights,
            )

            st.success(f"Fitness Score: {score:.4f}")
            st.subheader("Parameter")
            st.table({k: [v] for k, v in indiv.items()})

        # =====================================================
        # MODE 2 - GA
        # =====================================================
        elif mode == "Genetischer Algorithmus (GA)":
            st.info("Genetischer Algorithmus läuft …")

            best_params, best_score = genetic_algorithm(
                pop_size=pop_size,
                generations=ga_gens,
                num_epochs=num_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                fitness_objectives=fitness_objectives,
                weights=weights,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elite_size=elite_size,
                quick_run=quick_evaluation,
                local_search_space=LOCAL_SEARCH_SPACE,
                use_streamlit=True,
            )

        # =====================================================
        # MODE 3 - GA + HC
        # =====================================================
        else:
            st.info("GA + Hill Climbing läuft …")

            best_params, best_score = genetic_algorithm(
                pop_size=pop_size,
                generations=ga_gens,
                num_epochs=num_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                fitness_objectives=fitness_objectives,
                weights=weights,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elite_size=elite_size,
                quick_run=quick_evaluation,
                local_search_space=LOCAL_SEARCH_SPACE,
                use_streamlit=True,
            )

            best_params, best_score_hc = hill_climbing(
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
                use_streamlit=True,
            )

            # Fitness function is not deterministic due to training, re-evaluate can give lower score!
            best_score = evaluate_individual(
                best_params,
                num_epochs,
                train_loader,
                test_loader,
                device,
                quick_run=quick_evaluation,
                fitness_objectives=fitness_objectives,
                weights=weights,
            )

        # =====================================================
        # RESULT VISUALIZATION (GA / GA+HC)
        # =====================================================
        if mode != "Nur ein Individuum testen":
            st.success(f"Optimierung abgeschlossen - Best Fitness: {best_score:.4f}")
            st.subheader("Beste gefundene Parameter")
            st.table({k: [v] for k, v in best_params.items()})

            st.subheader("Vorhersagen des finalen Modells")
            model = build_model(best_params, device)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()
            model.train()

            for _ in range(num_epochs):
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            fig = visualize_predictions(model, test_loader, device)
            st.pyplot(fig)

        
# ========================================================
# TAB 4 - DISCUSSION
# ========================================================
with tabs[3]:

    st.markdown("## Diskussion")

    st.markdown(
        """
        ### Warum Genetischer Algorithmus (GA) + Hill Climbing (HC)?

        #### Ursprüngliche Überlegung: Differential Evolution (DE)

        Anfangs wurde **Differential Evolution (DE)** als globaler Optimierer in Betracht gezogen.
        DE ist ein leistungsfähiger evolutionärer Algorithmus, der besonders gut für **kontinuierliche** Optimierungsprobleme geeignet ist.

        **Das Problem:** Unser CNN-Hyperparameter-Suchraum besteht fast ausschließlich aus **diskreten/kategorischen Variablen**:
        - `num_conv_layers`: {1, 2, 3} — nur 3 Werte
        - `kernel_sizes`: {[3,3,3], [5,5,5]} — nur 2 Optionen
        - `pool_types`: {"max", "avg"} — nur 2 Optionen
        - `use_dropout`: {True, False} pro Layer
        - `fc_neurons`: {16, 32, 64, 128} — nur 4 Werte

        **Warum DE hier weniger geeignet ist:**
        - DE arbeitet mit **Differenzvektoren** zwischen Individuen.
        - Bei diskreten Variablen mit wenigen Optionen ergeben diese Differenzen **wenig Sinn**.
        - Beispiel: Die "Differenz" zwischen `kernel_size=3` und `kernel_size=5` ist mathematisch 2, aber als Vektor-Operation im DE-Sinne nicht sinnvoll interpretierbar.
        - DE erfordert zusätzliche **Diskretisierung/Rundung**, was zu suboptimalem Verhalten führt.

        **Wann wäre DE sinnvoll?**
        - Wenn die Hyperparameter **quasi-kontinuierlich** wären (viele nahe diskrete Werte).
        - Beispiel: `learning_rate` in {0.0001, 0.0002, ..., 0.01} oder `num_filters` in {8, 9, 10, ..., 128}.
        - In solchen Fällen approximieren die diskreten Werte einen kontinuierlichen Raum, und DE-Differenzen ergeben sinnvolle Suchrichtungen.

        #### Unsere Lösung: Genetischer Algorithmus (GA)

        **GAs sind für diskrete Suchräume besser geeignet**, weil:
        - **Crossover** tauscht direkt kategorische Werte zwischen Eltern aus (keine Differenzen nötig).
        - **Mutation** ersetzt einen Wert durch einen anderen aus der diskreten Menge.
        - **Tournament Selection** ist robust und funktioniert unabhängig vom Wertebereich.
        - **Elitismus** bewahrt die besten Lösungen über Generationen hinweg.

        #### Hill Climbing (HC) als lokale Verfeinerung

        **Warum HC nach GA?**
        - Der GA exploriert den Suchraum **breit**, kann aber lokale Optima überspringen.
        - HC **verfeinert** die beste gefundene Lösung durch systematisches Testen aller Nachbarn.
        - Die Kombination ermöglicht **globale Exploration (GA)** + **lokale Exploitation (HC)**.

        ---

        ### Fitness-Caching (Hashable Elitism)

        Eine wichtige Optimierung in unserer Implementierung ist das **Fitness-Caching**:

        ```python
        fitness_cache = {}
        def fitness_of(ind):
            key = individual_key(ind)  # Konvertiert zu hashbarem Tupel
            if key not in fitness_cache:
                fitness_cache[key] = evaluate_individual(...)
            return fitness_cache[key]
        ```

        **Vorteile:**
        - Eliten müssen nicht erneut evaluiert werden → erhebliche Zeitersparnis.
        - Identische Konfigurationen (durch Crossover entstanden) werden nur einmal trainiert.
        - Der Cache bleibt über alle GA-Generationen und HC-Schritte erhalten.

        ---

        ### Stochastische Fitness-Funktion

        **Kritischer Punkt:** Die Fitness-Funktion ist **nicht deterministisch**!

        Das liegt an mehreren Faktoren:
        - **Zufällige Gewichtsinitialisierung** des CNN bei jedem Training.
        - **Batch-Shuffling** während des Trainings.
        - **Dropout** führt zu unterschiedlichen Aktivierungen.
        - Bei `quick_run=True` wird nur ein **zufälliger Batch** pro Epoche verwendet.

        **Konsequenzen:**
        1. Dasselbe Individuum kann bei erneuter Evaluierung einen **anderen Fitness-Score** erhalten.
        2. Ein Individuum kann nach Re-Evaluierung sogar einen **niedrigeren Score** haben.
        3. Der Fitness-Cache mildert das Problem: Er speichert den **ersten** berechneten Wert.

        **Empfehlungen für robustere Ergebnisse:**
        - `quick_run=False` verwenden (vollständiges Training pro Epoche).
        - Mehr Epochen (`num_epochs >= 3`) für stabilere Konvergenz.
        - Bei kritischen Anwendungen: Mehrere Läufe durchführen und Ergebnisse mitteln.

        ---

        ### Algorithmen-Vergleich

        | Algorithmus | Eignung für unser Problem |
        |-------------|---------------------------|
        | **Genetischer Algorithmus (GA)** | Ideal für diskrete/kategorische Variablen. Crossover und Mutation operieren direkt auf den Werten. |
        | **Differential Evolution (DE)** | Für kontinuierliche Räume optimiert. Differenzen zwischen diskreten Werten wenig sinnvoll. |
        | **Simulated Annealing (SA)** | Einzelner Suchpfad; ineffizient bei vielen Parametern. |
        | **Particle Swarm Optimization (PSO)** | Geschwindigkeitsvektoren für kontinuierliche Räume; schlecht für kategorische Parameter. |
        | **Ant Colony Optimization (ACO)** | Für Wegprobleme; aufwändig anzupassen für CNN-Architekturen. |

        ---

        ### Rechenzeit und Compute-Betrachtung

        Ein wesentlicher Faktor in der Wahl der Optimierungsstrategie ist die **Rechenzeit**, denn:

        - Jedes Individuum entspricht einem **Trainingslauf eines CNNs** - selbst „quick runs“ sind teuer.  
        - Die Gesamtzeit eines GA-Durchlaufs ist:  

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

        Die GA + HC Optimierung zeigt:

        - Gefundene Parameter balancieren **Modellkomplexität** und **Generalisierung**.
        - Hill Climbing verbessert oft subtil einzelne Aspekte der Netzwerkstruktur.
        - Durch die stochastische Fitness können Ergebnisse zwischen Läufen variieren.

        Insgesamt bietet der Ansatz eine **effektive, gut kontrollierbare Methode** für CNN-Hyperparameter-Optimierung mit diskreten Suchräumen.

        ---"""
    )
