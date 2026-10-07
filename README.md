# 🏥 GlucoPredict — Diabetes Risk Prediction System

> An AI-powered, educational web app that estimates a person's likelihood of diabetes from eight clinical measurements, using a Support Vector Machine (SVM) trained on the Pima Indians Diabetes dataset and served through a Streamlit interface.

> ⚠️ **Medical disclaimer:** GlucoPredict is a machine-learning learning project. It is **not** a diagnostic tool and must not replace advice from a qualified healthcare professional.

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Tech Stack](#-tech-stack)
4. [Repository Structure](#-repository-structure)
5. [System Architecture](#-system-architecture)
6. [Machine Learning Pipeline](#-machine-learning-pipeline)
7. [Dataset & Features](#-dataset--features)
8. [Application Workflow](#-application-workflow)
9. [Risk Classification Logic](#-risk-classification-logic)
10. [Risk Factor & Recommendation Engine](#-risk-factor--recommendation-engine)
11. [UML & Design Diagrams](#-uml--design-diagrams)
12. [Installation & Usage](#-installation--usage)
13. [Deployment](#-deployment)
14. [Limitations & Known Issues](#-limitations--known-issues)
15. [Future Improvements](#-future-improvements)
16. [Contributing, License & Acknowledgements](#-contributing-license--acknowledgements)

---

## 🔎 Overview

Diabetes is often diagnosed late. GlucoPredict demonstrates how a classical ML model can turn routine clinical measurements into an instant, interpretable **risk estimate**.

A user enters eight values in a sidebar form. The app standardises them with a saved `StandardScaler`, feeds them to a pre-trained SVM, and displays:

- a **binary prediction** (Diabetic / Non-diabetic),
- a **probability breakdown** and an interactive **risk gauge**,
- a **three-tier risk label** (Low / Moderate / High),
- **rule-based risk factors** and **positive health indicators**,
- **general recommendations** and a medical disclaimer.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧮 SVM classifier | Pre-trained model loaded from `diabetes_model.pkl` |
| 📏 Feature scaling | Same `StandardScaler` used in training, loaded from `scaler_svm.pkl` |
| 🎛️ Interactive inputs | Sliders and number inputs with clinically sensible ranges |
| 📊 Plotly gauge | Colour-banded (green / yellow / red) risk meter |
| 🧠 Risk factor analysis | Transparent, rule-based flags for glucose, BMI, age, BP, pedigree |
| 💡 Recommendations | Different guidance for positive vs. negative predictions |
| 🛡️ Graceful errors | Clear messages if model files are missing or inputs mismatch |
| ⚡ Cached loading | `@st.cache_resource` loads the model once per session |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Web UI | Streamlit |
| Data handling | pandas, NumPy |
| ML | scikit-learn (SVM, StandardScaler) |
| Persistence | joblib (`.pkl` artefacts) |
| Visualisation | Plotly (`graph_objects.Indicator`) |
| Experimentation | Jupyter Notebook |

---

## 📁 Repository Structure

```text
GlucoPredict/
│
├── app.py                      # Streamlit web application
├── diabetes-prediction.ipynb   # EDA, preprocessing, training & evaluation
├── diabetes.csv                # Pima Indians Diabetes dataset (768 rows)
├── diabetes_model.pkl          # Trained SVM model
├── scaler_svm.pkl              # Fitted StandardScaler
├── requirements.txt            # Python dependencies
└── .gitignore
```

```mermaid
graph LR
    ROOT[GlucoPredict/]
    ROOT --> A[app.py<br/>Streamlit UI]
    ROOT --> B[diabetes-prediction.ipynb<br/>Training notebook]
    ROOT --> C[diabetes.csv<br/>Dataset]
    ROOT --> D[diabetes_model.pkl<br/>SVM model]
    ROOT --> E[scaler_svm.pkl<br/>StandardScaler]
    ROOT --> F[requirements.txt]
    ROOT --> G[.gitignore]
    B -. produces .-> D
    B -. produces .-> E
    C -. consumed by .-> B
    D -. loaded by .-> A
    E -. loaded by .-> A
```

---

## 🏗️ System Architecture

### High-level architecture

```mermaid
flowchart TB
    subgraph Client["🖥️ Client (Browser)"]
        UI["Streamlit Sidebar<br/>Patient Inputs"]
        OUT["Results Panel<br/>Gauge • Metrics • Advice"]
    end

    subgraph Server["⚙️ Streamlit Server (app.py)"]
        VAL["Input Collection<br/>(8 features)"]
        LOAD["load_model_and_scaler()<br/>@st.cache_resource"]
        SCALE["StandardScaler.transform()"]
        PRED["SVM.predict() / predict_proba()"]
        RISK["Risk Tiering<br/>Low / Moderate / High"]
        RULES["Rule-based<br/>Risk Factor Engine"]
        REC["Recommendation Engine"]
    end

    subgraph Artefacts["💾 Persisted Artefacts"]
        M[("diabetes_model.pkl")]
        S[("scaler_svm.pkl")]
    end

    UI --> VAL --> SCALE --> PRED --> RISK
    LOAD --> M
    LOAD --> S
    LOAD -.-> SCALE
    LOAD -.-> PRED
    VAL --> RULES
    PRED --> REC
    RISK --> OUT
    RULES --> OUT
    REC --> OUT
```

### Layered view

```mermaid
flowchart LR
    P["Presentation Layer<br/>Streamlit widgets, Plotly gauge, CSS"] --> L["Application Logic Layer<br/>Validation, risk tiering, rules, recommendations"]
    L --> I["Inference Layer<br/>StandardScaler + SVM (scikit-learn)"]
    I --> D["Data / Artefact Layer<br/>.pkl files, diabetes.csv"]
```

---

## 🤖 Machine Learning Pipeline

### Training pipeline (notebook)

```mermaid
flowchart LR
    A[("diabetes.csv<br/>768 × 9")] --> B[Load with pandas]
    B --> C[Exploratory Data Analysis<br/>distributions • correlations]
    C --> D[Data Cleaning<br/>handle zero / missing values]
    D --> E[Train / Test Split]
    E --> F[Fit StandardScaler<br/>on training data]
    F --> G[Train SVM Classifier]
    G --> H[Evaluate<br/>accuracy • confusion matrix • report]
    H --> I{Satisfactory?}
    I -- No --> J[Tune hyper-parameters<br/>C • kernel • gamma]
    J --> G
    I -- Yes --> K[Export with joblib]
    K --> L[("diabetes_model.pkl")]
    K --> M[("scaler_svm.pkl")]
```

> The notebook's exact preprocessing and tuning steps are best documented by the author; the stages above reflect the standard pipeline implied by the saved artefacts (`scaler_svm.pkl`, `diabetes_model.pkl`) and the reported ~78 % accuracy.

### Inference pipeline (app)

```mermaid
flowchart LR
    A["Raw input vector<br/>[Preg, Glu, BP, Skin, Ins, BMI, DPF, Age]"] --> B["np.array shape (1, 8)"]
    B --> C["scaler.transform()"]
    C --> D["model.predict()"]
    C --> E["model.predict_proba()"]
    D --> F["Class label 0 / 1"]
    E --> G["P(non-diabetic), P(diabetic)"]
    F --> H["Risk tier + messaging"]
    G --> H
```

---

## 📊 Dataset & Features

The data follows the **Pima Indians Diabetes Database** schema (768 records, 8 predictors, 1 binary target).

| # | Feature | Unit | UI Control | UI Range |
|---|---|---|---|---|
| 1 | Pregnancies | count | Number input | 0 – 20 |
| 2 | Glucose | mg/dL | Slider | 0 – 200 |
| 3 | BloodPressure | mm Hg | Slider | 0 – 130 |
| 4 | SkinThickness | mm | Slider | 0 – 100 |
| 5 | Insulin | µU/mL | Slider | 0 – 900 |
| 6 | BMI | kg/m² | Number input | 10.0 – 70.0 |
| 7 | DiabetesPedigreeFunction | score | Slider | 0.0 – 2.5 |
| 8 | Age | years | Slider | 21 – 100 |
| — | **Outcome** (target) | 0 / 1 | — | Non-diabetic / Diabetic |

> ⚠️ **Feature order is critical.** `app.py` builds the input array in exactly the order above; it must match the training order.

### Entity-relationship / data model

```mermaid
erDiagram
    PATIENT_RECORD {
        int    Pregnancies
        int    Glucose
        int    BloodPressure
        int    SkinThickness
        int    Insulin
        float  BMI
        float  DiabetesPedigreeFunction
        int    Age
        int    Outcome
    }
    TRAINED_MODEL {
        string type "SVM"
        string file "diabetes_model.pkl"
    }
    SCALER {
        string type "StandardScaler"
        string file "scaler_svm.pkl"
    }
    PREDICTION {
        int    label
        float  prob_negative
        float  prob_positive
        string risk_tier
    }
    PATIENT_RECORD ||--o{ TRAINED_MODEL : "trains"
    PATIENT_RECORD ||--o{ SCALER : "fits"
    SCALER ||--|| PREDICTION : "transforms input for"
    TRAINED_MODEL ||--|| PREDICTION : "produces"
```

---

## 🔄 Application Workflow

### End-to-end user flow

```mermaid
flowchart TD
    Start([User opens app]) --> Load["Load model + scaler (cached)"]
    Load --> Ok{Files found<br/>and valid?}
    Ok -- No --> Err["Show error + expected folder layout<br/>st.stop()"]
    Ok -- Yes --> Landing["Landing page:<br/>Model = SVM • ~78% • 768 samples"]
    Landing --> Input["User fills sidebar inputs"]
    Input --> Click{Click<br/>Predict?}
    Click -- No --> Landing
    Click -- Yes --> Build["Build 1×8 feature array"]
    Build --> Scale["Standardise features"]
    Scale --> Pred["Predict class + probabilities"]
    Pred --> Show["Render results:<br/>risk banner • metrics • gauge"]
    Show --> Factors["Risk factor analysis"]
    Factors --> Recs["Recommendations"]
    Recs --> Disc["Medical disclaimer"]
    Disc --> End([Done])
```

### Sequence diagram — one prediction

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Streamlit UI
    participant APP as app.py logic
    participant SC as StandardScaler
    participant M as SVM Model
    participant PL as Plotly

    U->>UI: Adjust sliders / inputs
    U->>UI: Click "Predict"
    UI->>APP: Trigger prediction block
    APP->>APP: Assemble np.array [[8 features]]
    APP->>SC: transform(input_data)
    SC-->>APP: input_std
    APP->>M: predict(input_std)
    M-->>APP: label (0 / 1)
    APP->>M: predict_proba(input_std)
    alt probabilities supported
        M-->>APP: [p_neg, p_pos]
    else not supported
        APP->>APP: Fallback: 100% / 0% from label
    end
    APP->>APP: Determine risk tier
    APP->>PL: Build gauge(prob_positive)
    PL-->>UI: Figure
    APP->>UI: Banner, metrics, factors, advice
    UI-->>U: Display results
```

### Startup / model-loading sequence

```mermaid
sequenceDiagram
    participant S as Streamlit
    participant L as load_model_and_scaler()
    participant FS as File System
    S->>L: call (cached via st.cache_resource)
    L->>FS: model path exists?
    FS-->>L: yes / no
    L->>FS: scaler path exists?
    FS-->>L: yes / no
    alt both exist
        L->>FS: joblib.load(model), joblib.load(scaler)
        FS-->>L: objects
        L-->>S: (model, scaler, None)
    else missing or load error
        L-->>S: (None, None, error message)
        S->>S: st.error(...) + st.stop()
    end
```

---

## 🎯 Risk Classification Logic

The displayed banner combines the **predicted class** with the **diabetic probability** (`P`):

| Predicted class | Probability of diabetes (P) | Banner |
|---|---|---|
| 0 (Non-diabetic) | P < 30 % | 🟢 LOW RISK – Not Diabetic |
| 0 (Non-diabetic) | P ≥ 30 % | 🟡 MODERATE RISK – Not Diabetic |
| 1 (Diabetic) | P > 70 % | 🔴 HIGH RISK – Diabetic |
| 1 (Diabetic) | P ≤ 70 % | 🟡 MODERATE RISK – Diabetic |

```mermaid
flowchart TD
    A{Prediction = ?} -->|0 Non-diabetic| B{P diabetic < 30%?}
    A -->|1 Diabetic| C{P diabetic > 70%?}
    B -->|Yes| G["🟢 LOW RISK<br/>Not Diabetic"]
    B -->|No| Y1["🟡 MODERATE RISK<br/>Not Diabetic"]
    C -->|Yes| R["🔴 HIGH RISK<br/>Diabetic"]
    C -->|No| Y2["🟡 MODERATE RISK<br/>Diabetic"]
```

### State diagram — risk gauge zones

```mermaid
stateDiagram-v2
    [*] --> LowZone: 0 – 30 %
    [*] --> MidZone: 30 – 70 %
    [*] --> HighZone: 70 – 100 %
    LowZone: 🟢 Green zone
    MidZone: 🟡 Yellow zone
    HighZone: 🔴 Red zone
    LowZone --> [*]
    MidZone --> [*]
    HighZone --> [*]
```

---

## 🧠 Risk Factor & Recommendation Engine

Independent of the ML output, `app.py` applies simple clinical heuristics to the raw inputs to give the user context.

| Input | Condition | Message | Type |
|---|---|---|---|
| Glucose | > 125 mg/dL | 🔴 High glucose level | Risk |
| Glucose | < 100 mg/dL | 🟢 Normal glucose level | Positive |
| BMI | > 30 | 🔴 High BMI | Risk |
| BMI | 18.5 – 24.9 | 🟢 Healthy BMI | Positive |
| Age | > 45 | 🟡 Age-related risk factor | Risk |
| Blood pressure | > 80 mm Hg | 🔴 Elevated blood pressure | Risk |
| Blood pressure | 60 – 80 mm Hg | 🟢 Within selected range | Positive |
| Pedigree function | > 0.5 | 🟡 Higher pedigree function | Risk |

```mermaid
flowchart LR
    IN[Raw inputs] --> G{Glucose}
    IN --> B{BMI}
    IN --> A{Age}
    IN --> P{BP}
    IN --> D{DPF}
    G -->|">125"| R1[Risk list]
    G -->|"<100"| P1[Positive list]
    B -->|">30"| R1
    B -->|"18.5–24.9"| P1
    A -->|">45"| R1
    P -->|">80"| R1
    P -->|"60–80"| P1
    D -->|">0.5"| R1
    R1 --> OUT[⚠️ Identified Risk Factors]
    P1 --> OUT2[✅ Positive Health Indicators]
```

**Recommendations** shown after prediction:

- **Diabetic prediction (1):** consult a healthcare professional, consider screening, monitor glucose as advised, keep a balanced diet and stay active.
- **Non-diabetic prediction (0):** maintain balanced diet, exercise regularly, keep a healthy weight, continue regular check-ups.

---

## 📐 UML & Design Diagrams

### Use-case diagram

```mermaid
flowchart LR
    User((👤 User))
    Dev((👨‍💻 Developer))

    subgraph GlucoPredict System
        UC1([Enter patient data])
        UC2([Run prediction])
        UC3([View risk level & probabilities])
        UC4([View risk gauge])
        UC5([Review risk factors])
        UC6([Read recommendations])
        UC7([Train / retrain model])
        UC8([Export model & scaler])
    end

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
    Dev --> UC7
    Dev --> UC8
    UC2 -. includes .-> UC1
    UC3 -. extends .-> UC2
    UC4 -. extends .-> UC2
```

### Component diagram

```mermaid
flowchart TB
    subgraph app.py
        C1[Page Config & CSS]
        C2[Model Loader]
        C3[Sidebar Input Panel]
        C4[Prediction Engine]
        C5[Gauge Renderer]
        C6[Risk Factor Analyzer]
        C7[Recommendation Module]
    end
    EXT1[(joblib)]
    EXT2[(scikit-learn)]
    EXT3[(Plotly)]
    EXT4[(Streamlit)]
    C2 --> EXT1
    C4 --> EXT2
    C5 --> EXT3
    C1 --> EXT4
    C3 --> EXT4
    C3 --> C4
    C2 --> C4
    C4 --> C5
    C3 --> C6
    C4 --> C7
```

### Class diagram (logical)

```mermaid
classDiagram
    class PatientInput {
        +int pregnancies
        +int glucose
        +int bloodPressure
        +int skinThickness
        +int insulin
        +float bmi
        +float dpf
        +int age
        +toArray() ndarray
    }
    class ModelLoader {
        +Path MODEL_PATH
        +Path SCALER_PATH
        +load_model_and_scaler() tuple
    }
    class StandardScaler {
        +transform(X) ndarray
    }
    class SVMModel {
        +predict(X) int
        +predict_proba(X) ndarray
    }
    class PredictionResult {
        +int label
        +float probNegative
        +float probPositive
        +string riskTier
    }
    class RiskAnalyzer {
        +analyze(PatientInput) tuple
    }
    class Recommender {
        +recommend(label) string
    }
    PatientInput --> StandardScaler : scaled by
    ModelLoader --> StandardScaler : loads
    ModelLoader --> SVMModel : loads
    StandardScaler --> SVMModel : feeds
    SVMModel --> PredictionResult : produces
    PatientInput --> RiskAnalyzer : analysed by
    PredictionResult --> Recommender : drives
```

### Data-flow diagram (Level 1)

```mermaid
flowchart LR
    U((User)) -->|"8 clinical values"| P1[1. Collect Input]
    P1 -->|"feature vector"| P2[2. Scale Features]
    S[(scaler_svm.pkl)] --> P2
    P2 -->|"standardised vector"| P3[3. Classify]
    M[(diabetes_model.pkl)] --> P3
    P3 -->|"label + probabilities"| P4[4. Assess Risk Tier]
    P1 -->|"raw values"| P5[5. Analyse Risk Factors]
    P4 --> P6[6. Compose Report]
    P5 --> P6
    P6 -->|"banner, gauge, advice"| U
```

### Activity diagram — developer retraining workflow

```mermaid
flowchart TD
    S([Start]) --> A[Open notebook]
    A --> B[Load diabetes.csv]
    B --> C[Clean & explore data]
    C --> D[Split data]
    D --> E[Fit scaler]
    E --> F[Train SVM]
    F --> G[Evaluate]
    G --> H{Accuracy acceptable?}
    H -- No --> F
    H -- Yes --> I[joblib.dump model + scaler]
    I --> J[Replace .pkl files next to app.py]
    J --> K[Restart Streamlit app]
    K --> E2([End])
```

### Deployment diagram

```mermaid
flowchart LR
    subgraph User Device
        BR[Web Browser]
    end
    subgraph Host["Host (local machine / Streamlit Community Cloud / VM)"]
        ST[Streamlit Server]
        PY[Python runtime + deps]
        FILES[(app.py • model.pkl • scaler.pkl)]
    end
    BR <-->|HTTP / WebSocket| ST
    ST --> PY --> FILES
    GH[(GitHub Repository)] -. deploy / pull .-> Host
```

### Project timeline (suggested Gantt)

```mermaid
gantt
    title GlucoPredict Development Lifecycle
    dateFormat  YYYY-MM-DD
    section Data
    Collect & inspect dataset      :done, d1, 2026-01-01, 5d
    Clean & EDA                    :done, d2, after d1, 7d
    section Modelling
    Scale + train SVM              :done, m1, after d2, 5d
    Evaluate & tune                :done, m2, after m1, 5d
    Export artefacts               :done, m3, after m2, 2d
    section Application
    Build Streamlit UI             :done, a1, after m3, 7d
    Risk logic & recommendations   :done, a2, after a1, 4d
    section Release
    Documentation & README         :active, r1, after a2, 3d
    Deployment                     :r2, after r1, 3d
```

*(Dates are illustrative placeholders — adjust to your actual timeline.)*

### Mind map

```mermaid
mindmap
  root((GlucoPredict))
    Data
      Pima Indians dataset
      768 samples
      8 features
    Model
      SVM
      StandardScaler
      ~78% accuracy
    App
      Streamlit
      Sidebar inputs
      Plotly gauge
    Output
      Risk tier
      Probabilities
      Risk factors
      Recommendations
    Safety
      Educational only
      Disclaimer
```

---

## 🚀 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/charveemasand108/GlucoPredict-.git
cd GlucoPredict-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

The repository's `requirements.txt` is currently empty. Use the following (pin the versions you trained with, especially **scikit-learn**, so the `.pkl` files load correctly):

```text
streamlit
pandas
numpy
scikit-learn
joblib
plotly
```

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### 5. Using the app

1. Set the patient's **age** and **pregnancies**.
2. Enter **glucose, blood pressure, skin thickness, insulin, BMI** and **pedigree function**.
3. Click **🔮 Predict**.
4. Read the risk banner, probability metrics, gauge, risk factors and recommendations.

### Retraining the model

Open `diabetes-prediction.ipynb`, run all cells, and save the new artefacts:

```python
import joblib
joblib.dump(model,  "diabetes_model.pkl")
joblib.dump(scaler, "scaler_svm.pkl")
```

Keep the feature order: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`.

---



---

## ⚠️ Limitations & Known Issues

- **Not a medical device.** The model is trained on a small, single-population dataset and is for learning only.
- **Dataset scope.** The Pima dataset covers adult women of Pima Indian heritage, aged 21+; results may not generalise to other populations or to men.
- **Zero values.** Sliders allow `0` for glucose, blood pressure, skin thickness and insulin. In this dataset zeros usually mean *missing*, and are not physiologically valid — consider raising the minimum values or adding validation.
- **Probability fallback.** If the SVM was not trained with `probability=True`, `predict_proba` is unavailable and the app shows 0 % / 100 %, which makes the gauge and risk tiers less informative.
- **Hard-coded metrics.** The landing page shows "~78 %" accuracy and "768 samples" as static text rather than values read from the evaluation.
- **Heuristic risk factors.** The risk-factor panel uses fixed thresholds, independent of the model; it is not feature importance. The "BP > 80" rule is a diastolic-style cut-off.
- **Empty `requirements.txt`.** Environment reproduction currently relies on manual installation.
- **Artefact compatibility.** `.pkl` files depend on the scikit-learn version used to create them.
- **No README / licence** in the original repository.

---

## 🔮 Future Improvements

- [ ] Populate `requirements.txt` with pinned versions
- [ ] Add input validation and treat zeros as missing values
- [ ] Compare SVM with Logistic Regression, Random Forest, XGBoost
- [ ] Add model explainability (SHAP / LIME) in place of fixed-threshold rules
- [ ] Probability calibration (`CalibratedClassifierCV`)
- [ ] Display real evaluation metrics (ROC-AUC, precision, recall, confusion matrix)
- [ ] Batch prediction via CSV upload
- [ ] Unit tests and CI with GitHub Actions
- [ ] Dockerise and publish a live demo
- [ ] Multilingual interface

---

## 🤝 Contributing, License & Acknowledgements

**Contributing:** Fork the repo, create a feature branch, commit your changes, and open a pull request.

**License:** No licence has been specified yet. Consider adding one (e.g. MIT) to the repository.

**Acknowledgements**
- Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases) via the UCI / Kaggle repositories
- scikit-learn, Streamlit and Plotly communities

---

**Author:** [@charveemasand108](https://github.com/charveemasand108)
