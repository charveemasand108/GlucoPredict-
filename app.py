import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding: 0rem 1rem;
}

.stAlert {
    padding: 1rem;
    border-radius: 0.5rem;
}

h1 {
    color: #1f77b4;
    padding-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏥 Diabetes Prediction System")
st.markdown("### AI-Powered Risk Assessment Tool")


# --------------------------------------------------
# FIND MODEL FILES
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "diabetes_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler_svm.pkl"


# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

@st.cache_resource
def load_model_and_scaler():

    if not MODEL_PATH.exists():
        return None, None, "diabetes_model.pkl is missing."

    if not SCALER_PATH.exists():
        return None, None, "scaler_svm.pkl is missing."

    try:

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        return model, scaler, None

    except Exception as e:

        return None, None, str(e)


model, scaler, error = load_model_and_scaler()


# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if model is None or scaler is None:

    st.error("❌ Model files not found!")

    st.write("The application is looking for these files:")

    st.code("""
diabetes_model.pkl
scaler_svm.pkl
""")

    st.write("Make sure both files are inside the same folder as `app.py`.")

    st.write("Your folder should look like:")

    st.code("""
GlucoPredict/
│
├── app.py
├── diabetes_model.pkl
├── scaler_svm.pkl
├── diabetes.csv
├── diabetes-prediction.ipynb
├── requirements.txt
└── .venv/
""")

    st.error(f"Reason: {error}")

    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Patient Information")


st.sidebar.subheader("Demographics")

age = st.sidebar.slider(
    "Age",
    min_value=21,
    max_value=100,
    value=30
)

pregnancies = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=0
)


st.sidebar.subheader("Medical Measurements")


glucose = st.sidebar.slider(
    "Glucose (mg/dL)",
    min_value=0,
    max_value=200,
    value=120
)

bp = st.sidebar.slider(
    "Blood Pressure (mm Hg)",
    min_value=0,
    max_value=130,
    value=70
)

skin = st.sidebar.slider(
    "Skin Thickness (mm)",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.sidebar.slider(
    "Insulin (mu U/ml)",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

dpf = st.sidebar.slider(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=2.5,
    value=0.5,
    step=0.01
)


# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------

st.sidebar.markdown("---")

predict_btn = st.sidebar.button(
    "🔮 Predict",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_btn:

    # IMPORTANT:
    # Order must match the order used while training
    input_data = np.array([[
        pregnancies,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    try:

        # Standardize input
        input_std = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_std)[0]


        # --------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------

        try:

            probability = model.predict_proba(input_std)[0]

            prob_negative = probability[0] * 100
            prob_positive = probability[1] * 100

        except Exception:

            prob_positive = 100 if prediction == 1 else 0
            prob_negative = 100 - prob_positive


        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        st.markdown("---")

        st.header("🎯 Prediction Results")


        col1, col2 = st.columns([2, 1])


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        with col1:

            if prediction == 0:

                if prob_positive < 30:

                    st.success(
                        "### 🟢 LOW RISK - Not Diabetic"
                    )

                else:

                    st.warning(
                        "### 🟡 MODERATE RISK - Not Diabetic"
                    )

            else:

                if prob_positive > 70:

                    st.error(
                        "### 🔴 HIGH RISK - Diabetic"
                    )

                else:

                    st.warning(
                        "### 🟡 MODERATE RISK - Diabetic"
                    )


            # --------------------------------------------------
            # PROBABILITY
            # --------------------------------------------------

            st.subheader("Probability Breakdown")


            pcol1, pcol2 = st.columns(2)


            pcol1.metric(
                "Non-Diabetic",
                f"{prob_negative:.1f}%"
            )


            pcol2.metric(
                "Diabetic",
                f"{prob_positive:.1f}%"
            )


        # --------------------------------------------------
        # GAUGE
        # --------------------------------------------------

        with col2:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_positive,
                    title={
                        "text": "Diabetes Risk"
                    },
                    number={
                        "suffix": "%"
                    },
                    gauge={

                        "axis": {
                            "range": [0, 100]
                        },

                        "bar": {
                            "color": "darkblue"
                        },

                        "steps": [

                            {
                                "range": [0, 30],
                                "color": "lightgreen"
                            },

                            {
                                "range": [30, 70],
                                "color": "yellow"
                            },

                            {
                                "range": [70, 100],
                                "color": "red"
                            }

                        ]

                    }
                )
            )


            fig.update_layout(
                height=300,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20
                )
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # --------------------------------------------------
        # RISK FACTORS
        # --------------------------------------------------

        st.markdown("---")

        st.subheader("⚠️ Risk Factor Analysis")


        risk_factors = []
        positive_factors = []


        if glucose > 125:

            risk_factors.append(
                "🔴 High Glucose Level (>125 mg/dL)"
            )

        elif glucose < 100:

            positive_factors.append(
                "🟢 Normal Glucose Level"
            )


        if bmi > 30:

            risk_factors.append(
                "🔴 High BMI (>30)"
            )

        elif 18.5 <= bmi <= 24.9:

            positive_factors.append(
                "🟢 Healthy BMI"
            )


        if age > 45:

            risk_factors.append(
                "🟡 Age-related risk factor"
            )


        if bp > 80:

            risk_factors.append(
                "🔴 Elevated Blood Pressure"
            )

        elif 60 <= bp <= 80:

            positive_factors.append(
                "🟢 Blood Pressure within selected range"
            )


        if dpf > 0.5:

            risk_factors.append(
                "🟡 Higher Diabetes Pedigree Function"
            )


        # --------------------------------------------------
        # DISPLAY RISK FACTORS
        # --------------------------------------------------

        if risk_factors:

            st.warning("**Identified Risk Factors:**")

            for factor in risk_factors:

                st.markdown(
                    f"- {factor}"
                )


        if positive_factors:

            st.success("**Positive Health Indicators:**")

            for factor in positive_factors:

                st.markdown(
                    f"- {factor}"
                )


        # --------------------------------------------------
        # RECOMMENDATIONS
        # --------------------------------------------------

        st.markdown("---")

        st.subheader("💡 Recommendations")


        if prediction == 1:

            st.error("""
**Recommended Actions:**

- Consult a qualified healthcare professional
- Consider appropriate diabetes screening
- Monitor blood glucose as advised by a professional
- Maintain a balanced diet and regular physical activity
""")


        else:

            st.success("""
**Healthy Practices:**

- Maintain a balanced diet
- Exercise regularly
- Maintain a healthy weight
- Continue regular health check-ups
""")


        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.markdown("---")

        st.warning("""
**⚠️ MEDICAL DISCLAIMER**

This application is an educational machine-learning project.
Its prediction should NOT be considered a medical diagnosis
or a substitute for professional medical advice.

Always consult a qualified healthcare professional for
medical concerns.
""")


    except Exception as e:

        st.error("❌ Prediction Error")

        st.code(str(e))

        st.info("""
Please make sure that:

1. The model was trained using 8 features.
2. The feature order is:
   Pregnancies, Glucose, BloodPressure, SkinThickness,
   Insulin, BMI, DiabetesPedigreeFunction, Age

3. The same StandardScaler used during training was saved
   as scaler_svm.pkl.
""")


# --------------------------------------------------
# INITIAL PAGE
# --------------------------------------------------

else:

    st.markdown("---")

    st.info(
        "👈 Enter patient information in the sidebar "
        "and click **Predict**."
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Model Type",
        "SVM"
    )


    col2.metric(
        "Accuracy",
        "~78%"
    )


    col3.metric(
        "Dataset",
        "768 Samples"
    )