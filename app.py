import streamlit as st
import pandas as pd

from src.inference import ServiceGuardInference


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="ServiceGuard-GNN",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# Paths
# ============================================================

MODELS_DIR = "models"
METADATA_PATH = "models/metadata.json"
BASELINES_PATH = "models/baselines.json"


# ============================================================
# Load inference engine
# ============================================================

@st.cache_resource
def load_serviceguard():

    return ServiceGuardInference(
        models_dir=MODELS_DIR,
        metadata_path=METADATA_PATH,
        baselines_path=BASELINES_PATH
    )


serviceguard = load_serviceguard()


# ============================================================
# Header
# ============================================================

st.title("🛡️ ServiceGuard-GNN")

st.markdown(
    """
    **Graph Neural Network based microservice fault localization**

    Analyze a distributed-service trace to identify the most
    likely faulty service and inspect latency-based diagnosis
    and GAT attention relationships.
    """
)


# ============================================================
# Application selection
# ============================================================

applications = list(
    serviceguard.metadata.keys()
)

app_name = st.selectbox(
    "Select application",
    applications
)


# ============================================================
# Trace input
# ============================================================

st.subheader("Trace Input")

uploaded_file = st.file_uploader(
    "Upload a CSV trace",
    type=["csv"]
)

trace = None

if uploaded_file is not None:

    trace = pd.read_csv(
        uploaded_file
    )

    st.write("Uploaded trace:")
    st.dataframe(
        trace,
        use_container_width=True
    )


# ============================================================
# Analyze
# ============================================================

if st.button(
    "🔍 Analyze Trace",
    type="primary"
):

    if trace is None:

        st.error(
            "Please upload a CSV trace first."
        )

    else:

        try:

            result = serviceguard.analyze(
                trace,
                app_name,
                top_k=10
            )

            prediction = result[
                "prediction"
            ]

            diagnosis = result[
                "service_diagnosis"
            ]

            attention = result[
                "attention_relationships"
            ]

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            st.subheader(
                "🎯 Fault Prediction"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Fault",
                    prediction[
                        "predicted_fault"
                    ]
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{prediction['confidence']:.2f}%"
                )

            with col3:

                st.metric(
                    "Predicted Class",
                    prediction[
                        "predicted_class"
                    ]
                )

            # ------------------------------------------------
            # Top predictions
            # ------------------------------------------------

            st.subheader(
                "Top Predictions"
            )

            top_predictions = pd.DataFrame(
                prediction[
                    "top_predictions"
                ]
            )

            st.dataframe(
                top_predictions,
                use_container_width=True
            )

            # ------------------------------------------------
            # Service diagnosis
            # ------------------------------------------------

            st.subheader(
                "🔬 Service Diagnosis"
            )

            st.dataframe(
                diagnosis,
                use_container_width=True
            )

            # ------------------------------------------------
            # Attention
            # ------------------------------------------------

            st.subheader(
                "🧠 GAT Attention Relationships"
            )

            st.dataframe(
                attention,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Analysis failed: {e}"
            )