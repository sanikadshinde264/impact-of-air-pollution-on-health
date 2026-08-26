"""
Air Pollution & Health Impact Prediction
-----------------------------------------
Streamlit application (converted from the original Flask app).

The trained model (air_quality_health_model.pkl) and dataset
(cleaned_air_quality_health_impact_data.csv) are used exactly as-is.
No retraining, no changes to feature names/order.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Air Pollution Health Impact Prediction",
    page_icon="🌍",
    layout="wide",
)

MODEL_PATH = "air_quality_health_model.pkl"
DATA_PATH = "cleaned_air_quality_health_impact_data.csv"

# Exact 13 columns / order the model was trained on (from the original Flask app)
FEATURE_COLUMNS = [
    "AQI",
    "PM10",
    "PM2_5",
    "NO2",
    "SO2",
    "O3",
    "temperature",
    "humidity",
    "wind_speed",
    "respiratory_cases",
    "cardiovascular_cases",
    "hospital_admissions",
    "health_impact_class",
]

# Friendlier labels for the UI only (does not affect model input)
FIELD_LABELS = {
    "AQI": "AQI",
    "PM10": "PM10",
    "PM2_5": "PM2.5",
    "NO2": "NO2",
    "SO2": "SO2",
    "O3": "O3",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "wind_speed": "Wind Speed",
    "respiratory_cases": "Respiratory Cases",
    "cardiovascular_cases": "Cardiovascular Cases",
    "hospital_admissions": "Hospital Admissions",
    "health_impact_class": "Health Impact Class",
}

# Grouping of the same 13 fields into dashboard categories (display only)
CATEGORY_GROUPS = [
    {
        "title": "Air Quality",
        "icon": "🌫️",
        "fields": ["AQI", "PM10", "PM2_5", "NO2", "SO2", "O3"],
    },
    {
        "title": "Environmental Conditions",
        "icon": "🌡️",
        "fields": ["temperature", "humidity", "wind_speed"],
    },
    {
        "title": "Health Indicators",
        "icon": "🏥",
        "fields": [
            "respiratory_cases",
            "cardiovascular_cases",
            "hospital_admissions",
            "health_impact_class",
        ],
    },
]

# --------------------------------------------------------------------------
# Custom styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #14532d;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .info-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, #0f172a 0%, #14532d 100%);
        padding: 0.9rem 1.6rem;
        border-radius: 12px 12px 0 0;
        margin-bottom: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .navbar-brand span.icon {
        font-size: 1.5rem;
        color: #22C55E;
    }
    div[data-testid="stButton"] button,
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #F1F5F9 !important;
        color: #334155 !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.1rem !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
    }
    div[data-testid="stButton"] button:hover,
    .stButton > button:hover {
        background-color: #15803D !important;
        color: #ffffff !important;
        border-color: #15803D !important;
    }
    div[data-testid="stButton"] button[kind="primary"],
    .stButton > button[kind="primary"] {
        background-color: #16A34A !important;
        color: #ffffff !important;
        border: 1px solid #16A34A !important;
        font-weight: 700 !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background-color: #15803D !important;
        border-color: #15803D !important;
        color: #ffffff !important;
    }
    div[data-testid="stMarkdown"]:has(.navbar) + div[data-testid="stHorizontalBlock"] {
        background: linear-gradient(90deg, #0f172a 0%, #14532d 100%);
        border-radius: 0 0 12px 12px;
        padding: 0.6rem 1rem 0.9rem 1rem;
        margin-top: -1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    div[data-testid="stSelectbox"] label p {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        min-height: 2.1rem !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 2.1rem !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
        border-left: 4px solid #16A34A !important;
        margin-bottom: 1.4rem;
    }
    .section-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #14532d;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---------------------------------------------------------------- */
    /* Dashboard-style prediction result page                            */
    /* ---------------------------------------------------------------- */
    .result-wrap {
        background-color: #F8FAFC;
        padding: 0.2rem 0 0.4rem 0;
    }
    .result-card-pro {
        background-color: #ECFDF5;
        border: 1.5px solid #16A34A;
        border-radius: 16px;
        padding: 1.4rem 1.8rem 1.2rem 1.8rem;
        max-width: 480px;
        margin: 0 auto 1.6rem auto;
        text-align: center;
        box-shadow: 0 4px 14px rgba(22,163,74,0.12);
    }
    .result-card-pro .result-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .result-card-pro .result-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    .result-card-pro .result-score {
        font-size: 3rem;
        font-weight: 800;
        color: #047857;
        line-height: 1.1;
        margin-bottom: 0.6rem;
    }
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background-color: #ffffff;
        border: 1px solid #A7F3D0;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
    }
    .result-badge .dot {
        color: #16A34A;
        font-size: 0.7rem;
    }

    .details-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: #14532D;
        margin: 0.4rem 0 0.8rem 0;
    }

    .category-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.9rem 1.1rem 0.6rem 1.1rem;
        box-shadow: 0 1px 4px rgba(15,23,42,0.05);
        margin-bottom: 1rem;
        height: 100%;
    }
    .category-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #166534;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding-bottom: 0.5rem;
        margin-bottom: 0.35rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .kv-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.28rem 0;
        font-size: 0.86rem;
        border-bottom: 1px dashed #F1F5F9;
    }
    .kv-row:last-child {
        border-bottom: none;
    }
    .kv-key {
        color: #64748B;
        font-weight: 500;
    }
    .kv-val {
        color: #0F172A;
        font-weight: 700;
    }

    .info-box-pro {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #16A34A;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-top: 0.6rem;
    }
    .info-box-pro .info-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #14532D;
        margin-bottom: 0.3rem;
    }
    .info-box-pro .info-text {
        font-size: 0.85rem;
        color: #334155;
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    .info-box-pro .info-disclaimer {
        font-size: 0.78rem;
        color: #64748B;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


model = load_model()
dataset = load_data()

# --------------------------------------------------------------------------
# Top navigation bar
# --------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown(
    """
    <div class="navbar">
        <div class="navbar-brand"><span class="icon">🧭</span> AirHealth Insights</div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_spacer, nav_home, nav_predict = st.columns([5, 2, 2])
with nav_home:
    if st.button("Home", type=("primary" if st.session_state.page == "Home" else "secondary"), use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()
with nav_predict:
    if st.button(
        "Predict Health Impact",
        type=("primary" if st.session_state.page == "Predict Health Impact" else "secondary"),
        use_container_width=True,
    ):
        st.session_state.page = "Predict Health Impact"
        st.rerun()

page = st.session_state.page

# --------------------------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------------------------
if page == "Home":
    st.markdown('<div class="main-title">Air Pollution & Health Impact Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Understanding how air quality conditions relate to public health outcomes.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
            <h4>🌫️ What is Air Pollution?</h4>
            <p>Air pollution is the presence of harmful substances — gases, particulates,
            and chemical compounds — in the atmosphere at levels that can harm human health,
            ecosystems, and the climate. Common sources include vehicle emissions, industrial
            activity, construction dust, and the burning of fossil fuels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-card">
            <h4>🫁 How Air Pollution Affects Health</h4>
            <p>Prolonged exposure to polluted air is linked to respiratory conditions
            (such as asthma and bronchitis), cardiovascular disease, reduced lung function,
            and increased hospital admissions — particularly among children, the elderly,
            and people with pre-existing conditions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
            <h4>📊 What is AQI?</h4>
            <p>The Air Quality Index (AQI) is a standardized scale used to communicate how
            polluted the air currently is. Higher AQI values indicate greater health risk,
            combining measurements of several pollutants into a single number.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-card">
            <h4>🧪 Why PM10, PM2.5, NO2, SO2 and O3 Matter</h4>
            <p><b>PM10</b> and <b>PM2.5</b> are fine particulate matter that can penetrate deep
            into the lungs and bloodstream. <b>NO2</b> and <b>SO2</b> are gases largely produced
            by traffic and industrial combustion that irritate airways. <b>O3</b> (ground-level
            ozone) can trigger and worsen respiratory symptoms. Each is tracked individually
            because they affect the body in different ways.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-card">
        <h4>🌡️ Environmental Conditions</h4>
        <p>Temperature, humidity, and wind speed influence how pollutants disperse or
        accumulate in the air. Stagnant, humid, or hot conditions can trap pollutants
        closer to the ground, intensifying their health impact, while stronger winds
        tend to disperse them.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
        <h4>🤖 What This Project Predicts</h4>
        <p>This project uses a machine-learning model trained on historical air quality
        and health-outcome data to estimate a <b>Health Impact Score</b> based on
        pollution levels, environmental conditions, and reported case counts. Use the
        <b>Predict Health Impact</b> tab above to try it out.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------------------------------
elif page == "Predict Health Impact":
    st.markdown('<div class="main-title">Find Health Impact Score For Your Input</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Select or enter values for each parameter below, then click predict.</div>',
        unsafe_allow_html=True,
    )

    if dataset is None:
        st.error(f"Dataset file not found. Please make sure '{DATA_PATH}' is in the project folder.")
    elif model is None:
        st.error(f"Model file not found. Please make sure '{MODEL_PATH}' is in the project folder.")
    else:
        user_inputs = {}
        missing_fields = []

        def render_field_row(features, n_cols=4):
            """Render a row of selectboxes in an n_cols-wide grid, filling only
            as many columns as there are features so every field stays the
            same width across rows/sections."""
            row_cols = st.columns(n_cols)
            for i, feature in enumerate(features):
                with row_cols[i]:
                    options = sorted(dataset[feature].dropna().unique().tolist())
                    field_label = FIELD_LABELS.get(feature, feature)
                    selected = st.selectbox(
                        field_label,
                        options,
                        index=None,
                        placeholder=f"Select {field_label}",
                        key=f"input_{feature}",
                    )
                    user_inputs[feature] = selected
                    if selected is None:
                        missing_fields.append(field_label)

        # ---------------- Air Quality ----------------
        with st.container(border=True):
            st.markdown('<div class="section-heading">🌫️ Air Quality</div>', unsafe_allow_html=True)
            render_field_row(["AQI", "PM10", "PM2_5", "NO2"])
            render_field_row(["SO2", "O3"])

        # ---------------- Environmental Conditions ----------------
        with st.container(border=True):
            st.markdown('<div class="section-heading">🌡️ Environmental Conditions</div>', unsafe_allow_html=True)
            render_field_row(["temperature", "humidity", "wind_speed"])

        # ---------------- Health Indicators ----------------
        with st.container(border=True):
            st.markdown('<div class="section-heading">🫁 Health Indicators</div>', unsafe_allow_html=True)
            render_field_row(["respiratory_cases", "cardiovascular_cases", "hospital_admissions"])
            render_field_row(["health_impact_class"])

        st.markdown("")
        btn_col1, btn_col2, btn_col3 = st.columns([3, 2, 3])
        with btn_col2:
            predict_clicked = st.button("🔮 Predict Score", type="primary", use_container_width=True)

        if predict_clicked and missing_fields:
            st.warning(f"Please select a value for: {', '.join(missing_fields)}")
        elif predict_clicked:
            try:
                # Build the input DataFrame with exact column names/order the model expects
                input_df = pd.DataFrame(
                    [[user_inputs[col] for col in FEATURE_COLUMNS]],
                    columns=FEATURE_COLUMNS,
                )

                # Ensure numeric dtypes (matches original dataset dtypes)
                for col in FEATURE_COLUMNS:
                    input_df[col] = pd.to_numeric(input_df[col])

                raw_result = model.predict(input_df)

                # Safely extract a scalar from whatever shape the model returns
                prediction = np.array(raw_result).reshape(-1)[0]

                st.markdown('<div class="result-wrap">', unsafe_allow_html=True)

                # ---------------- Prediction Result Card ----------------
                st.markdown(
                    f"""
                    <div class="result-card-pro">
                        <div class="result-label">Prediction Result</div>
                        <div class="result-heading">Health Impact Score</div>
                        <div class="result-score">{prediction:.2f}</div>
                        <div class="result-badge"><span class="dot">●</span> ML Model Prediction</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---------------- Prediction Details ----------------
                st.markdown('<div class="details-heading">Prediction Details</div>', unsafe_allow_html=True)

                detail_cols = st.columns(3)
                for col, group in zip(detail_cols, CATEGORY_GROUPS):
                    with col:
                        rows_html = ""
                        for feature in group["fields"]:
                            label = FIELD_LABELS.get(feature, feature)
                            value = user_inputs[feature]
                            rows_html += (
                                f'<div class="kv-row">'
                                f'<span class="kv-key">{label}</span>'
                                f'<span class="kv-val">{value}</span>'
                                f'</div>'
                            )
                        st.markdown(
                            f"""
                            <div class="category-card">
                                <div class="category-title">{group['icon']} {group['title']}</div>
                                {rows_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # ---------------- Prediction Interpretation ----------------
                st.markdown(
                    """
                    <div class="info-box-pro">
                        <div class="info-title">Prediction Information</div>
                        <div class="info-text">This health impact score is generated by the trained machine-learning
                        model using the environmental and health-related inputs provided above.</div>
                        <div class="info-disclaimer">For educational and project purposes only. This prediction is
                        not medical advice.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error("Something went wrong while generating the prediction. Please check your inputs and try again.")
                with st.expander("Technical details (for debugging)"):
                    st.exception(e)