# Air Pollution & Health Impact Prediction

## Description

This project predicts a **Health Impact Score** from air quality and
environmental data using a pre-trained machine-learning model. It was
originally built as a Flask web application and has been converted into a
single-file **Streamlit** application for a simpler, more modern interface.

The model itself is unchanged — this conversion only replaces the
Flask/HTML/Jinja interface layer with Streamlit. No retraining, no changes
to feature names, order, or the underlying algorithm.

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Machine Learning (Linear Regression pipeline)

## Input Features

The model expects exactly these 13 features, in this order:

1. `AQI`
2. `PM10`
3. `PM2_5`
4. `NO2`
5. `SO2`
6. `O3`
7. `temperature`
8. `humidity`
9. `wind_speed`
10. `respiratory_cases`
11. `cardiovascular_cases`
12. `hospital_admissions`
13. `health_impact_class`

Dropdown options for each field are populated dynamically from the unique
values found in `cleaned_air_quality_health_impact_data.csv`.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (typically
`http://localhost:8501`).

## Model

The application loads the existing trained model from
`air_quality_health_model.pkl` using `pickle`, cached with
`@st.cache_resource` so it's only loaded once per session. The model is a
scikit-learn `Pipeline` (a passthrough `ColumnTransformer` followed by
`LinearRegression`) and is used exactly as originally trained — it is never
retrained or replaced.

The dataset `cleaned_air_quality_health_impact_data.csv` is loaded with
`@st.cache_data` and used only to populate the dropdown selectors on the
prediction page; it is never modified.

## Validation

The app was verified against the original Flask app's behavior using this
input combination:

| Parameter | Value |
|---|---|
| AQI | 3 |
| PM10 | 3 |
| PM2_5 | 2 |
| NO2 | 3 |
| SO2 | 2 |
| O3 | 13 |
| Temperature | 13 |
| Humidity | 23 |
| Wind Speed | 16 |
| Respiratory Cases | 19 |
| Cardiovascular Cases | 13 |
| Hospital Admissions | 6 |
| Health Impact Class | 2 |

Both the original Flask app and this Streamlit app produce a predicted
Health Impact Score of **≈ 28.42**.

## Disclaimer

This is an educational machine-learning project. Predictions are generated
by a statistical model trained on a sample dataset and **should not be
considered medical advice or a diagnostic tool**.
