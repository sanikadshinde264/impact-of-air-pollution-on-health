# 🌍 Impact Of Air Pollution On Health

Predicts the health impact score based on air quality, environmental conditions, and health-related factors. Built with Streamlit and a scikit-learn Linear Regression pipeline.

## 📌 Overview

This project uses a Machine Learning regression model to predict the potential **Health Impact Score** associated with air pollution and environmental conditions. Users can provide air-quality and health-related parameters through an interactive Streamlit web application and receive a predicted health impact score.

## ❓ Problem Statement

Air pollution has a significant impact on human health, and its effects can vary depending on pollutant levels, environmental conditions, and health-related factors. Estimating the potential health impact from multiple parameters can be difficult manually.

This project builds a data-driven Machine Learning model that learns from historical air-quality and health-impact data to provide a quick and consistent prediction of the **Health Impact Score**.

## ✨ Features

* Enter air-quality and environmental parameters through an interactive web interface
* Predict Health Impact Score using a trained Linear Regression model
* Uses a pre-trained scikit-learn pipeline for prediction
* Dataset-based input values for consistent predictions
* Simple and user-friendly Streamlit interface
* Fast prediction results without retraining the model

## 📊 Dataset

* **File:** `cleaned_air_quality_health_impact_data.csv`
* **Target:** Health Impact Score
* **Format:** CSV
* **Type:** Cleaned air-quality and health-impact dataset

### Input Features

| Feature                | Description                    |
| ---------------------- | ------------------------------ |
| `AQI`                  | Air Quality Index              |
| `PM10`                 | Particulate Matter 10          |
| `PM2.5`                | Fine Particulate Matter        |
| `NO2`                  | Nitrogen Dioxide               |
| `SO2`                  | Sulfur Dioxide                 |
| `O3`                   | Ozone                          |
| `Temperature`          | Environmental temperature      |
| `Humidity`             | Relative humidity              |
| `Wind Speed`           | Wind speed                     |
| `Respiratory Cases`    | Number of respiratory cases    |
| `Cardiovascular Cases` | Number of cardiovascular cases |
| `Hospital Admissions`  | Number of hospital admissions  |
| `Health Impact Class`  | Health impact classification   |

The model predicts:

```text
Health Impact Score
```

## 🛠️ Tools & Technologies

* **Python 3** — programming language
* **Streamlit** — web application and user interface
* **pandas / numpy** — data handling and processing
* **scikit-learn** — Machine Learning and Linear Regression
* **pickle** — loading the trained Machine Learning model
* **HTML / Streamlit Components** — application interface

## ⚙️ Methodology

1. Load the cleaned air-quality and health-impact dataset (`cleaned_air_quality_health_impact_data.csv`)
2. Prepare the required environmental and health-related input features
3. Train a `LinearRegression` Machine Learning model
4. Build a scikit-learn pipeline for prediction
5. Save the trained pipeline as `air_quality_health_model.pkl`
6. Load the trained model in the Streamlit application (`app.py`)
7. Accept user input through the web interface
8. Use `model.predict()` to generate the predicted Health Impact Score

## 🗂️ Project Directory Structure

```text
impact-of-air-pollution-on-health/
├── app.py                                  # Streamlit application
├── cleaned_air_quality_health_impact_data.csv  # Dataset
├── air_quality_health_model.pkl            # Trained ML model
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
└── .gitignore                               # Git ignored files
```

## 🖥️ Dashboard / Output

* **Home / Information page** → provides information about air pollution and its health impact
* **Prediction section** → allows users to enter/select air-quality and health parameters
* **Machine Learning model** → processes the provided input values
* **Result section** → displays the predicted Health Impact Score

## 💡 Key Insights

* Higher levels of air pollutants can be associated with increased health risks
* PM2.5 and PM10 are important indicators of particulate pollution
* NO₂, SO₂, and O₃ contribute to overall air-quality conditions
* Environmental factors such as temperature, humidity, and wind speed can influence air pollution conditions
* Respiratory and cardiovascular health indicators provide additional information about potential health impacts
* A Machine Learning model can help identify relationships between air-quality conditions and health-impact scores
* The Linear Regression model provides a simple and interpretable baseline for prediction

## ✅ Results & Conclusion

The Linear Regression pipeline provides a fast and interpretable approach for predicting the **Health Impact Score** from air-quality, environmental, and health-related parameters.

The project demonstrates how Machine Learning can be applied to environmental and healthcare-related data to generate predictive insights through an interactive web application.

The model is intended primarily as an **educational and demonstration project**. Its predictions should not be considered medical diagnoses or professional healthcare advice.

## ▶️ How to Run the Project

```bash
# 1. Clone the repository
git clone https://github.com/sanikadshinde264/impact-of-air-pollution-on-health.git

# 2. Navigate to the project directory
cd impact-of-air-pollution-on-health

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app.py

# 5. Open the application
http://localhost:8501
```

## 🚀 Future Work

* Try advanced Machine Learning models such as Random Forest and XGBoost
* Compare multiple regression algorithms
* Improve model accuracy through hyperparameter tuning
* Add interactive air-quality visualizations
* Add real-time AQI data using an external API
* Add location-based air-quality analysis
* Add prediction history and downloadable reports
* Deploy the application using Streamlit Cloud
* Add model performance metrics such as MAE, MSE, RMSE, and R² score

## 👤 Author and Contact

**Sanika Shinde** <br>
📧 [sanikadshinde264@gmail.com](mailto:sanikadshinde264@gmail.com) | 🔗 [linkedin.com/in/sanikadshinde264](https://www.linkedin.com/in/sanikadshinde264)
