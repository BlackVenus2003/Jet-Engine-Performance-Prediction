\## Jet Engine Performance Prediction



This project uses a real-world ICAO aircraft engine emissions dataset to predict jet engine thrust using machine learning techniques. The goal is to analyze key engine design parameters and estimate engine thrust output.



\## 🚀 Project Summary



\- \*\*Objective\*\*: Predict engine thrust (kN) from parameters like Bypass Ratio, Overall Pressure Ratio, and more.

\- \*\*Dataset Source\*\*: \[ICAO Aircraft Engine Emissions](https://www.kaggle.com/datasets/ahmedeltom/icao-aircraft-engine-emissions)

\- \*\*Approach\*\*:

&nbsp; - Clean and preprocess emissions dataset

&nbsp; - Select key numerical features (e.g., BPR, OPR, year, mode percentage)

&nbsp; - Train a Gradient Boosting Regressor model

&nbsp; - Evaluate performance using RMSE



\## 📁 Directory Structure



JetPerf\_Prediction/

│

├── data/

│ └── emissions\_raw.csv

│ └── clean\_engine\_perf.csv

│

├── src/

│ ├── make\_dataset.py # Cleans and selects relevant features

│ └── train\_model.py # Trains and evaluates model

│

├── README.md

└── requirements.txt





\## 🧠 Key Features Used



\- Bypass Ratio (BPR)

\- Overall Pressure Ratio (OPR)

\- Engine manufacturing year

\- Mode percent (`mode\_pct`)

\- Target: Rated Thrust (kN)



\## 📊 Model Performance



\- \*\*Model\*\*: Gradient Boosting Regressor

\- \*\*Evaluation Metric\*\*: Root Mean Squared Error (RMSE)

\- \*\*RMSE Achieved\*\*: ~14.2 kN (example, based on current config)



\## 📦 Tech Stack



\- Python 3.12

\- pandas, scikit-learn

\- Jupyter or command-line compatible



\## 🛠️ How to Run



1\. Place your `emissions\_raw.csv` in the `data/` folder.

2\. Clean and process the dataset:



```bash

python src/make\_dataset.py

Train and evaluate the model:

python src/train\_model.py

```



📈 Possible Improvements

Feature engineering for combustion efficiency or ambient condition factors



Model tuning with GridSearchCV or Random Forest



Interactive dashboards with Plotly or Streamlit



\## Author: Paramjyot Tiwana



