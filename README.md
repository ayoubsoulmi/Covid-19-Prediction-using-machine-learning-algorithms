🔬 COVID-19 Infection Prediction System (SVM-Based)

A machine-learning system that predicts the likelihood of COVID-19 infection based on patient symptoms and medical indicators.

This project was developed as part of an academic machine-learning research initiative and demonstrates the full lifecycle of a data-driven prediction system: data preprocessing, model training, evaluation, deployment, and GUI-based user interaction.

📘 Overview

The COVID-19 Prediction System uses a Support Vector Machine (SVM) classifier trained on structured symptom data.
The system analyzes 19 clinical features — such as body temperature, cough, breathing issues, and medical history — and predicts whether a patient is likely Negative, Positive, or Undetermined.

A graphical interface built with Anvil allows users to input symptoms and instantly receive predictions from the trained ML model.

🚀 Features
🔍 Machine Learning

SVM classifier as the main prediction model

Cleaned + structured dataset with 19 medical indicators

Preprocessing pipeline for data validation

Performance evaluation using standard ML metrics (Accuracy, Precision, Recall, F1)

🧑‍⚕️ Medical Indicators Included

Age, gender, body temperature

Dry cough, sore throat, weakness, chest pain

Breathing difficulty, drowsiness

Diabetes, heart disease, lung disease

Travel history, loss of smell, and more

🖥 GUI Integration (Anvil)

Simple, user-friendly form

Inputs 19 symptom fields

Sends data to server module

Displays:

Negative

Positive

Undetermined

📡 Server-Side Prediction Module

Connects the GUI to the trained model

Converts user input to a feature vector

Runs prediction logic

Returns human-readable results

🧠 Technologies Used

Python 3

Scikit-Learn – SVM, k-NN, Linear Regression models

Pandas – dataset handling

OpenPyXL – Excel dataset reading

Joblib – model saving/loading

Anvil Uplink – GUI integration

NumPy – numerical operations

📊 Dataset Summary

The dataset contains symptom-based screening data with 19 features and 1 target variable.

Target values:

0 — Negative

1 — Medium/Uncertain

2 — Positive

Only medically relevant indicators were kept in the final model features.

▶️ Usage
1. Launch the GUI

Open the Anvil application and input patient symptoms.

2. Model Processing

The GUI sends all 19 features to the server-side function.

3. Output

The model returns:

Negative

Positive

📈 Evaluation

The SVM classifier was selected because it delivered the best balance of:

Accuracy

Stability

Low overfitting

Performance on binary symptom features

Evaluation included:

Confusion matrix

Classification report

Train/test performance comparison

📖 Testing Environment

The model was tested using sample cases that represent:

Negative patients

Mild-symptom patients

Severe symptoms with high-risk indicators

GUI responses were cross-validated with expected outcomes.

⚠️ Disclaimer

This project is for academic and educational purposes only.
It is not a diagnostic medical tool and must not be used for real clinical decisions.

📄 License

MIT License (recommended).
Can be changed based on your preference.

👤 Author

Ayoub Soulmi
Machine Learning • Data Science • Cyber Security
📧 ayoubsoulmi@gmail.com

🌐 github.com/ayoubsoulmi

Undetermined case

Results appear instantly in the GUI interface.
