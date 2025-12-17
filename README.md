# 🎗️ Breast Cancer Prediction Model

A machine learning–based web application that predicts whether a breast tumor is **Benign** or **Malignant** using clinical features extracted from digitized images of a breast mass.
The app is built using **Python**, **Scikit-learn**, and **Streamlit**, and supports both **manual input** and **CSV file upload**.

---

## 🚀 Live Demo

🔗 **Streamlit App:**
https://breastcancerpredictionmodel-njqtykfptq3n5hfyzbuzsi.streamlit.app/

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early detection plays a crucial role in effective treatment.
This project uses a **Logistic Regression model** trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset** to assist in tumor classification.

The application allows users to:

* Manually enter tumor feature values
* Upload a CSV file containing feature data
* Get real-time predictions with confidence scores

---

## 🧠 Machine Learning Model

* **Algorithm:** Logistic Regression
* **Dataset:** Breast Cancer Wisconsin (Diagnostic Dataset)
* **Target Variable:**
  * `0` → Malignant
  * `1` → Benign

The trained model is stored as a pickle file and loaded directly into the Streamlit app.

---

## 📊 Features Used

The model uses **30 numerical features**, grouped into:

* **Mean features** (e.g., radius_mean, texture_mean)
* **Standard error features** (e.g., radius_se, texture_se)
* **Worst features** (e.g., radius_worst, texture_worst)

These features describe tumor size, texture, shape, and complexity.

---

## 🖥️ Application Features

* 📥 **CSV Upload (Optional)**
  Upload a dataset with the required feature columns. A random row is selected automatically.

* ✍️ **Manual Feature Input**
  Enter or adjust all feature values using input fields.

* 🔍 **Real-Time Prediction**
  Predicts whether the tumor is **Benign** or **Malignant**.

* 📈 **Prediction Confidence**
  Displays model confidence (if supported by the classifier).

* ☁️ **Deployed on Streamlit Cloud**

---

## 🛠️ Tech Stack

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Streamlit**

---

## 📂 Project Structure

```
Breast_Cancer_Prediction/
│
├── app.py                  # Streamlit application
├── best_ml_model.pkl       # Trained ML model
├── requirements.txt        # Required dependencies
├── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup (Local)

1. **Clone the repository**

```bash
git clone https://github.com/parshva106/Breast_Cancer_Prediction_Model.git
cd breast-cancer-prediction
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## ⚠️ Disclaimer

This application is intended **for educational and demonstration purposes only**.
It should **not be used for medical diagnosis or treatment decisions**.

---

## 👤 Author

**Parshva Mehta**
B.Tech (Electronics & Telecommunication)
Aspiring Data Analyst / Data Scientist

📌 GitHub: https://github.com/parshva106

---

## ⭐ If you like this project

Don’t forget to **star ⭐ the repository** and share your feedback!

---

