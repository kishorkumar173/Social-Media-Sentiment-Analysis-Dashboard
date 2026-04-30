# 🚀 Social Media Sentiment Analysis Dashboard

## 📌 Overview

This project is a **Machine Learning-powered dashboard** that analyzes social media text (tweets, comments, reviews) and classifies sentiment into:

* 😊 Positive
* 😡 Negative
* 😐 Neutral

It simulates how real companies analyze customer feedback and public opinion in real time.

---

## 💡 Problem Statement

Companies receive thousands of comments daily from social media platforms.
Manually analyzing this data is time-consuming and inefficient.

This project solves that problem by:

* Automating sentiment detection
* Providing real-time insights
* Visualizing data for decision-making

---

## 🏢 Industry Relevance

Sentiment analysis is widely used by companies like:

* Amazon, Flipkart → Product reviews
* Swiggy, Zomato → Customer feedback
* Netflix → Content reactions

This project replicates a **real-world business use case**.

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, NLTK
* **Machine Learning:** Scikit-learn
* **Feature Engineering:** TF-IDF
* **Model:** Logistic Regression
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Dashboard:** Streamlit

---

## 🏗️ Project Architecture

```text
Raw Data → Cleaning → Preprocessing → TF-IDF → ML Model → Prediction → Dashboard
```

---

## 📂 Folder Structure

```text
Social-Media-Sentiment-Analysis-Dashboard/
│
├── data/              # Raw & processed datasets
├── src/               # ML pipeline (preprocessing, training, prediction)
├── models/            # Saved model & vectorizer
├── app/               # Streamlit dashboard
├── outputs/           # Graphs & results
├── images/            # Screenshots
├── README.md
├── requirements.txt
└── main.py
```

---

## 🔍 Dataset

This project uses the **Twitter US Airline Sentiment Dataset**, which contains real customer tweets labeled as positive, negative, and neutral.

---

## ▶️ How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/kishorkumar173git add README.md/Social-Media-Sentiment-Analysis-Dashboard.git
cd Social-Media-Sentiment-Analysis-Dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Model Training (Optional)

```bash
python src/train_model.py
```

### 4. Launch Dashboard

```bash
streamlit run app/app.py
```

---

## 📊 Features

* ✅ Real-time sentiment prediction
* ✅ CSV upload for bulk analysis
* ✅ Interactive Pie & Bar Charts
* ✅ KPI Metrics (Positive / Negative / Neutral counts)
* ✅ Download results as CSV
* ✅ Clean and user-friendly UI

---

## 📸 Screenshots

## 📸 Demo

![Dashboard](images/dashboard.png)
![Prediction](images/prediction.png)
![Charts](images/pie_chart.png)

---

## 📈 Results

* Achieved ~80–90% accuracy
* Successfully classified social media sentiments
* Visualized insights for better decision-making

---

## 🎯 Learning Outcomes

* Natural Language Processing (NLP)
* Text preprocessing & cleaning
* Feature extraction (TF-IDF)
* Machine Learning model training
* Model evaluation (accuracy, confusion matrix)
* Dashboard development using Streamlit

---

## 🚀 Future Improvements

* Integrate real-time Twitter API
* Use Deep Learning models (BERT, LSTM)
* Add multilingual sentiment analysis
* Deploy on cloud (Streamlit Cloud / AWS)

---

## 👨‍💻 Author

**Kishor**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
