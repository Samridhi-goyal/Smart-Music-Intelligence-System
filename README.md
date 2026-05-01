# 🎵 Smart Music Intelligence System

An AI-based music recommendation system that predicts mood, estimates popularity, and recommends songs using machine learning techniques.

---

## 🚀 Features

- 🎧 Mood Prediction using Random Forest Classifier  
- 📈 Popularity Prediction using Random Forest Regressor  
- 🎶 Song Recommendation using K-Nearest Neighbors (KNN)  
- ❤️ Personalized Recommendations based on user preferences  
- 🖥️ Interactive UI built with Streamlit  

---

## 🧠 Technologies Used

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Streamlit  
- Matplotlib  

---

## 📊 Models Used

- Random Forest Classifier → Mood Prediction  
- Random Forest Regressor → Popularity Prediction  
- K-Nearest Neighbors (KNN) → Recommendation Engine  

---

## 📂 Project Structure

app.py → Streamlit frontend  
train.py → Model training  
models/ → Saved models (if included)  
data/ → Dataset  
report/ → Conference paper and report  
screenshots/ → Application screenshots  

---

## 📸 Screenshots

### UI Before Prediction  
![UI](screenshots/app1.png)

### Prediction Output  
![Prediction](screenshots/app2.png)

### Recommendations  
![Recommendations](screenshots/app3.png)

---

## ▶️ How to Run the Project

1. Install dependencies:
pip install -r requirements.txt

2. Download large model files from the Releases section

3. Place them in the project directory

4. Run the app:
streamlit run app.py

---

## 📦 Important Note

Some large files such as:
- songs.pkl  
- popularity_model.pkl  

are not included in the main repository due to GitHub file size limitations.

They are available in the Releases section of this repository.

---

## 📌 Conclusion

This project demonstrates how machine learning can enhance music discovery by combining mood prediction with recommendation systems. It provides a simple yet effective solution for personalized music selection.

---

## 👩‍💻 Authors

- Aditi Singh  
- Samridhi  
- Adhya Singh Chauhan  
- Khushi Bisht  
