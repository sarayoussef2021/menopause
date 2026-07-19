# 🩸 Menopause Stage Predictor

Menopause Stage Predictor is a free, open-source web application that predicts whether a woman is premenopausal, perimenopausal, or postmenopausal. It uses simple health data like age, depression score, hormones, and sleep.

**We believe every woman deserves to understand her body.**

---

## 🌟 Why this matters

Millions of women struggle with menopause symptoms like fatigue, mood changes, and irregular cycles. Yet, they often wait years for a clear answer. This tool gives women the power to understand their stage and have better conversations with their doctors.

---

## ✨ Features

- 🔮 **Smart Prediction** – predicts menopause stage with 92% accuracy
- 🧠 **Explainable AI** – SHAP shows *why* the prediction was made
- 💬 **Personalized Advice** – OpenAI gives caring, practical tips in simple language
- 🤖 **Health Chatbot** – ask any question about your results or symptoms
- 📊 **Clear Visuals** – easy-to-understand charts and graphs
- ❤️ **Psychological Support** – depression score included and mental health matters

---

## 🛠️ How it works

1. Enter your health data (age, depression, hormones, sleep)
2. AI predicts your menopause stage
3. SHAP explains the key factors
4. OpenAI gives personalized advice
5. Chatbot answers your questions

---

## 📊 Results

| Metric | Result |
| :--- | :--- |
| **Accuracy** | 92% |
| **AUC** | 0.986 |
| **Dataset** | 3,917 women from NHANES |
| **Features** | 15 health indicators |

---

## 🧰 Technologies

- **Language**: Python
- **ML Framework**: Scikit-learn (Random Forest)
- **Explainability**: SHAP
- **AI Integration**: OpenAI GPT-3.5
- **App**: Streamlit
- **Data**: NHANES (CDC)

---

## 🚀 Run it yourself

```bash
# Clone the repository
git clone https://github.com/your-username/menopause-predictor.git
cd menopause-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
Or
python -m streamlit run app/streamlit_app.py
