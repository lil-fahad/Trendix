# OmniMarket Prophet

OmniMarket Prophet هو نظام تنبؤ وتحليل ذكي للأسواق المالية يستخدم تقنيات التعلم الآلي العميق والكمي لتوقع أسعار الأصول (مثل الأسهم والعقود الآجلة) بدقة عالية مع واجهة تفاعلية عبر Streamlit.

---

## 🚀 الميزات الرئيسية

- **رفع ملفات CSV يدويًا** (Open, High, Low, Close, Volume)
- **تدريب نماذج XGBoost و LSTM تلقائيًا** عبر الواجهة
- **تقسيم البيانات 80% تدريب / 20% اختبار**
- **حساب MAE ودقة الاتجاه لكل نموذج**
- **تفسير التنبؤات باستخدام GPT**
- **مخططات تحليل تفاعلية عبر Plotly**
- **واجهة Streamlit جاهزة للاستخدام**

---

## 🗂️ هيكل المشروع

```
omni_project/
├── core/                      ← وحدات التنبؤ والتفسير والتوصية
├── data/historical/           ← مكان ملفات CSV المرفوعة
├── models/                    ← النماذج المدربة (XGBoost / LSTM)
├── training/                  ← trainer.py لتدريب النماذج
├── streamlit_ui/              ← واجهة Streamlit
├── OmniMarket_Colab_Trainer.ipynb  ← دفتر تدريب من Google Colab
└── README.md
```

---

## ⚙️ طريقة التشغيل

1. تثبيت الحزم:
```bash
pip install -r requirements.txt
```

2. تشغيل الواجهة:
```bash
streamlit run streamlit_ui/app.py
```

---

## 📈 التدريب اليدوي من الطرفية

```bash
python training/trainer.py --csv data/historical/AAPL.csv --model all
```

---

## 🧠 تكنولوجيا مستخدمة

- XGBoost
- LSTM (Keras / TensorFlow)
- Scikit-learn
- Streamlit
- Plotly
- GPT API (OpenAI)

---

## 🧪 قيد التطوير

- مقارنة رسومية بين النماذج
- نظام تنبيه ذكي للتغيرات الكبيرة
- دعم بيانات مباشرة (live API)

---

## 👨‍💻 فريق التطوير

تم التطوير بواسطة Autodev Code Hacker 2050 بإشراف فهد (lil-fahad)


---
### 🚀 شغّل المشروع مباشرة عبر Streamlit Cloud:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
