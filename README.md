<div align="center">

# Work Tagger – TFM Extended Version  
### Development of an Interactive Tool for the Visualization and Improvement of Time Use at Work

<img src="https://github.com/user-attachments/assets/f6f9b4f5-dd50-4baa-8a80-dd6165702efe" width="420" alt="WorkTagger logo"/>

[![Open Work Tagger](https://img.shields.io/badge/Open%20App-Streamlit-blue)](https://worktagger-tfm.streamlit.app/)

</div>

---

## 🧩 About this Project

This project extends the original **Work Tagger** tool (Resinas et al., 2024) as part of the **Master’s Thesis**  
*"Development of an Interactive Tool for the Visualization and Improvement of Time Use at Work"*,  
at the **University of Seville (2025)**.

The tool allows users to import **Active Window Tracking (AWT)** logs (e.g., from Tockler), classify them manually or automatically, visualize their time allocation, and receive **personalized productivity insights** through AI recommendations.

---

## ✨ Main Improvements in this TFM Version

- 🧠 **Heuristic & GPT-based classification:** hybrid logic for accurate activity labeling.  
- 📊 **Interactive dashboards:** dynamic visualizations (donut, bars, timelines, heatmaps).  
- 💬 **AI productivity assistant:** contextual chatbot powered by OpenAI/OpenRouter APIs.  
- 🗂️ **Exportable PDF reports:** automatic generation of executive summaries and Eisenhower analysis.  
- ⚙️ **Streamlined UX:** redesigned interface with tab navigation and real-time validation.  
- 🔒 **Privacy-first design:** data processed locally, no external storage.  

---

## ⚙️ Installation and Execution

Follow these steps to run Work Tagger locally:

### 1️⃣ Create a virtual environment
```bash
python3 -m venv venv
```

### 2️⃣ Activate the environment
```bash
source venv/bin/activate
```
(Use `venv\Scripts\activate` on Windows)

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application
```bash
streamlit run app_core_act.py
```

---

## 📦 Deployment Options
- 🐳 **Docker image** available for containerized execution.  
- ☁️ **Streamlit Cloud** deployment: [https://worktagger-tfm.streamlit.app](https://worktagger-tfm.streamlit.app)

---

## 🧠 Functional Highlights

- **Automatic and manual classification** of activities and cases.  
- **Eisenhower Matrix integration** for productivity quadrant analysis.  
- **Real-time insights** and behavioral pattern detection.  
- **Conversation history export** from the AI assistant.  
- **Lightweight design**, runs locally with minimal setup.

---

## 📚 References
- Resinas, M., Goñi, R., Beerepoot, I., del Río Ortega, A., & Reijers, H. A. (2024). *Work Tagger: A Labeling Companion*. University of Seville.  
- Rebolledo, E. (2025). *Development of an Interactive Tool for the Visualization and Improvement of Time Use at Work*. University of Seville.

---

## 👨‍💻 Author
Developed by **Esdrick Rebolledo**  
Master’s Degree in Software Engineering: Cloud, Data and IT Management  
*University of Seville – 2025*
