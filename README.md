# Anemia Risk Predictor DSS

An ensemble learning-powered web application designed to diagnose anemia and estimate its likelihood based on clinical hematological and demographic parameters. A real-time interface, developed with Streamlit, predicts anemia and enables data-driven clinical decisions at the point of care. This tool empowers medical practitioners and health professionals with a decision support system (DSS) that not only predicts anemia status but also explains the reasoning behind predictions using SHAP (SHapley Additive exPlanations) visualizations.

## 🌐 Live Web App Preview
**Live Predictive System:** [https://anemia-dss.streamlit.app/](https://dss-anemia.streamlit.app/)  

![App Header](https://github.com/pjbk/dss-anemia/blob/main/App%20interface.png)

## Key Features

- **Accurate Disease Diagnosis**: Diagnoses Anemia with high precision (ensemble model's prediction accuracy: 99.67%).
- **Model Explainability**: Utilizes SHAP-XAI to enhance understanding of AI predictions. Highlights the **top 5 influential features** for clinical insights..
- **Responsive UI Design**: Ensures smooth user experience on both desktop and mobile devices.
- **Dark Mode Support**: Automatically adapts to the user's preferred theme.
- **Confidence Metrics**: Displays prediction probabilities to reflect the model’s certainty.

## Dataset

The model is trained using clinical records, collected from Aalok Healthcare Ltd., Dhaka, Bangladesh. The dataset can be accessed from the following Mendeley Data link: [https://data.mendeley.com/datasets/y7v7ff3wpj/1](https://data.mendeley.com/datasets/y7v7ff3wpj/1)  <br>
Add the Kaggle open-source dataset is available at:[https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset)   

**Ref 1.** Mojumdar, M.U., et al.: Pediatric Anemia Dataset: Hematological Indicators and Diagnostic Classification. Mendeley Data, V1(2024). https://doi.org/10.17632/y7v7ff3wpj.1  
**Ref 2.** Mojumdar, M.U., et al.: AnaDetect: An extensive dataset for advancing anemia detection, di-agnostic methods, and predictive analytics in healthcare. Data in Brief 58, 111195 (2025). https://doi.org/10.1016/j.dib.2024.111195  

## Model Architecture
```python
# model architecture  
base_learners = [
    ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features='log2', min_samples_split=3, random_state=42)),
    ('lr', make_pipeline(StandardScaler(), LogisticRegression(C= 10, solver= ‘liblinear’, random_state=42))),
    ('svm', make_pipeline(StandardScaler(), SVC(C=10, kernel= 'linear', probability=True, random_state=42))),
    ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(algorithm= 'auto', n_neighbors= 4)))
]

meta_learner = make_pipeline(StandardScaler(), SVC(C=10, kernel= 'linear', probability=True, random_state=42))

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    passthrough=True,
    n_jobs=-1
)
```

## Quick Start

To set up and run the application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pjbk/dss-anemia.git
   cd anemia-DSS
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at `http://localhost:8501`.

## Project Pipeline

```
anemia-DSS/
├── README.md
├── app.py
├── rf_model.pkl
├── scaler.pkl
└── requirements.txt
```

### User Guides

1. **Upload Model**: The pretrained model (`ensemble_model.pkl`) is included in the repository.
2. **Input Patient's Data**: Choose Hematological and Demographic data of patient for diagnosis.
3. **Predict Anemia Risk**: The app will display the diagnosis along with the model's confidence score.
4. **Explainability**: SHAP-enhaced visualization will appear that explains the reasoning.


## Tools and Technologies

- **Scikit-Learn**: For model development and and performance analysis.
- **Streamlit**: Framework for building interactive web applications.
- **SHAP-XAI**: Technique for visualizing model attention and explaining predictions.
- **Matplotlib**: Library used for generating visualizations.

