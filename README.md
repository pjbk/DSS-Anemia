# Interpretable Real-Time Anemia Risk Predictor

A web application powered by a pre-trained Random Forest Classifier is designed to diagnose anemia and estimate its likelihood using clinical hematological and demographic parameters. Developed with Streamlit, this real-time interface supports data-driven clinical decision-making at the point of care. The tool provides medical practitioners and healthcare professionals with a decision support system (DSS) that not only predicts anemia status but also explains the underlying reasoning using SHAP (SHapley Additive exPlanations) visualizations. Additionally, it offers probability-based risk scores to further inform clinical assessments.

## 🌐 Web App Preview
**Live Predictive System:** [https://anemia-dss.streamlit.app/](https://dss-anemia.streamlit.app/)  

![App Header](https://github.com/pjbk/dss-anemia/blob/main/App%20interface.png)

## Key Features

- **Accurate Disease Diagnosis**: Diagnoses Anemia with high precision.
- **Model Explainability**: Utilizes SHAP-XAI to enhance understanding of AI predictions. Highlights the **top 5 influential features** for clinical insights..
- **Responsive UI Design**: Ensures smooth user experience on both desktop and mobile devices.
- **Dark Mode Support**: Automatically adapts to the user's preferred theme.
- **Confidence Metrics**: Displays prediction probabilities (**Risk Score**) to reflect the model’s certainty.

## Dataset

The model is trained using clinical records, collected from Aalok Healthcare Ltd., Dhaka, Bangladesh. The dataset can be accessed from the following Mendeley Data link: [https://data.mendeley.com/datasets/y7v7ff3wpj/1](https://data.mendeley.com/datasets/y7v7ff3wpj/1)  <br>
Add the Kaggle open-source dataset is available at:[https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset)   

**Ref 1.** Mojumdar, M.U., et al.: Pediatric Anemia Dataset: Hematological Indicators and Diagnostic Classification. Mendeley Data, V1(2024). https://doi.org/10.17632/y7v7ff3wpj.1  
**Ref 2.** Mojumdar, M.U., et al.: AnaDetect: An extensive dataset for advancing anemia detection, di-agnostic methods, and predictive analytics in healthcare. Data in Brief 58, 111195 (2025). https://doi.org/10.1016/j.dib.2024.111195  

## Model Pipeline Pseudocode
```python
# model pipeline  
Input: 
    Dataset D with Observations O, Features X, Target y
Output: 
    Processed Dataset D_processed, Model Evaluation M_ev, 
    Confidence Intervals CI, Permutation Scores P_scores

1.  D ← Load_Anemia_Dataset(O, X, y)

2.  # Handle Missing Values
3.  for each feature Xi in X do
4.      D ← Impute_Missing_Values(D, Xi, method="mean", add_flag=True)
5.      if Missing_Percentage(Xi) ≥ 50% then
6.          D ← Drop_Feature(D, Xi)
7.      end if
8.  end for

9.  # Encode Categorical Features
10. for each categorical feature Xi in X do
11.     D ← One_Hot_Encode(D, Xi)
12. end for

13. # Remove Outliers using Z-Score
14. for each datapoint Xij in D do
15.     Z ← Compute_Z_Score(Xij)
16.     if |Z| > 3 then
17.         D ← Remove_Outlier(D, Xij)
18.     end if
19. end for

20. # Analyze Target Distribution
21. for each class yi in y do
22.     p(yi) ← Count(yi) / Total_Count(y)
23. end for

24. # Statistical Feature Analysis
25. for each feature Xi in X do
26.     Compute: mean, median, std, skewness, kurtosis, p-value
27.     for each feature Xj ≠ Xi do
28.         ρ_ij ← Pearson_Correlation(Xi, Xj)
29.     end for
30.     Importance[Xi] ← Feature_Importance(RandomForest, Xi)
31. end for

32. D_processed ← D

33. # Split Dataset
34. D_train_val, D_test_strat ← Stratified_Split(D_processed, test_size=20%)

35. # Define Models
36. Models ← {DecisionTree, GradientBoosting, RandomForest, ExtraTrees}

37. # Model Optimization on Training Data
38. for each model M in Models do
39.     M_optimized ← Grid_Search_Tuning(M, D_train_val)
40.     Optimized_Models.add(M_optimized)
41. end for

42. # Cross-Validation Performance on Train/Val Set
43. for each M_opt in Optimized_Models do
44.     CV_Scores ← Cross_Validate(M_opt, D_train_val, folds=5)
45.     M_ev.add((M_opt, CV_Scores))
46. end for

47. # Evaluate Final Models on Unseen Stratified Test Set
48. for each M_opt in Optimized_Models do
49.     Test_Score ← Evaluate(M_opt, D_test_strat)
50.     CI ← Compute_Confidence_Interval(Test_Score, method="bootstrap", confidence_level=95%)
51.     P_scores ← Permutation_Validation(M_opt, D_test_strat, n_permutations=1000)
52.     Final_Eval.add((M_opt, Test_Score, CI, P_scores))
53. end for

54. # SHAP Explainability on Best Model
55. Best_Model ← Select_Best_Model(Final_Eval)
56. SHAP_Explain(Best_Model, D_processed)

57. return D_processed, Final_Eval, Confidence_Intervals CI, Permutation Scores P_scores

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

