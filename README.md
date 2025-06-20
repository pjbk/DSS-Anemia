# 🩺 Interpretable Real-Time Anemia Risk Predictor

A clinical decision support system (CDSS) designed to diagnose anemia and assess its likelihood based on hematological and demographic parameters. Powered by a pre-trained **Random Forest Classifier** and developed using **Streamlit**, this web-based application offers real-time predictions and interpretable explanations to assist healthcare professionals at the point of care.

---

## 🌐 Live Web App

**Launch Now**: [https://dss-anemia.streamlit.app/](https://dss-anemia.streamlit.app/)

![App Interface](https://github.com/pjbk/dss-anemia/blob/main/App%20interface.png)

---

## 🧠 Key Features

- ✅ **Real-Time Diagnosis**: Predicts anemia status based on clinical input.
- 🧬 **Model Interpretability**: Explains predictions with SHAP visualizations; highlights the top 5 contributing features.
- 📱 **Responsive UI**: Works seamlessly on both desktop and mobile devices.
- 🌙 **Dark Mode Support**: Automatically adapts to the user's system theme.
- 📊 **Risk Probability**: Displays confidence score for each prediction.

---

## 📊 Dataset Sources

The model is trained using datasets from both local clinical sources and public repositories:

- **📁 Mendeley Dataset** (Aalok Healthcare Ltd., Dhaka, Bangladesh):  
  [https://data.mendeley.com/datasets/y7v7ff3wpj/1](https://data.mendeley.com/datasets/y7v7ff3wpj/1)

- **📁 Kaggle Dataset** (Open Source):  
  [https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset)

### 📚 References
1. Mojumdar, M.U., et al. *Pediatric Anemia Dataset: Hematological Indicators and Diagnostic Classification*. Mendeley Data, V1 (2024). [DOI](https://doi.org/10.17632/y7v7ff3wpj.1)  
2. Mojumdar, M.U., et al. *AnaDetect: An extensive dataset for advancing anemia detection, diagnostic methods, and predictive analytics in healthcare.* Data in Brief 58, 111195 (2025). [DOI](https://doi.org/10.1016/j.dib.2024.111195)

---

## 🧪 Model Pipeline (Pseudocode)

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

## 🚀 Quick Start
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pjbk/dss-anemia.git
   cd dss-anemia
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

## 📁 Project Structure

```
dss-anemia/
├── README.md
├── app.py
├── rf_model.pkl
├── scaler.pkl
└── requirements.txt
```

## 📘 User Guide
**Input Data:** Provide patient's hematological and demographic data.
**Prediction Output:** Receive anemia risk prediction and associated probability.
**Visual Explanation:** View SHAP-based plot explaining the key contributing factors.

## 🛠 Tools and Technologies
| Tool           | Purpose                                         |
| -------------- | ----------------------------------------------- |
| `Scikit-learn` | Machine learning models and evaluation          |
| `Streamlit`    | Interactive real-time web application interface |
| `SHAP`         | Interpretable model explanations (XAI)          |
| `Matplotlib`   | Data and SHAP visualization                     |


