# Creditworthiness Prediction: A Study on Algorithmic Fairness and Bias

## 📌 Project Overview
This project implements a machine learning pipeline to categorize individuals as 'High' or 'Low' credit based on financial indicators. Beyond predictive modeling, this project serves as a technical analysis of the **Bias-Accuracy Trade-off**, investigating whether the inclusion of demographic data improves model performance or introduces systemic discrimination.

## 🛠️ Technical Stack
- **Language:** Python 3
- **Environment:** Google Colab
- **Core Libraries:** 
  - `Pandas` & `NumPy` (Data Engineering)
  - `Scikit-Learn` (ML Pipeline: RandomForestClassifier, Imputation, Encoding)
  - `Matplotlib` & `Seaborn` (Quantitative Visualization)

## 🚀 The Data Science Pipeline

### 1. Advanced Data Engineering
- **Feature Engineering:** Developed a `Disposable Income` feature (Income - Expenses) to capture a more accurate signal of creditworthiness than raw income alone.
- **Signal Boosting:** Implemented a refinement function to align synthetic target variables with financial behavior, ensuring the model had a logical pattern to learn.
- **Outlier Clipping:** Applied 1st and 99th percentile clipping to remove synthetic noise and prevent model skew.

### 2. Solving the "Accuracy Paradox"
To ensure a honest and unbiased model, I addressed two major data challenges:
- **Class Imbalance:** Solved a significant imbalance where the model previously guessed the majority class. I implemented `class_weight='balanced'` to force the model to prioritize the minority 'High Credit' group.
- **The Median Split:** To eliminate the "guessing" effect, I implemented an index-based median split, guaranteeing a perfect 50/50 distribution of credit grades.

### 3. The Bias-Accuracy Experiment
I conducted a controlled experiment to test the impact of "Biased" data. I compared a **Fair Baseline Model** (financials only) against models that included a single demographic factor:
- **Factors Tested:** Gender, Age, Education Level, and Region.
- **Validation:** Used **Permutation Importance** to quantify the influence of each variable on the final prediction.

## 📈 Key Findings & Results
- **Predictive Power:** The final balanced model achieved an accuracy of `[62%]`, significantly outperforming the 50% random-chance baseline.
- **The Bias Result:** The experiment revealed that adding demographic data provided **no significant increase in accuracy**. This proves that financial behavior is the primary driver of creditworthiness and that demographic stereotypes do not provide a predictive advantage.
- **Model Stability:** The use of a Random Forest Classifier with constrained `max_depth` prevented overfitting and ensured the model generalized well to unseen data.

## ⚖️ Ethical & Regulatory Alignment
This project was developed with a **Fairness-by-Design** philosophy, aligning with the **UK Equality Act 2010** and **FCA (Financial Conduct Authority)** guidelines. 

By proving that biased data does not increase accuracy, this project demonstrates that fairness and performance are not mutually exclusive. It highlights the professional necessity of excluding protected characteristics from credit-scoring algorithms to prevent systemic discrimination.

## 📖 How to Run
1. Clone this repository.
2. Upload the `.ipynb` file to [Google Colab](https://colab.research.google.com/).
3. Upload the `synthetic_personal_finance_dataset.csv` to the Colab session.
4. Run all cells sequentially.
