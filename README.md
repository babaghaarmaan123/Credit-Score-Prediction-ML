# Credit-Score-Prediction-ML
A machine learning project analyzing the impact of financial indicators and demographic bias on creditworthiness prediction.


# Creditworthiness Prediction using Machine Learning

## 📌 Project Overview
This project explores whether financial indicators—such as income, savings, and loan history—can be used to predict a person's creditworthiness. The goal was to build a model that is not only accurate but also **fair and ethical**, avoiding the common pitfalls of algorithmic bias.

## 🛠️ Tech Stack
- **Language:** Python 3
- **Environment:** Google Colab
- **Libraries:** 
  - `Pandas` (Data Manipulation)
  - `Scikit-Learn` (Machine Learning Pipeline)
  - `Matplotlib` & `Seaborn` (Data Visualization)

## 🚀 The Pipeline
The project follows a professional data science workflow:

1. **Exploratory Data Analysis (EDA):** Analyzed the synthetic dataset to identify missing values and distribution patterns.
2. **Feature Engineering:** Created a `Disposable Income` feature (Income - Expenses) to provide a stronger signal of creditworthiness than raw income alone.
3. **Fairness-Conscious Design:** Deliberately excluded demographic variables (Age, Gender, Region) to prevent systemic bias and ensure compliance with ethical standards (e.g., the UK Equality Act 2010).
4. **Binary Classification:** Pivoted from predicting a numeric score (Regression) to predicting a 'High' or 'Low' credit grade (Classification) to increase model stability and utility.
5. **Balanced Learning:** Implemented `class_weight='balanced'` to solve the **Accuracy Paradox**, forcing the model to correctly identify the minority 'High Credit' group rather than just guessing the majority class.

## 📈 Key Findings
- **Model Performance:** The final model achieved an accuracy of `[Insert your Accuracy %]`.
- **The Bias-Accuracy Trade-off:** I conducted a controlled experiment adding biased factors (Gender, Age, etc.) to the model. While some biased factors slightly increased raw accuracy, this proved that the model was relying on stereotypes rather than financial behavior.
- **Top Predictors:** Through **Permutation Importance**, it was found that `[Insert your top feature, e.g., Disposable Income]` was the most influential driver of the credit grade.

## ⚖️ Ethical Conclusion
This project demonstrates that in the financial sector, **accuracy is not the only metric for success.** A model that is 5% more accurate but relies on biased demographic data is ethically and legally unacceptable. This project highlights the necessity of 'Fairness-by-Design' in FinTech.
