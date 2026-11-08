# Bank Marketing: Term Deposit Subscription Classifier

**Live app:** https://bank-marketing-ml-classifier-5yr376rp29hduz4mqimmir.streamlit.app/

## a. Problem statement
Bank runs phone marketing campaign to sell term deposits. Calling every customer is costly, so the bank wants to know in advance which customers will subscribe. This project builds and compares 5 classification models that predict whether a customer will subscribe to a term deposit from customer details.

## b. Dataset description
The dataset is from the UCI Machine Learning Repository. It comes from direct marketing campaigns. I used the file bank.csv, which has 4,521 rows (a 10% sample of the full 45,211-row data), 16 input features and one target column y. There are 7 numeric features (age, balance, day, duration, campaign, pdays, previous) and 9 categorical features (job, marital, education, default, housing, loan, contact, month, poutcome). The target y says whether the client subscribed to a term deposit: 4,000 clients said no and 521 said yes, so only about 11.5% are positive and the data is imbalanced. There are no missing values ("unknown" is used as a category in some columns). I split the data 80/20 with stratification (3,616 training rows and 905 test rows), scaled the numeric columns and one-hot encoded the categorical columns. The duration column is kept.

## c. GitHub Repository Link
https://github.com/anandhaganesh03/bank-marketing-ml-classifier

## d. Models used

### Comparison table
Precision, Recall and F1 are for the "yes" class. Test set: 905 rows.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8972 | 0.8759 | 0.6222 | 0.2692 | 0.3758 | 0.3639 |
| Decision Tree | 0.8983 | 0.8232 | 0.6034 | 0.3365 | 0.4321 | 0.4008 |
| KNN | 0.8939 | 0.8181 | 0.6538 | 0.1635 | 0.2615 | 0.2906 |
| Naive Bayes | 0.8442 | 0.7734 | 0.3520 | 0.4231 | 0.3843 | 0.2976 |
| Support Vector Machine | 0.9039 | 0.8759 | 0.7742 | 0.2308 | 0.3556 | 0.3893 |
