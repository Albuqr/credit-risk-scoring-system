
Data Quality Issues -
 - MonthlyIncome: 19,8% missing - strategy: median imputation
 - NumberOfDependents: 2.6% missing - strategy: fill with 0
 - RevolvingUtilization: values > 1.0 exist - strategy: cap at 1
 - One borrower with age = 0 - strategy: drop row


Class Imbalance
 - 93.3% non-default/ 6.7% default
 - Accuracy isn't a valid metric for this
 - Selected metric: AUC-ROC

Strongest Predictors (linear correlation)
 - Positive: total late payments, revolving utilization
 - Negative: monthly income, age
