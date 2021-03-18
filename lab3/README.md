# Lab 3 - Classification

[Pima Indian Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database?select=diabetes.csv) from the UCI Machine Learning Repository

## My Output

Experiement | Accuracy | Confusion Matrix | Comment  
Baseline | 0.6770833333333334 | [[114  16] [46 16]] | features: pregnant, insulin, bmi, age  
Solution1 | 0.7708333333333334 | [[115  15] [29 33]] | features: glucose, insulin, bmi, age  
Solution2 | 0.7864583333333334 | [[117  13] [28 34]] | features: pregnant, glucose, insulin, bmi, age  
Solution3 | 0.796875 | [[118  12] [27 35]] | features: pregnant, glucose, bp, insulin, bmi, age  

## Question

* Can we predict the diabetes status of a patient given their health measurements? Build a classifer and calculate Confusion matrix with

- True Positives (TP): we correctly predicted that they do have diabetes
- True Negatives (TN): we correctly predicted that they don't have diabetes
- False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error")
- False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error")






