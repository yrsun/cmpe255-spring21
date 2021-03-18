import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        self.X_test = None
        self.y_test = None
        

    def define_feature(self):
        feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y

    def define_feature1(self):
        feature_cols = ['glucose', 'insulin', 'bmi', 'age']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y

    def define_feature2(self):
        feature_cols = ['pregnant', 'glucose', 'insulin', 'bmi', 'age']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y

    def define_feature3(self):
        feature_cols = ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'age']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def train(self, num):
        # split X and y into training and testing sets
        if num == 0:
            X, y = self.define_feature()
        elif num == 1:
            X, y = self.define_feature1()
        elif num == 2:
            X, y = self.define_feature2()
        elif num == 3:
            X, y = self.define_feature3()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self, num):
        model = self.train(num)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    print("Experiement | Accuracy | Confusion Matrix | Comment")
    rst0 = "Baseline | "
    classifer = DiabetesClassifier()
    result = classifer.predict(0)
    score = classifer.calculate_accuracy(result)
    rst0 += f"{score} | "
    con_matrix = classifer.confusion_matrix(result)
    rst0 += f"[{con_matrix[0]} {con_matrix[1]}] | features: pregnant, insulin, bmi, age"
    print(rst0)

    rst1 = "Solution1 | "
    classifer = DiabetesClassifier()
    result = classifer.predict(1)
    score = classifer.calculate_accuracy(result)
    rst1 += f"{score} | "
    con_matrix = classifer.confusion_matrix(result)
    rst1 += f"[{con_matrix[0]} {con_matrix[1]}] | features: glucose, insulin, bmi, age"
    print(rst1)

    rst2 = "Solution2 | "
    classifer = DiabetesClassifier()
    result = classifer.predict(2)
    score = classifer.calculate_accuracy(result)
    rst2 += f"{score} | "
    con_matrix = classifer.confusion_matrix(result)
    rst2 += f"[{con_matrix[0]} {con_matrix[1]}] | features: pregnant, glucose, insulin, bmi, age"
    print(rst2)

    rst3 = "Solution3 | "
    classifer = DiabetesClassifier()
    result = classifer.predict(3)
    score = classifer.calculate_accuracy(result)
    rst3 += f"{score} | "
    con_matrix = classifer.confusion_matrix(result)
    rst3 += f"[{con_matrix[0]} {con_matrix[1]}] | features: pregnant, glucose, bp, insulin, bmi, age"
    print(rst3)
