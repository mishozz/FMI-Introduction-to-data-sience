from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_training_and_test_data(iris):
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_logistic_regression_model(X_train, y_train):
    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, y_train)
    return log_reg

def plot_confusion_matrix(confusion_matrix_log_reg, iris):
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix_log_reg, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = load_training_and_test_data(iris)
    log_reg_model = train_logistic_regression_model(X_train, y_train)
    
    y_predictions = log_reg_model.predict(X_test)
    confusion_matrix_log_reg = confusion_matrix(y_test, y_predictions)
    plot_confusion_matrix(confusion_matrix_log_reg, iris)
        