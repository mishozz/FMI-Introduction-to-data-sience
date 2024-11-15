from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

UNIFORM = 'uniform'
DISTANCE = 'distance'
K_VALUES = [1, 33, 66]

def load_training_and_test_data(iris):
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_logistic_regression_model(X_train, y_train):
    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, y_train)
    return log_reg


def train_knn_model(X_train, y_train, k, weight_type):
    knn = KNeighborsClassifier(n_neighbors=k, weights=weight_type)
    knn.fit(X_train, y_train)
    return knn


def plot_confusion_matrix(confusion_matrix, name_of_the_plot, iris):
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(name_of_the_plot)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    
def plot_regression_classification(X_train, y_train, X_test, y_test,iris):
    log_reg_model = train_logistic_regression_model(X_train, y_train)
    y_predictions_log_reg = log_reg_model.predict(X_test)  
    confusion_matrix_log_reg = confusion_matrix(y_test, y_predictions_log_reg)
    
    plot_confusion_matrix(confusion_matrix_log_reg, 'Confusion Matrix - Logistic Regression', iris)

    
def plot_knn_classifications(X_train, y_train, X_test, y_test, k_values, weight_type, iris):
    for k in k_values:
        knn_model = train_knn_model(X_train, y_train, k, weight_type)
        y_predictions_knn = knn_model.predict(X_test)
        confusion_matrix_knn = confusion_matrix(y_test, y_predictions_knn)
        
        plot_confusion_matrix(confusion_matrix_knn, f'Confusion Matrix - KNN (k={k}, weights={weight_type})', iris)
 

if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = load_training_and_test_data(iris)
    plot_regression_classification(X_train, y_train, X_test, y_test, iris)
    
    plot_knn_classifications(X_train, y_train, X_test, y_test,K_VALUES, UNIFORM, iris)
    plot_knn_classifications(X_train, y_train, X_test, y_test,K_VALUES, DISTANCE, iris)
    
    plt.show()
