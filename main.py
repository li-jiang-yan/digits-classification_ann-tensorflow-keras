from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load dataset
digits = datasets.load_digits()
A = digits.data
b = digits.target
num_classes = len(digits.target_names)
num_features = len(digits.feature_names)

# Split training/testing data
A_train, A_test, b_train, b_test = train_test_split(A, b, random_state=1)
