from sklearn import datasets

# Load dataset
digits = datasets.load_digits()
A = digits.data
b = digits.target
num_classes = len(digits.target_names)
num_features = len(digits.feature_names)
