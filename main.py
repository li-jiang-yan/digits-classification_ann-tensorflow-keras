from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load dataset
digits = datasets.load_digits()
A = digits.data
b = digits.target
num_classes = len(digits.target_names)
num_features = len(digits.feature_names)

# Split training/testing data
A_train, A_test, b_train, b_test = train_test_split(A, b, random_state=1)

# Compile model
num_neurons = num_features
x = Input(shape=(num_features,))
h = Dense(num_neurons, activation="sigmoid")(x)
num_neurons //= 2
while (num_neurons > num_classes):
    h = Dense(num_neurons, activation="sigmoid")(h)
    num_neurons //= 2
y = Dense(num_classes, activation="sigmoid")(h)
model = Model(x, y)
optimizer = Adam(learning_rate=0.005)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train model
model.fit(A_train, to_categorical(b_train, num_classes=num_classes), epochs=100)

# Test model
b_pred = np.argmax(model.predict(A_test, verbose=0), axis=1)

# Compute the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(b_test, b_pred), display_labels=digits.target_names)
disp.plot()
plt.show()

# Get the model classification metrics (will only show after the confusion matrix display window is closed)
print(classification_report(b_test, b_pred))

# Save the model for use later
model.save("model.keras")
