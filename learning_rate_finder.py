from sklearn import datasets
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
digits = datasets.load_digits()
A = digits.data
b = digits.target
num_classes = len(digits.target_names)
num_features = len(digits.feature_names)

# Plot accuracy vs learning rate for 30 epochs
learning_rates = np.linspace(1e-8, 0.03, 50)
accuracies = []

for lr in learning_rates:
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
    optimizer = Adam(learning_rate=lr)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Train model
    model.fit(A, to_categorical(b, num_classes=num_classes), verbose=0, epochs=30)

    # Find accuracy with training data
    b_pred = np.argmax(model.predict(A, verbose=0), axis=1)

    # Store results
    accuracies.append(accuracy_score(b, b_pred))

plt.plot(learning_rates, accuracies, "-x")
plt.xlabel("Learning rate")
plt.ylabel("Accuracy")
plt.title("Model accuracy after 30 epochs vs learning rate")
plt.show()
