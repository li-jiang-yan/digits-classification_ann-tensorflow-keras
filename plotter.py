"""
Returns a matplotlib illustration of digit images with their corresponding predicted and
true labels
"""

from sklearn import datasets
import keras
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
digits = datasets.load_digits()
images = digits.images
x = digits.data
y = digits.target

print(x.shape)

# Load model
model = keras.models.load_model("model.keras")

# Plot graphs with their labels
ROWS = 3
COLS = 3
PLOTS = ROWS * COLS
fig, axs = plt.subplots(ROWS, COLS)
for index in range(PLOTS):
    row = index % COLS
    col = index // COLS
    ax = axs[row, col]
    image = images[index]
    x_input = x[[index], :]
    p = model.predict(x_input, verbose=0)
    y_true = y[index]
    y_pred = np.argmax(p)
    label = f"Prediction: {y_pred}\nTrue: {y_true}"
    ax.matshow(image, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(label)
plt.show()
