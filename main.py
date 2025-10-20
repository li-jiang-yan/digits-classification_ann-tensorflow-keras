from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

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
optimizer = Adam()
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train model
model.fit(A_train, to_categorical(b_train, num_classes=num_classes), epochs=100)
