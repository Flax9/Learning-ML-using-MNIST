import numpy as np
import gzip
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_local_mnist():
    # Helper function to read the idx files
    def read_idx(filename):
        with gzip.open(filename, 'rb') as f:
            # First 4 bytes are a magic number (not needed here)
            # Next 4 bytes are the number of items
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            # First 16 bytes contain magic number, sizes, rows, and cols
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # Reshape to 28x28 images
        return data.reshape(-1, 28, 28)

    # Loading the files you downloaded
    x_train = read_images('train-images-idx3-ubyte.gz')
    y_train = read_idx('train-labels-idx1-ubyte.gz')
    x_test = read_images('t10k-images-idx3-ubyte.gz')
    y_test = read_idx('t10k-labels-idx1-ubyte.gz')

    return (x_train, y_train), (x_test, y_test)

# 1. Load data from your downloaded files
(x_train, y_train), (x_test, y_test) = load_local_mnist()

# 2. Preprocess (Normalization)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Build & Train the Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training using local files...")
model.fit(x_train, y_train, epochs=5)

# 4. Final Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# --- Visualization Block ---
def plot_prediction(index):
    # Predict the class for a specific test image
    prediction = model.predict(x_test[index:index+1])
    predicted_label = np.argmax(prediction)
    
    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"True: {y_test[index]} | Predicted: {predicted_label}")
    plt.show()

# Show the first 5 test images and their predictions
for i in range(5):
    plot_prediction(i)