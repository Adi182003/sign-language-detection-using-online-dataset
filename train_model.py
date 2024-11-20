import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.callbacks import TensorBoard
import tensorflow as tf  # Import TensorFlow here

# Define actions (replace with your actions)
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Path for exported data, numpy arrays
DATA_PATH = 'MP_Data' 

# Number of sequences per action
no_sequences = 30

# Length of each sequence
sequence_length = 30

# Mapping from action labels to integers
label_map = {label: num for num, label in enumerate(actions)}

# Initialize data lists
sequences, labels = [], []

# Iterate over actions and sequences to load data
for action in actions:
    for sequence in range(no_sequences):
        sequence_data = []
        
        # Iterate over frames in the sequence
        for frame_num in range(sequence_length):
            # Load keypoints from .npy file
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            keypoints = np.load(npy_path)
            
            # Append keypoints to sequence_data
            sequence_data.append(keypoints)
        
        # Append sequence_data and corresponding label to sequences and labels lists
        sequences.append(sequence_data)
        labels.append(label_map[action])

# Convert lists to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Define TensorBoard callback for visualization
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))  # Adjust input_shape based on your keypoints dimensions
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with callback to print accuracy and loss after each epoch
class AccuracyLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Loss = {logs['loss']}, Accuracy = {logs['accuracy']}")

model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, AccuracyLossCallback()])

# Print final model summary
model.summary()

# Save model architecture as JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save('model.h5')

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
