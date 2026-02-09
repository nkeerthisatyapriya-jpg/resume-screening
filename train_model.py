
# Note
This is a prototype project built for learning and academic purposes.

# Resume Screening using TensorFlow

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 1. Load dataset
data = pd.read_csv("resume_data.csv")

texts = data["resume_text"].values
labels = data["label"].values

# 2. Convert text to numbers
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=20)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# 4. Build the model
model = Sequential([
    Dense(16, activation="relu", input_shape=(20,)),
    Dropout(0.3),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

# 5. Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 6. Train the model
print("Training started...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 7. Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("Model Accuracy:", accuracy)
