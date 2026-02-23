from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# -------- Create a very small dummy CNN model -------- #
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# compile model with new API
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

# save model
model.save("emotion_model.h5")

print("Lightweight 48x48 emotion_model.h5 created successfully!")
