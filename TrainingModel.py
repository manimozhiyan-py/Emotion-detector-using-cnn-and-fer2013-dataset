# Import necessary packages
import cv2
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.layers import Rescaling
import tensorflow as tf

# Initialize image data generator with rescaling
rescale_layer = Rescaling(1./255)

# Load the datasets with the rescaling layer
train_data_gen = image_dataset_from_directory(
    'data/train',
    image_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    label_mode='categorical',
)

validation_data_gen = image_dataset_from_directory(
    'data/test',
    image_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    label_mode='categorical',
)

# Apply the rescaling to both train and validation datasets
train_data_gen = train_data_gen.map(lambda x, y: (rescale_layer(x), y))
validation_data_gen = validation_data_gen.map(lambda x, y: (rescale_layer(x), y))

# Create model structure
emotion_model = Sequential()

# Note: Using Input(shape=...) in the first layer instead of input_shape
emotion_model.add(tf.keras.layers.Input(shape=(48, 48, 1)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

# Compile the model using the correct parameter name for learning rate
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the neural network/model
emotion_model_info = emotion_model.fit(
        train_data_gen,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_data_gen,
        validation_steps=7178 // 64)

# Save model structure in JSON file
#model_json = emotion_model.to_json()
#with open("emotion_model.json", "w") as json_file:
#    json_file.write(model_json)

# Save trained model weights in .h5 file
emotion_model.save_weights('emotion_model.weights.h5')
