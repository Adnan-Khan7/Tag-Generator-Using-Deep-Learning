#Importing the libraries required
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

#Note below have to enter the directory of your data (images)presen in your computer's/cloud 

training_data_generator = ImageDataGenerator(rescale=1/255, validation_split=0.1)
training_data = training_data_generator.flow_from_directory(
    "/content/drive/My Drive/Colabbar Tags Data/Images",
    batch_size=32,
    target_size=(300, 300),
    subset="training",
    class_mode='categorical')
validation_data = training_data_generator.flow_from_directory(
    "/content/drive/My Drive/Colabbar Tags Data/Images",
    batch_size=32,
    target_size=(300, 300),
    subset="validation",
    class_mode='categorical')

#Model definition
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(300, 300, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19, activation='sigmoid'))

# Now we are going to Compile our model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit_generator(training_data, validation_data=validation_data, epochs=50)

model.save('/content/drive/My Drive/Colabbar Tags Data/tags_model.h5')
