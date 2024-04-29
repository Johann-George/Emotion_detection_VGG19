# Importing necessary libraries
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler

train_dir='data/train'
test_dir='data/test'

def lr_schedule(epoch):
    initial_lr = 0.0001
    decay_factor = 0.1
    decay_epochs = 10
    return initial_lr * (decay_factor ** (epoch // decay_epochs))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1.0/255.0,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True)
val_datagen=ImageDataGenerator(rescale=1.0/255.0)

trainDatagen=train_datagen.flow_from_directory(train_dir,
                                               target_size=(48,48),
                                               batch_size=100,
                                               class_mode='categorical',
                                               color_mode='rgb')

valDatagen=val_datagen.flow_from_directory(test_dir,
                                           target_size=(48,48),
                                           batch_size=100,
                                           class_mode='categorical',
                                           color_mode='rgb')


# Load the pre-trained VGG19 model without the top (fully connected) layers
base_model = VGG19(weights='imagenet', include_top=False)

# Adding custom layers for emotion detection
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Adding a fully connected layer
predictions = Dense(7, activation='softmax')(x)  # Output layer with 7 classes for emotions

# Creating the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers in the base VGG19 model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)

history=model.fit(trainDatagen,validation_data=valDatagen,epochs=5,callbacks=[lr_scheduler])

model.save_weights('emotion_model_vgg19.weights.h5')

