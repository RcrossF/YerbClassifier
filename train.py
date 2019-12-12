# Mostly taken from https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d and adapted to recognise yerbs
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.models import Sequential
from keras.callbacks import callbacks
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
import os


config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth = True  # don't pre-allocate memory; allocate as-needed
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#K.tensorflow_backend.set_session(tf.compat.v1.Session(config=config)) # create sess w/ above settings

# Dimentions to rescale images to
img_width = img_height = 256

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])
epochs = 20
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_datagen = ImageDataGenerator(
#        rotation_range=40,
 #       width_shift_range=0.2,
  #      height_shift_range=0.2,
        rescale=1./255
  #      shear_range=0.2,
   #     zoom_range=0.2,
    #    horizontal_flip=True,
     #   fill_mode='nearest')
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary')

filepath="weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)

model.save('model.h5')