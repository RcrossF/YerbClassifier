from keras.models import load_model
import cv2
import numpy as np

img_width = img_height = 250

model = load_model('model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



img = cv2.imread('test.jpg')
img = cv2.resize(img,(img_width,img_height))
img = np.reshape(img,[1,img_width,img_height,3])

classes = model.predict_classes(img)

print(classes)