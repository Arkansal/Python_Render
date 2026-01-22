from keras.applications import ResNet50
from keras import Sequential

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential([
    conv_base
])

model.summary()

import pickle

filename = 'imgmodel.pickle'
pickle.dump(model, open(filename, 'wb'))