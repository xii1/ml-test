import numpy as np
from PIL import Image
from tensorflow import keras

model = keras.models.load_model('pre-trained_models/classification/dog_cat.h5')


def predict_dog_cat(img):
    resized_img = Image.open(img).resize((64, 64))
    data = np.array(resized_img)
    data = data / 255
    data = np.expand_dims(data, axis=0)
    predict = model.predict(data)
    if predict[:, :] > 0.5:
        return 'Dog (%1.2f)' % (predict[0, 0])
    else:
        return 'Cat (%1.2f)' % (1.0 - predict[0, 0])
