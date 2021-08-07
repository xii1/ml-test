import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential, models
from tensorflow.keras import activations, losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

SAVED_MODEL = 'trained_models/classification/fashion_model'
BATCH_SIZE = 32
NUM_CLASSES = 10
NAME = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}


def train_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation=activations.relu))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation=activations.relu))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation=activations.softmax))

    model.compile(optimizer=optimizers.Adam(), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    callback = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=20, callbacks=callback)
    model.save(SAVED_MODEL)

    visualize([{"train": history.history["loss"], "validate": history.history["val_loss"]},
               {"train": history.history["accuracy"], "validate": history.history["val_accuracy"]}],
              ["Model Loss", "Model Accuracy"],
              ["epoch", "epoch"], ["loss", "accuracy"])

    return

def predict_fashion(img):
    resized_img = Image.open(img).convert('RGB').convert('L').resize((28, 28))
    resized_img = ImageOps.invert(resized_img)
    data = np.array(resized_img)
    data = data.reshape(28, 28, 1)
    data = data.astype('float32') / 255
    data = np.expand_dims(data, axis=0)
    model = models.load_model(SAVED_MODEL)
    predict = np.squeeze(model.predict(data))
    index = np.argmax(predict, axis=-1)

    return 'Name: {} (%1.2f)'.format(NAME[index]) % predict[index]


def visualize(data, titles, xlabels, ylabels):
    fig, axes = plt.subplots(1, len(titles), squeeze=False)
    fig.suptitle('Visualization', fontsize=16)

    for i in range(len(titles)):
        axes[0, i].set_title(titles[i])
        axes[0, i].set_xlabel(xlabels[i])
        axes[0, i].set_ylabel(ylabels[i])

        for s in data[i].keys():
            axes[0, i].plot(data[i][s], label=s)

        axes[0, i].legend(loc="best")
        axes[0, i].grid()

    plt.tight_layout()
    plt.show()


# train_model()
