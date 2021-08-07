import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential, models
from tensorflow.keras import activations, losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SAVED_MODEL = 'trained_models/classification/cifar10_model'
BATCH_SIZE = 32
NUM_CLASSES = 10
NAME = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer',
        5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'}


def train_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, rotation_range=10, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_set = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    test_set = test_datagen.flow(x_test, y_test, batch_size=BATCH_SIZE)

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation=activations.relu))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation=activations.relu))
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
    history = model.fit(train_set, validation_data=test_set, epochs=20, callbacks=callback)
    model.save(SAVED_MODEL)

    visualize([{"train": history.history["loss"], "validate": history.history["val_loss"]},
               {"train": history.history["accuracy"], "validate": history.history["val_accuracy"]}],
              ["Model Loss", "Model Accuracy"],
              ["epoch", "epoch"], ["loss", "accuracy"])

    return


def predict_cifar10(img):
    resized_img = Image.open(img).convert('RGB').resize((32, 32))
    data = np.array(resized_img)
    data = data.astype('float32') / 255
    data = np.expand_dims(data, axis=0)
    model = models.load_model(SAVED_MODEL)
    predict = np.squeeze(model.predict(data))
    predict = np.argmax(predict, axis=-1)

    return 'Name: {}'.format(NAME[predict])


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
