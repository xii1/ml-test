import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential, models
from tensorflow.keras import activations, losses, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SAVED_MODEL = 'trained_models/classification/vgg16_dog_cat_model'

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3

N_CLASSES = 2
BATCH_SIZE = 32


def train_model_by_transfer_learning_with_vgg16():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, rotation_range=10, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('data/dog_cat/training_set', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                     batch_size=BATCH_SIZE, class_mode='categorical')
    test_set = test_datagen.flow_from_directory('data/dog_cat/test_set', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                batch_size=BATCH_SIZE, class_mode='categorical')

    # Load VGG-16 pretrained model
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), pooling='max')
    # Freeze all the layers (do not train again)
    vgg.trainable = False
    vgg.summary()

    # Define custom model on top of pretrained model
    model = Sequential()
    model.add(vgg)
    # model.add(Flatten())
    model.add(Dense(128, activation=activations.relu))
    model.add(Dense(N_CLASSES, activation=activations.softmax))
    model.summary()

    # Compile & train model
    model.compile(optimizer=optimizers.Adam(), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    callback = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5, restore_best_weights=True)
    history = model.fit(training_set, validation_data=test_set, epochs=20, callbacks=callback)
    model.save(SAVED_MODEL)

    # Visualize loss & accuracy
    visualize([{"train": history.history["loss"], "validate": history.history["val_loss"]},
               {"train": history.history["accuracy"], "validate": history.history["val_accuracy"]}],
              ["Model Loss", "Model Accuracy"],
              ["epoch", "epoch"], ["loss", "accuracy"])

    return


def predict_image(img):
    resized_img = Image.open(img).convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    data = np.array(resized_img)
    data = data.astype('float32') / 255
    data = np.expand_dims(data, axis=0)
    model = models.load_model(SAVED_MODEL)
    predict = np.squeeze(model.predict(data))

    if predict[0] > 0.5:
        return 'Cat (%1.2f)' % predict[0]
    else:
        return 'Dog (%1.2f)' % predict[1]


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


# train_model_by_transfer_learning_with_vgg16()
