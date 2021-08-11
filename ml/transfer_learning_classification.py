import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import activations, losses, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

SAVED_WEIGHTS = 'trained_models/classification/vgg16_dog_cat_model_weights.h5'

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CHANNELS = 3

N_CLASSES = 2
BATCH_SIZE = 32

NAME = {0: 'Cat', 1: 'Dog'}


def create_model_with_vgg16():
    # Load VGG-16 pretrained model
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), pooling='max')
    # Freeze all the layers (do not train again)
    vgg.trainable = False

    # Define custom model on top of pretrained model
    fc_layer = Dense(128, activation=activations.relu)(vgg.output)
    output_layer = Dense(N_CLASSES, activation=activations.softmax)(fc_layer)

    model = Model(inputs=vgg.input, outputs=output_layer)
    model.compile(optimizer=optimizers.Adam(), loss=losses.categorical_crossentropy, metrics=['accuracy'])

    return model


def train_model_by_transfer_learning_with_vgg16():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, rotation_range=10,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('data/dog_cat/training_set',
                                                     target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                     batch_size=BATCH_SIZE, class_mode='categorical')
    test_set = test_datagen.flow_from_directory('data/dog_cat/test_set', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                batch_size=BATCH_SIZE, class_mode='categorical')

    model = create_model_with_vgg16()
    model.summary()

    # ModelCheckpoint callback to save best weights
    checkpoint_callback = ModelCheckpoint(filepath=SAVED_WEIGHTS, save_best_only=True, verbose=1)

    # EarlyStopping callback to stop train
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5,
                                        restore_best_weights=True)

    # Train model
    history = model.fit(training_set, validation_data=test_set, epochs=20,
                        callbacks=[early_stop_callback, checkpoint_callback])

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
    input_data = np.expand_dims(data, axis=0)

    model = create_model_with_vgg16()
    model.load_weights(SAVED_WEIGHTS)
    predict = np.squeeze(model.predict(input_data))

    # get top 2 maximum indices
    indices = (-predict).argsort()[:2]

    cam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    # cam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)
    heatmaps = cam(CategoricalScore([indices[0], indices[1]]), np.array([data, data]), penultimate_layer=-1)

    return '{} ({:.2f}) | {} ({:.2f})'.format(NAME[indices[0]], predict[indices[0]],
                                              NAME[indices[1]], predict[indices[1]]), heatmaps


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
