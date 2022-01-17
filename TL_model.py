import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

BATCH_SIZE = 32
IMG_SIZE = (7, 7)
directory = "dataset/"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)

validation_dataset = image_dataset_from_directory(directory,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  validation_split=0.2,
                                                  subset='validation',
                                                  seed=42)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def data_augmenter():
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip("horizontal"))
    data_augmentation.add(RandomRotation(0.2))
    return data_augmentation


def model_net(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    input_shape = image_shape + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    # Freeze the base model by making it non trainable
    base_model.trainable = False

    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape)
    # apply data augmentation to the inputs
    x = data_augmentation(inputs)
    # data preprocessing using the same weights the model was trained on
    # Already Done -> preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    x = preprocess_input(x)
    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False)

    # Add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x)
    # include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.2)(x)

    # create a prediction layer with one neuron (as a classifier only needs one)
    prediction_layer = tfl.Dense(8)  # number of classes

    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model, base_model


def plot_accuracy(history):
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    plt.figure(figsize=(10, 6))
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')
    plt.show()


def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.show()

def fine_tune(base_model):
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    # Fine-tune from this layer onwards
    fine_tune_at = 126
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        print('Layer ' + layer.name + ' frozen.')
        layer.trainable = None


data_augmentation = data_augmenter()
features_model, base_model = model_net(IMG_SIZE, data_augmentation)
# Define a BinaryCrossentropy loss function. Use from_logits=True
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Define an Adam optimizer with a learning rate of 0.1 * base_learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 0.001
# Use accuracy as evaluation metric
metrics = ['accuracy']

features_model.compile(loss=loss_function,
                       optimizer=optimizer,
                       metrics=metrics)
initial_epochs = 5
history = features_model.fit(train_dataset,
                             validation_data=validation_dataset,
                             epochs=initial_epochs)

plot_loss(history)
plot_accuracy(history)

fine_tune(base_model)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

features_model.compile(loss=loss_function,
                       optimizer=optimizer,
                       metrics=metrics)

fine_tune_epochs = 5
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = features_model.fit(train_dataset,
                                  epochs=total_epochs,
                                  initial_epoch=history.epoch[-1],
                                  validation_data=validation_dataset)
plot_loss(history)
plot_accuracy(history)

