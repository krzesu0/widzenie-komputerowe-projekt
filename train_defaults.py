from module.generator import ImageGenerator, ImageResize, ImageRotate, ImageRemoveZeros
from module.sequence import CustomGenerator
from module.callbacks import GenerateSummaryImage
import tensorflow as tf
import os

import numpy as np
from mnist import MNIST

LEARN = 1000

mnist = MNIST("./mnist/", return_type="numpy")
images, labels = mnist.load_training()
images = images.reshape(-1, 28, 28).astype(np.uint8)

generator = ImageGenerator(images, labels)
generator.add_layer(ImageRemoveZeros())
generator.add_layer(ImageRotate(-15, 15))
generator.add_layer(ImageResize(5, 10))

generator_val = ImageGenerator(images[LEARN:], labels[LEARN:])
generator_val.add_layer(ImageRemoveZeros())
generator_val.add_layer(ImageRotate(-15, 15))
generator_val.add_layer(ImageResize(5, 10))

values_generator = CustomGenerator(LEARN, generator, n=1)
valida_genetator = CustomGenerator(100, generator_val, n=1)

def _try(func, exc, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except exc:
        pass

def generate_fit_params(name: str, epochs: int = 150):

    _try(os.mkdir, Exception, f"output/{name}/")
    _try(os.mkdir, Exception, f"output/{name}/log/")
    _try(os.mkdir, Exception, f"output/{name}/model/")
    _try(os.mkdir, Exception, f"output/{name}/img/")


    return {
        "x":values_generator,
        "batch_size": 1,
        "epochs": epochs,
        "verbose": 1,
        "validation_data": valida_genetator,
        "use_multiprocessing":False,
        "workers":1,
        "callbacks": [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.CSVLogger(f"output/{name}/log/loss.csv"),
            tf.keras.callbacks.ModelCheckpoint(f"output/{name}/model/{{epoch}}", save_freq=LEARN * 10, save_weights_only=True),
            GenerateSummaryImage(generator_val, f"output/{name}/img/"),
        ]}

def generate_compile_params(lr: int = 0.01):
    return {
        "loss": tf.losses.SparseCategoricalCrossentropy(),
        "metrics": [
            "accuracy",
            tf.losses.SparseCategoricalCrossentropy(ignore_class=0, name="loss_no_background")
        ],
        "optimizer": tf.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True),
    }