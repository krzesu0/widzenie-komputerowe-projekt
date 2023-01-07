from train_defaults import *

from module.mobilenet import model

values_generator.preprocess = tf.keras.applications.mobilenet.preprocess_input
valida_genetator.preprocess = tf.keras.applications.mobilenet.preprocess_input

if __name__ == "__main__":
    model.summary()

    model.compile(**generate_compile_params())
    model.fit(**generate_fit_params("mobile"))