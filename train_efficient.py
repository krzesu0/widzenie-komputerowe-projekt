from train_defaults import *

from module.efficientnet import model

values_generator.preprocess = lambda a: a*255
valida_genetator.preprocess = lambda a: a*255

if __name__ == "__main__":
    model.summary()
    model.compile(**generate_compile_params())
    model.fit(**generate_fit_params("efficient"))
