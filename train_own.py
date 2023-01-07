from train_defaults import *

from module.model import model

if __name__ == "__main__":
    model.summary()

    model.compile(**generate_compile_params())
    model.fit(**generate_fit_params("own"))
