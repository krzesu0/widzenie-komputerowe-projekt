import tensorflow as tf
from .generator import ImageGenerator

from typing import Callable, Any

class CustomGenerator(tf.keras.utils.Sequence):
    generator_instance: ImageGenerator
    n_of_images: int
    n: int

    def __init__(self, n_of_images: int, generator: ImageGenerator, n: int = 1, preprocess: Callable = lambda a: a):
        self.generator_instance = generator
        self.n_of_images = n_of_images
        self.n = n
        self.preprocess = preprocess

    def __len__(self):
        return self.n_of_images
        
    def __getitem__(self, idx):
        x, y = self.generator_instance.generate_single_image(idx * self.n, n=self.n)
        return self.preprocess(x), y