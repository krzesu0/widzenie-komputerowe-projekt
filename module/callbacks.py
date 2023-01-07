import tensorflow as tf
from .generator import ImageGenerator
import matplotlib.pyplot as plt


class GenerateSummaryImage(tf.keras.callbacks.Callback):
    generator: ImageGenerator

    def __init__(self, generator: ImageGenerator, location: str):
        self.generator = generator
        self.location = location

    def on_epoch_end(self, epoch, logs=None):

        # reset seeds on all layers
        self.generator.reset()

        fig, ax = plt.subplots(2, 13, figsize=(12, 2))
        (image1, label1), (image2, label2) = self.generator.generate_single_image(10, n=1), self.generator.generate_single_image(20, n=1)
        resp1, resp2 = self.model.predict(image1, verbose=0), self.model.predict(image2, verbose=0)

        ax[0, 0].imshow(label1[0])
        ax[0, 1].imshow(image1[0])
        for i in range(11):
            ax[0, 2+i].imshow(resp1[0, :, :, i], vmin=0, vmax=1)
            ax[0, 2+i].axis("off")
        ax[0, 0].axis("off")
        ax[0, 1].axis("off")
        ax[0, 2].axis("off")

        ax[1, 0].imshow(label2[0])
        ax[1, 1].imshow(image2[0])
        for i in range(11):
            ax[1, 2+i].imshow(resp2[0, :, :, i], vmin=0, vmax=1)
            ax[1, 2+i].axis("off")
        ax[1, 0].axis("off")
        ax[1, 1].axis("off")
        ax[1, 2].axis("off")

        fig.savefig(f"{self.location}{epoch}.pdf", bbox_inches="tight", pad_inches=0, dpi=500)
        plt.close("all")
