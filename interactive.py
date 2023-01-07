from module.generator import ImageGenerator, ImageResize, ImageRotate, ImageIntensity, ImageRemoveZeros, ImageAddNoise, StretchContrast
from mnist import MNIST
import numpy as np
import pygame
import sys
import tensorflow as tf
import time

tf.get_logger().setLevel(50)

mnist = MNIST("./mnist/", return_type="numpy")
images, labels = mnist.load_training()
images = images.reshape(-1, 28, 28).astype(np.uint8)

generator = ImageGenerator(images, labels)
generator.add_layer(ImageRemoveZeros())
generator.add_layer(ImageRotate(-30, 30))
generator.add_layer(ImageResize(2, 12))
generator.add_layer(StretchContrast())

# https://stackoverflow.com/a/32848377
def tile_array(a, b0, b1):
    r, c = a.shape
    rs, cs = a.strides
    x = np.lib.stride_tricks.as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0))
    return x.reshape(r*b0, c*b1)

i = 0

SIZE = 512
PAINT_SIZE = 25
CURRENT_LAYER = 0

SIZES = {
    pygame.K_q: 0,
    pygame.K_w: 10,
    pygame.K_e: 20,
    pygame.K_r: 30,
    pygame.K_t: 40,
    pygame.K_y: 50,
    pygame.K_u: 60,
    pygame.K_i: 70,
    pygame.K_o: 80,
    pygame.K_p: 90,
}

LAYERS = {
    pygame.K_1: 2,
    pygame.K_2: 3,
    pygame.K_3: 4,
    pygame.K_4: 5,
    pygame.K_5: 6,
    pygame.K_6: 7,
    pygame.K_7: 8,
    pygame.K_8: 9,
    pygame.K_9: 10,
    pygame.K_0: 1,
    pygame.K_BACKQUOTE: 0,
}


mouseX, mouseY = 0, 0
lastX, lastY = 0, 0
click = False
predict = False
running = True
predictions = np.ones(shape=(32, 32, 11), dtype=int)

pygame.init()
from module.model import model

if len(sys.argv) == 1:
    print("Need path to model weights")
    exit(-1)

model.load_weights(sys.argv[1])

def draw(x: int, y: int, lastX, lastY, size: int, screen: np.ndarray) -> None:
    left = np.clip(x-size, 0, SIZE)
    top = np.clip(y-size, 0, SIZE)
    pygame.draw.rect(screen, (255, 255, 255), pygame.rect.Rect(left - (size//2), top - (size//2), size, size), 0)

screen_pygame = pygame.display.set_mode(size=(2*SIZE, SIZE))

clock = pygame.time.Clock()

def set_caption():
    pygame.display.set_caption(f"bercikpaint | W:{CURRENT_LAYER-1} | S:{PAINT_SIZE}")

set_caption()

while running:
    clock.tick(120)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.MOUSEMOTION:
            lastX, lastY, (mouseX, mouseY) = mouseX, mouseY, pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        elif event.type == pygame.MOUSEBUTTONUP:
            click = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            predict = True
            click = False
            print("predict")
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
            screen_pygame.fill((0, 0, 0))
            print("clear")
        elif event.type == pygame.KEYDOWN and event.key in SIZES:
            PAINT_SIZE = SIZES[event.key]
        elif event.type == pygame.KEYDOWN and event.key in LAYERS:
            CURRENT_LAYER = LAYERS[event.key]
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
            image, _ = generator.generate_single_image(i, n=1)
            i += 1
            temp_surf = pygame.surfarray.pixels3d(screen_pygame)[:512, :512]
            for x in range(image[0].shape[0]):
                for y in range(image[0].shape[1]):
                    temp_surf[x, y] = int(image[0, y, x] * 255)
            del temp_surf

    if click:
        draw(mouseX, mouseY, lastX, lastY, PAINT_SIZE, screen_pygame)

    if predict:
        t_ = time.perf_counter()
        surface = pygame.surfarray.pixels3d(screen_pygame)[:512, :512, 0].T.reshape(1, 512, 512, 1).astype(float) / 255

        predictions = np.swapaxes((model.predict(surface, verbose=False)[0]), 0, 1)
        print(f"took {(time.perf_counter() - t_) / 1000}")
        predict = False


    prediction = pygame.surfarray.make_surface(tile_array(predictions[..., :, CURRENT_LAYER] * 12, 16, 16))
    screen_pygame.blit(prediction, (512, 0))

    set_caption()
    pygame.display.flip()

pygame.quit()   
