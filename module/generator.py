from abc import abstractmethod, ABC
import numpy as np
from PIL import Image

from typing import List, Tuple


class ImagePreprocessor(ABC):

    def __init__(self, seed: int = 0x80085) -> None:
        self.seed = seed
        self._reset()

    def _reset(self):
        self.generator = np.random.default_rng(self.seed)

    @abstractmethod
    def generate(self, data: np.ndarray) -> np.ndarray:
        ...


class ImageRotate(ImagePreprocessor):
    deg_min: float
    deg_max: float

    def __init__(self, deg_min: float, deg_max: float, **kwargs):
        super().__init__(**kwargs)
        self.deg_min = deg_min
        self.deg_max = deg_max

    def _get_random_angle(self) -> float:
        return self.generator.uniform(self.deg_min, self.deg_max)

    def generate(self, data: np.ndarray) -> np.ndarray:
        image = Image.fromarray(data)
        size = data.shape[0]
        image = image.rotate(self._get_random_angle(), Image.BICUBIC, expand=True)
        return np.asarray(image.resize((size, size), Image.BICUBIC), dtype=np.uint8)


class ImageResize(ImagePreprocessor):
    scale_min: float
    scale_max: float

    def __init__(self, scale_min: float, scale_max: float, **kwargs):
        super().__init__(**kwargs)
        self.scale_min = scale_min
        self.scale_max = scale_max

    def _get_random_scale(self) -> float:
        return self.generator.uniform(self.scale_min, self.scale_max)

    def generate(self, data: np.ndarray) -> np.ndarray:
        image = Image.fromarray(data)
        size = int(data.shape[0] * self._get_random_scale())
        return np.asarray(image.resize((size, size), Image.BICUBIC), dtype=np.uint8)


class ImageIntensity(ImagePreprocessor):
    def __init__(self, intensity_min: float, intensity_max: float, **kwargs):
        super().__init__(**kwargs)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        if intensity_max > 1 or intensity_max <= 0:
            raise Exception("Invalid max intensity")

        if intensity_min < 0 or intensity_min >= 1:
            raise Exception("Invalid min intensity")

        if intensity_min > intensity_max:
            raise Exception("intensity_min > intensity_max")
    

    def _get_random_intensity(self) -> float:
        return self.generator.uniform(self.intensity_min, self.intensity_max)

    def generate(self, data: np.ndarray) -> np.ndarray:
        return (data * self._get_random_intensity()).astype(np.uint8)


class ImageRemoveZeros(ImagePreprocessor):

    def generate(self, data: np.ndarray) -> np.ndarray:
        left = np.min([np.where(row != 0)[0][0] for row in data if np.where(row != 0)[0].size > 0])
        up = np.min([np.where(column != 0)[0][0] for column in data.T if np.where(column != 0)[0].size > 0])
        right = np.max([np.where(row != 0)[0][-1] for row in data if np.where(row != 0)[0].size > 0])
        down = np.max([np.where(column != 0)[0][-1] for column in data.T if np.where(column != 0)[0].size > 0])
        return data[up:down, left:right].astype(np.uint8)


class ImageAddNoise(ImagePreprocessor):
    noise_low: int
    noise_high: int

    def __init__(self, noise_low: int, noise_high: int, **kwargs):
        super().__init__(**kwargs)
        self.noise_low = noise_low
        self.noise_high = noise_high

    def generate(self, data: np.ndarray) -> np.ndarray:
        noise = self.generator.integers(self.noise_low, self.noise_high, size=data.shape, dtype=np.uint8)
        return np.clip(np.add(data, noise, dtype=np.uint16), 0, 255).astype(np.uint8)


class StretchContrast(ImagePreprocessor):

    def generate(self, data: np.ndarray) -> np.ndarray:
        return (data * (255/np.max(data))).astype(np.uint8)


class ImageGenerator:
    _layers: List[ImagePreprocessor]

    def __init__(self, source_data: np.ndarray, source_labels: np.ndarray, seed=0xDEADBEEF):
        assert source_data.shape[0] == source_labels.shape[0]
        self._layers = []
        self.source_data = source_data
        self.source_labels = source_labels
        self.seed = seed
        self.reset()

    def reset(self):
        self.generator = np.random.default_rng(self.seed)
        for layer in self._layers:
            layer._reset()

    def add_layer(self, layer: ImagePreprocessor):
        self._layers.append(layer)

    def _perform_generate(self, layer_index: int, data: np.ndarray):
        curr = self._layers[layer_index].generate(data)
        if layer_index:
            return self._perform_generate(layer_index - 1, curr)
        else:
            return curr

    def generate(self, source_data: np.ndarray) -> np.ndarray:
        return self._perform_generate(len(self._layers) - 1, source_data)

    def generate_single_image(self, idx: int, noise: bool = True, n=2):
        image = np.zeros(shape=(512, 512), dtype=np.uint16)
        locations = [] # x, y, width, height, label
        for digit, label in zip(list(map(self.generate, self.source_data[idx:idx+n])), self.source_labels[idx:idx+n]):
            x = self.generator.integers(0, 511 - digit.shape[0])
            y = self.generator.integers(0, 511 - digit.shape[1])
            image[x:x + digit.shape[0], y:y+digit.shape[1]] += digit

            locations.append((x, y, digit, label))

        if noise:
            noise_data = self.generator.integers(0, 70, size=image.shape, dtype=np.uint8)
            data = np.clip(image + noise_data, 0, 255).astype(np.uint8).reshape(1, 512, 512)
        else:
            data = np.clip(image, 0, 255).astype(np.uint8).reshape(1, 512, 512)

        return (data / 255).astype(np.float32), self.get_response(locations)

    def get_image(self, n_of_images: int, noise=True, n=2) -> Tuple[np.ndarray, list]:
        for k in range(n_of_images):
            yield self.generate_single_image(k, noise, n)

    def get_response(self, locations: list) -> np.ndarray:
        # 32 x 32 x 1
        resp = np.zeros(shape=(1, 512//16, 512//16), dtype=np.float32)
        for location in locations:
            x, y, digit, label = location
            digit_width, digit_height, *_ = digit.shape
            scaled = np.array(Image.fromarray(digit).resize((digit_height // 16, digit_width // 16)))
            scaled_width, scaled_height, *_ = scaled.shape
            _x, _y = x//16, y//16

            resp[0, _x:_x + scaled_width, _y:_y + scaled_height] = np.where(scaled > 0, label + 1, 0)
        return resp