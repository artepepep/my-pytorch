import struct


def load_images(path):
    with open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))

        images = []

        for _ in range(count):
            pixels = file.read(rows * cols)

            image = [
                pixel / 255.0
                for pixel in pixels
            ]

            images.append(image)

        return images


def load_labels(path):
    with open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))

        labels = list(file.read(count))

        return labels