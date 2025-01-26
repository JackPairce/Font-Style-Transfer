from typing import List
import numpy as np


from ..utils import (
    box_object_in_image,
    generate_image,
    get_font_path,
    merge_images,
    put_image_in_box,
    crop_image,
    locate_empty_space,
    resize_image,
    split_image,
    predict_letter,
    put_image_in_square,
)


def Predict(model, image: np.ndarray) -> str:
    # split image to images with different letters
    images, _ = split_image(image)

    # crop only the letter
    images = [crop_image(img, *locate_empty_space(img)) for img in images]

    # squarify each image
    images = [put_image_in_square(img) for img in images]

    # resize each image to 28x28
    images = [resize_image(img, (28, 28)).squeeze().numpy() for img in images]

    # predict each letter
    predicted_text: List[str] = [predict_letter(model, img) for img in images]

    return "".join(predicted_text)


def get_box_shape(box):
    top, bottom, left, right = locate_empty_space(box)
    return (bottom - top + 1, right - left + 1)


def Generate(
    image: np.ndarray,
    text: str,
    fonts: str,
) -> np.ndarray:
    # split image to images with different letters
    images, coordinates = split_image(image)

    # get box of each image
    boxes = [
        box_object_in_image(img, *coord) for img, coord in zip(images, coordinates)
    ]

    # get box shape
    # box_shapes = [get_box_shape(box) for box in boxes]

    coord2shape = lambda coord: (coord[1] - coord[0] + 1, coord[3] - coord[2] + 1)

    # generate image for each letter
    target_images = [
        generate_image(letter, get_font_path(fonts), coord2shape(coord))
        for letter, coord in zip(text, coordinates)
    ]
    # crop only the letter
    target_images = [crop_image(img, *locate_empty_space(img)) for img in target_images]

    # put each letter in each box
    for idx, (target_image, box) in enumerate(zip(target_images, boxes)):
        put_image_in_box(box, target_image)
        boxes[idx] = box

    # merge all images
    image = merge_images(boxes)

    return image
