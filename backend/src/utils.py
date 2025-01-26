import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Tuple
from torch import tensor

from .models.scripts.dataset import LetterDataset

idx_to_letter = {
    l: i
    for i, l in LetterDataset(
        pd.read_pickle("./src/models/data/multi_shape_dataset.pkl").to_dict("records")
    ).letter_to_idx.items()
}


# split image to images with different letters
def split_image(
    image: np.ndarray,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    # check if the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # invert the image
    image = 255 - image

    # convert to binary
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # find the horizontal lines
    horizontal_lines = (np.sum(binary, axis=1) != 0) * 1
    # find the indices of the horizontal lines (after 0, before 0)
    horizontal_cuts = extract_segments(horizontal_lines)

    # get connected components in the vertical direction
    images = []
    for i, j in horizontal_cuts:
        V_bin = binary[i:j].T
        num_labels, labels_im = cv2.connectedComponents(V_bin)
        # split the image into multiple images, each containing a single connected component
        for label in range(1, num_labels):
            component = np.zeros_like(V_bin.T)
            component[labels_im.T == label] = 255
            dummy = np.zeros_like(binary)
            dummy[i:j] = component
            images.append(dummy)

    def manage_dot_image(dot_image: np.ndarray, idx: int) -> Tuple[np.ndarray, int]:
        # invert the dot_image
        dot_image = 255 - dot_image

        # find closest image to that dot_image
        if idx == 0:
            # the next image is the closest
            return merge_images([dot_image, images[idx + 1]]), idx + 1

        if idx == len(images) - 1:
            # the previous image is the closest
            return merge_images([images[idx - 1], dot_image]), idx - 1

        # find the closest image by getting the image that is on the bottom of the dot_image
        # get coordinates of before and after images and the dot_image
        dot_image_bottom = locate_empty_space(dot_image)[1]
        before_image_top = locate_empty_space(images[idx - 1])[0]
        after_image_top = locate_empty_space(images[idx + 1])[0]
        # get the distance between the dot_image and the before and after images
        before_distance = dot_image_bottom - before_image_top
        after_distance = after_image_top - dot_image_bottom
        # get the closest image
        if before_distance < after_distance:
            return merge_images([images[idx - 1], dot_image]), idx - 1
        return merge_images([dot_image, images[idx + 1]]), idx + 1

    # merge the dot images with the closest image
    img_sum = [np.sum(cv2.connectedComponents(img)[1]) for img in images]

    # mark the images that are not letters (less then 10% of the total pixels)
    for idx, img in enumerate(images):
        if img_sum[idx] < 0.1 * max(img_sum):
            # merge the dot image with the closest image
            img, n_idx = manage_dot_image(img, idx)
            images[n_idx] = img
            images[idx] = None

    # remove the None images
    images = [img for img in images if img is not None]

    # reinvert back the images
    images = [255 - img for img in images]

    # get the coordinates of each image
    coordinates = []
    for img in images:
        top, bottom, left, right = locate_empty_space(img)
        coordinates.append((top, bottom, left, right))

    boxes = create_box_image(coordinates, binary.shape)

    for box, img in zip(boxes, images):
        put_image_in_box(box, img)

    return boxes, coordinates


def create_box_image(coordinates, image_shape):
    boxes = []
    for i, j, x, y in coordinates:
        canvas = np.ones(image_shape, dtype=np.uint8) * 255
        canvas[i:j, x:y] = 0
        boxes.append(canvas)
    return boxes


def put_image_in_square(image):
    square_shape = max(image.shape)
    square_image = np.ones((square_shape, square_shape), dtype=np.uint8) * 255
    top = (square_shape - image.shape[0]) // 2
    left = (square_shape - image.shape[1]) // 2
    square_image[top : top + image.shape[0], left : left + image.shape[1]] = image
    return square_image


def extract_segments(line_segments: List[int]) -> List[Tuple[int, int]]:
    segments = []
    for i_idx, i in enumerate(line_segments):
        if i != 0 and line_segments[i_idx - 1] == 0:
            for j_idx, j in enumerate(line_segments[i_idx + 1 :]):
                if j == 0:
                    segments.append((i_idx, i_idx + j_idx))
                    break
    return segments


def locate_empty_space(image):
    top = 0
    bottom = 0
    left = 0
    right = 0

    for i in range(image.shape[0]):
        if np.any(image[i] == 0):
            top = i
            break

    for i in range(image.shape[0] - 1, -1, -1):
        if np.any(image[i] == 0):
            bottom = i
            break

    for i in range(image.shape[1]):
        if np.any(image[:, i] == 0):
            left = i
            break

    for i in range(image.shape[1] - 1, -1, -1):
        if np.any(image[:, i] == 0):
            right = i
            break

    return top, bottom, left, right


def box_object_in_image(image, i, j, x, y):
    image[i:j, x:y] = 0
    return image


def crop_image(image, top, bottom, left, right):
    return image[top : bottom + 1, left : right + 1]


def resize_image(image: torch.Tensor | np.ndarray, new_shape: tuple) -> torch.Tensor:
    """
    Resize an image tensor to the target shape using downsampling or upsampling.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
        new_shape (tuple): Target shape as (height, width).

    Returns:
        torch.Tensor: Resized image tensor.
    """
    # Validate input tensor shape
    if len(image.shape) != 4:
        # convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        # convert to 4D tensor
        image = image.unsqueeze(0).unsqueeze(0).float()

    return F.interpolate(image, size=new_shape, mode="bilinear", align_corners=False)


def put_image_in_box(boximage, object: np.ndarray):
    top, bottom, left, right = locate_empty_space(boximage)
    box_shape = (bottom - top + 1, right - left + 1)
    original_object = object.copy()
    # check if the box shape is larger than the object shape
    if object.shape != box_shape:
        # resize the object to fit the box
        object = (
            resize_image(
                torch.tensor(original_object).unsqueeze(0).unsqueeze(0).float(),
                box_shape,
            )
            .squeeze()
            .numpy()
        )
    boximage[top : bottom + 1, left : right + 1] = object


def merge_images(images: list[np.ndarray]) -> np.ndarray:
    # all images has the same shape
    max_height = images[0].shape[0]
    max_width = images[0].shape[1]

    # create a blank canvas
    canvas = np.ones((max_height, max_width), dtype=np.uint8) * 255

    threshold = np.mean(images)

    # put each image in the canvas
    for y in range(max_height):
        for x in range(max_width):
            for image in images:
                if image[y, x] < threshold:
                    canvas[y, x] = image[y, x]
    return canvas


def checker(image: np.ndarray) -> bool:
    top = image[0] > 127
    bottom = image[-1] > 127
    left = image[:, 0] > 127
    right = image[:, -1] > 127
    return top.all() and bottom.all() and left.all() and right.all()  # type: ignore


def generate_image(text: str, fonts: str, image_size=(28, 28)) -> np.ndarray:
    font_size = max(image_size)
    for _ in range(font_size - 1):
        image = Image.new("L", image_size, color="white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(fonts, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (image_size[0] - text_width) // 2
        text_y = (image_size[1] - text_height) // 2 - text_bbox[1]
        draw.text(
            (text_x, text_y),
            text,
            fill="black",
            font=font,
        )
        if checker(np.array(image)):
            break
        font_size -= 1
    return np.array(image)  # type: ignore


def get_font_path(font: str) -> str:
    temp = lambda x: f"./src/fonts/{x}.ttf"
    return temp(font)  # type: ignore


def predict_letter(model, image: np.ndarray):
    # check if the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    image = tensor(image).unsqueeze(0).unsqueeze(0).float()  # type: ignore
    idx: int = model(image).detach().numpy().argmax()
    return idx_to_letter[idx]
