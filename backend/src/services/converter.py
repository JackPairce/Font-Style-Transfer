from PIL import Image
from io import BytesIO
import numpy as np
import base64


def image2matrix(image):
    # from base64
    image = Image.open(BytesIO(base64.b64decode(image)))
    return np.array(image)


def matrix2image(matrix: np.ndarray):
    img = Image.fromarray(matrix)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # to base64
    base64_img = base64.b64encode(img_byte_arr.read())
    return base64_img.decode("utf-8")
