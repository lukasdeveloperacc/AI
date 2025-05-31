from PIL import Image

import requests, io, logging


def get_image_from_url(url: str) -> Image.Image | None:
    try:
        if not url:
            return None

        image = None
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))

        if not isinstance(image, Image.Image):
            raise ValueError(f"Image is not exist : {type(image)}")

        return image

    except Exception as e:
        logging.error(e)
        return None
