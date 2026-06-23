from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_has_iccp_chunk(path):
    try:
        with Path(path).open("rb") as file:
            if file.read(8) != PNG_SIGNATURE:
                return False

            while True:
                length_bytes = file.read(4)
                if len(length_bytes) != 4:
                    return False

                chunk_length = int.from_bytes(length_bytes, "big")
                chunk_type = file.read(4)

                if len(chunk_type) != 4:
                    return False

                if chunk_type == b"iCCP":
                    return True

                file.seek(chunk_length + 4, 1)

                if chunk_type == b"IEND":
                    return False
    except OSError:
        return False


def safe_read_image(path, skip_iccp=True):
    path = Path(path)

    if skip_iccp and png_has_iccp_chunk(path):
        return None

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.array(rgb)
    except (OSError, UnidentifiedImageError, ValueError):
        return None

    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def collect_problem_pngs(root_dir, include_unreadable=False):
    problem_files = []

    for path in Path(root_dir).rglob("*.png"):
        reasons = []

        if png_has_iccp_chunk(path):
            reasons.append("iCCP profile")

        if include_unreadable and safe_read_image(path, skip_iccp=False) is None:
            reasons.append("unreadable")

        if reasons:
            problem_files.append((path, reasons))

    return problem_files
