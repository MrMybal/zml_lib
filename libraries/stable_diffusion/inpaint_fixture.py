"""Write a deterministic landscape and a grayscale mask for inpainting tests."""

from PIL import Image, ImageDraw


def write_fixture(directory, width, height):
    source = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(source)
    horizon = height * 5 // 8
    for y in range(horizon):
        draw.line((0, y, width - 1, y), fill=(38 + 145 * y // horizon, 94 + 106 * y // horizon, 181 + 52 * y // horizon))
    draw.rectangle((0, horizon, width - 1, height - 1), fill=(32, 100, 127))
    draw.polygon([(0, horizon + height // 12), (width // 4, horizon - height // 8),
                  (width // 2, horizon + height // 10), (0, horizon + height // 7)], fill=(32, 67, 59))
    draw.rectangle((width // 5, height // 3, width // 4, horizon), fill=(238, 211, 171))
    draw.polygon([(width // 6, height // 3), (width // 3, height // 3), (width // 4, height // 4)], fill=(157, 44, 32))
    left, top, right, bottom = width * 3 // 5, height * 2 // 5, width * 4 // 5, height * 3 // 4
    draw.rectangle((left, top, right, bottom), fill=(136, 106, 75))
    mask = Image.new("L", (width, height))
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((left - 4, top - 4, right + 4, bottom + 4), radius=8, fill=255)
    draw.line((left - 2, top, left - 2, bottom), fill=128)
    draw.line((right + 2, top, right + 2, bottom), fill=127)
    image_path, mask_path = directory / "landscape.png", directory / "mask.png"
    source.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path
