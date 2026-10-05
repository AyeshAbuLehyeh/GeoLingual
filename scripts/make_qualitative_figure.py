
from PIL import Image, ImageDraw, ImageFont

from geolingual.dataset_utils import pano_image_path, sat_image_path

QID = "Chicago_8GIqesI3QHS9IHfbcIzGaA,41.887583,-87.648761,.jpg"
CITY = "Chicago"
PANO_FILE = "8GIqesI3QHS9IHfbcIzGaA,41.887583,-87.648761,.jpg"
TRUE_SAT = "satellite_41.887485897337505_-87.64890082900035.png"
TEXT_WRONG_SAT = "satellite_41.87337878187409_-87.68150444398977.png"

TEXT_REASON = ('Text-only picked candidate #3 instead: "matches the query\'s T-intersection, '
               'looped road topology, and distinctive anchors including the red arched roof '
               'building and red car" -- description-level similarity, but the WRONG location. '
               'True match (#7) was ranked 8th of 10.')
IMAGE_REASON = ('Image-only correctly picked candidate #7 (rank 1): "the ground-level panorama '
                'shows a parking lot with a red-roofed building and a white delivery truck, which '
                'matches the satellite view of candidate 7, where the same building, truck, and '
                'parking layout are visible."')


def load(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    pano = load(pano_image_path(CITY, PANO_FILE), size=(760, 380))
    true_sat = load(sat_image_path(CITY, TRUE_SAT), size=(320, 320))
    wrong_sat = load(sat_image_path(CITY, TEXT_WRONG_SAT), size=(320, 320))

    W, H = 1280, 900
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_h = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font_h = font_b = ImageFont.load_default()

    draw.text((20, 20), "QUERY: Ground-level panorama", font=font_h, fill="black")
    canvas.paste(pano, (20, 50))

    draw.text((820, 20), "TRUE MATCH (satellite, #7)", font=font_h, fill="#2e7d32")
    canvas.paste(true_sat, (820, 50))
    draw.rectangle([818, 48, 820 + 320 + 2, 50 + 320 + 2], outline="#2e7d32", width=4)

    draw.text((20, 460), "Text-only's top pick (candidate #3) -- WRONG", font=font_h, fill="#c62828")
    canvas.paste(wrong_sat, (20, 490))
    draw.rectangle([18, 488, 20 + 320 + 2, 490 + 320 + 2], outline="#c62828", width=4)

    text_x = 380
    draw.text((text_x, 490), "Text-only reasoning (rank 8/10 for true match):", font=font_h, fill="#c62828")
    for i, line in enumerate(wrap_text(draw, TEXT_REASON, font_b, 760)):
        draw.text((text_x, 520 + i * 20), line, font=font_b, fill="black")

    draw.text((text_x, 660), "Image-only reasoning (rank 1/10 for true match):", font=font_h, fill="#2e7d32")
    for i, line in enumerate(wrap_text(draw, IMAGE_REASON, font_b, 760)):
        draw.text((text_x, 690 + i * 20), line, font=font_b, fill="black")

    out_path = "figures/qualitative_hard_example.png"
    canvas.save(out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
