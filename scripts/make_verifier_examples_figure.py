
from PIL import Image, ImageDraw, ImageFont

from geolingual.dataset_utils import load
from geolingual.dataset_utils import pano_image_path, sat_image_path

GROUND_KEYS = ["road_topology_360", "spatial_layout", "distinctive_anchors", "orientation_cues"]
SAT_KEYS = ["cardinal_orientation", "road_geometry", "linear_sequence", "distinctive_anchors"]

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


def fonts():
    try:
        return (
            ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 20),
            ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 16),
            ImageFont.truetype(FONT_DIR + "DejaVuSans.ttf", 14),
        )
    except OSError:
        f = ImageFont.load_default()
        return f, f, f


F_H, F_SUB, F_B = fonts()


def load_img(path, size):
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)


def wrap(draw, text, font, max_width):
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


def draw_field_block(draw, x, y, title, fields, keys, max_width, color):
    draw.text((x, y), title, font=F_SUB, fill=color)
    y += 24
    for k in keys:
        val = fields.get(k, "(not described)")
        label = k.replace("_", " ")
        lines = wrap(draw, f"{label}: {val}", F_B, max_width)
        for line in lines:
            draw.text((x, y), line, font=F_B, fill="black")
            y += 18
        y += 4
    return y


def build_figure(out_path, header, ground_img_path, ground_fields,
                  main_sat_img_path, main_sat_fields, main_sat_label, main_sat_color,
                  verdict_score, verdict_label, verdict_reason, verdict_color,
                  ref_sat_img_path=None, ref_sat_label=None):
    W = 1300
    H = 825
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    img_top = 25
    ground_img = load_img(ground_img_path, (620, 310))
    canvas.paste(ground_img, (15, img_top))
    draw.rectangle([13, img_top - 2, 15 + 620 + 2, img_top + 310 + 2], outline="#555555", width=3)
    draw.text((15, img_top - 22), "GROUND PANORAMA (query)", font=F_SUB, fill="#333333")

    sat_img = load_img(main_sat_img_path, (300, 300))
    sat_x = 670
    canvas.paste(sat_img, (sat_x, img_top))
    draw.rectangle([sat_x - 2, img_top - 2, sat_x + 300 + 2, img_top + 300 + 2], outline=main_sat_color, width=4)
    draw.text((sat_x, img_top - 22), main_sat_label, font=F_SUB, fill=main_sat_color)

    if ref_sat_img_path is not None:
        ref_img = load_img(ref_sat_img_path, (180, 180))
        ref_x = sat_x + 320
        canvas.paste(ref_img, (ref_x, img_top))
        draw.rectangle([ref_x - 2, img_top - 2, ref_x + 180 + 2, img_top + 180 + 2], outline="#2e7d32", width=3)
        draw.text((ref_x, img_top - 22), ref_sat_label, font=F_B, fill="#2e7d32")

    y = img_top + 330

    y = draw_field_block(draw, 15, y, "Ground description (fields shown to the verifier):",
                         ground_fields, GROUND_KEYS, W - 30, "#333333")
    y += 10
    y = draw_field_block(draw, 15, y, f"Satellite candidate description ({main_sat_label}):",
                         main_sat_fields, SAT_KEYS, W - 30, main_sat_color)
    y += 15

    box_top = y
    draw.rectangle([10, box_top, W - 10, H - 10], outline=verdict_color, width=3)
    y += 12
    draw.text((22, y), f"VERIFIER OUTPUT:  score={verdict_score}/10   verdict={verdict_label}",
              font=F_SUB, fill=verdict_color)
    y += 26
    for line in wrap(draw, f'"{verdict_reason}"', F_B, W - 60):
        draw.text((22, y), line, font=F_B, fill="black")
        y += 19

    canvas.save(out_path)
    print(f"Saved -> {out_path}")


def main():
    records = load()
    by_pano = {f"{r['city']}_{r['pano_file']}": r for r in records}

    qid1 = "NewYork_rVDp_-ir3OYqrC5AsN0kOg,40.753628,-73.985201,.jpg"
    r1 = by_pano[qid1]
    build_figure(
        "figures/verifier_example_correct.png",
        "Verifier Example 1: Correct match, grounded in named permanent landmarks",
        pano_image_path("NewYork", r1["pano_file"]), r1["pano_fields"],
        sat_image_path("NewYork", r1["sat_file"]), r1["sat_fields"],
        "TRUE MATCH (satellite)", "#2e7d32",
        9, "MATCH", ("The ground-level description of a complex four-way intersection with a modern "
                     "glass building, food cart, construction site, and high-rise office building aligns "
                     "with the satellite view of the intersection of 5th Avenue and W 40th Street in "
                     "Midtown Manhattan, including the curved corner of 4 Times Square and the grid layout."),
        "#2e7d32",
    )

    qid2 = "Chicago_S5aNH61g2dA_2So-w_CZqA,41.894077,-87.667156,.jpg"
    r2 = by_pano[qid2]
    wrong_sat_file = "satellite_41.8943754188429_-87.66696499406206.png"
    wrong_sat_fields = None
    for rr in records:
        if rr["city"] == "Chicago" and rr["sat_file"] == wrong_sat_file:
            wrong_sat_fields = rr["sat_fields"]
            break
    build_figure(
        "figures/verifier_example_false_match.png",
        "Verifier Example 2: False match, leaning on a transient object (a parked van)",
        pano_image_path("Chicago", r2["pano_file"]), r2["pano_fields"],
        sat_image_path("Chicago", wrong_sat_file), wrong_sat_fields,
        "WRONG CANDIDATE (accepted)", "#c62828",
        6, "MATCH", ("Both descriptions feature a multi-lane road with a T-intersection, a Marathon gas "
                     "station on the right, residential buildings on the left, and a white van; the road "
                     "orientation and morning sunlight suggest consistent alignment, though satellite "
                     "details like the flat-roofed building and tree row are not confirmed in the "
                     "ground-level view."),
        "#c62828",
        ref_sat_img_path=sat_image_path("Chicago", r2["sat_file"]), ref_sat_label="(true match, for reference)",
    )

    qid3 = "Chicago_E0I7n6rC71oJJ6_WwrXw_Q,41.876906,-87.681678,.jpg"
    r3 = by_pano[qid3]
    build_figure(
        "figures/verifier_example_false_reject.png",
        "Verifier Example 3: True match wrongly rejected -- captioning inconsistency, not reasoning failure",
        pano_image_path("Chicago", r3["pano_file"]), r3["pano_fields"],
        sat_image_path("Chicago", r3["sat_file"]), r3["sat_fields"],
        "TRUE MATCH (wrongly rejected)", "#e65100",
        2, "NO_MATCH", ("The ground-level description shows a cul-de-sac with a dead-end and a school "
                        "building, while the satellite view shows a straight road with a central median "
                        "and houses, indicating incompatible road topology and layout."),
        "#e65100",
    )


if __name__ == "__main__":
    main()
