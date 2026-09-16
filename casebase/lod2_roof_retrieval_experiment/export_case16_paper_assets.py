"""Export text-free paper-figure assets from the saved case-16 comparison."""

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


EXPERIMENT_DIR = Path(__file__).resolve().parent
QUERY_ID = "DE_BE_008010"
SOURCE_PATH = (
    EXPERIMENT_DIR
    / "outputs"
    / "multicity"
    / "global_queries"
    / "figures"
    / "all_queries_ranked_axonometric"
    / "16_steep_DE_BE_008010_axonometric.png"
)
OUTPUT_DIR = (
    EXPERIMENT_DIR
    / "outputs"
    / "paper_figure_assets"
    / "case_16_DE_BE_008010"
)

# Interior regions of the already-rendered panels. Titles, ranks and metric
# annotations lie outside these boxes. Coordinates refer to the 4041 x 1911
# source image produced by notebook 04.
QUERY_BOX = (36, 790, 898, 1470)
M1_BOXES = (
    (988, 407, 1526, 820),
    (1608, 407, 2147, 820),
    (2228, 407, 2768, 820),
    (2849, 407, 3389, 820),
    (3470, 407, 4005, 820),
)
M2_BOXES = tuple((left, 1440, right, 1850) for left, _, right, _ in M1_BOXES)


def _trim_and_fit(image: Image.Image, size: tuple[int, int] = (700, 520)) -> Image.Image:
    """Trim white margins and fit a model onto a consistent white canvas."""

    rgb = image.convert("RGB")
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    mask = difference.convert("L").point(lambda value: 255 if value > 8 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("裁切区域中没有检测到模型。")
    model = rgb.crop(bbox)
    model.thumbnail((size[0] - 44, size[1] - 36), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - model.width) // 2
    y = (size[1] - model.height) // 2
    canvas.paste(model, (x, y))
    return canvas


def _extract_panels(source: Image.Image) -> tuple[Image.Image, list[Image.Image], list[Image.Image]]:
    query_crop = source.crop(QUERY_BOX)
    # Remove the vertical orange row label overlaid inside the query panel.
    query_draw = ImageDraw.Draw(query_crop)
    query_draw.rectangle((658, 460, 733, 680), fill="white")
    query = _trim_and_fit(query_crop)
    m1 = [_trim_and_fit(source.crop(box)) for box in M1_BOXES]
    m2 = [_trim_and_fit(source.crop(box)) for box in M2_BOXES]
    return query, m1, m2


def _make_contact_sheet(
    query: Image.Image,
    m1: list[Image.Image],
    m2: list[Image.Image],
) -> Image.Image:
    tiles = [[query, *m1], [query, *m2]]
    gap_x, gap_y = 32, 36
    tile_w, tile_h = query.size
    sheet = Image.new(
        "RGB",
        (6 * tile_w + 5 * gap_x, 2 * tile_h + gap_y),
        "white",
    )
    for row_index, row in enumerate(tiles):
        for column_index, tile in enumerate(row):
            x = column_index * (tile_w + gap_x)
            y = row_index * (tile_h + gap_y)
            sheet.paste(tile, (x, y))
    return sheet


def main() -> None:
    source = Image.open(SOURCE_PATH).convert("RGB")
    if source.size != (4041, 1911):
        raise ValueError(f"源图尺寸应为 4041 x 1911，实际为 {source.size}。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_dir = OUTPUT_DIR / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    query, m1, m2 = _extract_panels(source)

    query.save(panel_dir / "query_DE_BE_008010.png")
    for index, panel in enumerate(m1, start=1):
        panel.save(panel_dir / f"m1_{index:02d}.png")
    for index, panel in enumerate(m2, start=1):
        panel.save(panel_dir / f"m2_{index:02d}.png")

    output_path = OUTPUT_DIR / f"{QUERY_ID}_axonometric_no_text.png"
    _make_contact_sheet(query, m1, m2).save(output_path, dpi=(300, 300))
    print(output_path)


if __name__ == "__main__":
    main()
