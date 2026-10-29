from pathlib import Path
import random
import math

import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"C:\Users\saiar\Pictures\cardava_thesis\dataset")

# Stage 1 = cleaned images with black background
STAGE1_DIR = DATA_ROOT / "stage1_cls" / "train"

# Stage 2 = original/raw images
STAGE2_DIR = DATA_ROOT / "stage2_det" / "images" / "train"

# Where previews will be saved
OUTPUT_DIR = DATA_ROOT / "augmentation_previews"

# Number of real dataset images to inspect
SAMPLES_PER_STAGE = 3

# Fixed seed for repeatable image selection
RANDOM_SEED = 42

# Preview dimensions
PREVIEW_WIDTH = 400
PREVIEW_HEIGHT = 280

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FONTS
# ============================================================

def get_font(size):
    possible_fonts = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]

    for font_path in possible_fonts:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


FONT = get_font(22)
SMALL_FONT = get_font(16)


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(path):
    return Image.open(path).convert("RGB")


def get_images(folder):
    if not folder.exists():
        print(f"[WARNING] Folder not found: {folder}")
        return []

    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ])


def choose_samples(folder, number):
    images = get_images(folder)

    if not images:
        return []

    rng = random.Random(RANDOM_SEED)

    if len(images) <= number:
        return images

    return rng.sample(images, number)


def resize_for_preview(image):
    """
    Resize while preserving aspect ratio.
    This is ONLY for displaying the preview.
    """
    image = image.copy()
    image.thumbnail((PREVIEW_WIDTH, PREVIEW_HEIGHT))

    canvas = Image.new(
        "RGB",
        (PREVIEW_WIDTH, PREVIEW_HEIGHT),
        "white"
    )

    x = (PREVIEW_WIDTH - image.width) // 2
    y = (PREVIEW_HEIGHT - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


def add_label(image, label):
    """
    Put a text label underneath the preview.
    """
    label_height = 42

    canvas = Image.new(
        "RGB",
        (PREVIEW_WIDTH, PREVIEW_HEIGHT + label_height),
        "white"
    )

    preview = resize_for_preview(image)

    canvas.paste(preview, (0, 0))

    draw = ImageDraw.Draw(canvas)

    bbox = draw.textbbox(
        (0, 0),
        label,
        font=SMALL_FONT
    )

    text_width = bbox[2] - bbox[0]

    x = (PREVIEW_WIDTH - text_width) // 2

    draw.text(
        (x, PREVIEW_HEIGHT + 8),
        label,
        fill="black",
        font=SMALL_FONT
    )

    return canvas


def make_grid(panels, columns):
    if not panels:
        return None

    cell_width = PREVIEW_WIDTH
    cell_height = PREVIEW_HEIGHT + 42

    rows = math.ceil(len(panels) / columns)

    grid = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height
        ),
        "white"
    )

    for i, panel in enumerate(panels):
        row = i // columns
        col = i % columns

        x = col * cell_width
        y = row * cell_height

        grid.paste(panel, (x, y))

    return grid


def add_filename_header(grid, filename):
    """
    Add the source filename above a preview grid.
    """
    header_height = 40

    result = Image.new(
        "RGB",
        (
            grid.width,
            grid.height + header_height
        ),
        "white"
    )

    result.paste(grid, (0, header_height))

    draw = ImageDraw.Draw(result)

    draw.text(
        (10, 10),
        filename,
        fill="black",
        font=SMALL_FONT
    )

    return result


def stack_grids(grids):
    grids = [g for g in grids if g is not None]

    if not grids:
        return None

    width = max(g.width for g in grids)
    height = sum(g.height for g in grids)

    result = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    y = 0

    for grid in grids:
        result.paste(grid, (0, y))
        y += grid.height

    return result


# ============================================================
# GEOMETRIC AUGMENTATIONS
# ============================================================

def horizontal_flip(image):
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def vertical_flip(image):
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def rotate_image(image, angle):
    """
    Rotate while keeping the original image dimensions.

    The preview is intentionally dimension-preserving so
    we can visually inspect what the rotation does without
    generating an expanded image.
    """
    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=image.getpixel((0, 0))
    )


def translate_image(image, horizontal_percent=0, vertical_percent=0):
    """
    Translate the image by a percentage of width/height.

    Examples:
        +10 horizontal = move right
        -10 horizontal = move left
        +10 vertical   = move down
        -10 vertical   = move up

    The empty region is filled using the top-left pixel.
    """
    width, height = image.size

    dx = int(width * horizontal_percent / 100.0)
    dy = int(height * vertical_percent / 100.0)

    fill_color = image.getpixel((0, 0))

    translated = Image.new(
        "RGB",
        (width, height),
        fill_color
    )

    translated.paste(
        image,
        (dx, dy)
    )

    return translated


# ============================================================
# APPEARANCE AUGMENTATIONS
# ============================================================

def adjust_brightness(image, percent):
    """
    Brightness:
        -20 -> factor 0.80
        -15 -> factor 0.85
        -10 -> factor 0.90
        +10 -> factor 1.10
        +15 -> factor 1.15
        +20 -> factor 1.20
    """
    factor = 1.0 + percent / 100.0

    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image, percent):
    """
    Contrast:
        -20 -> factor 0.80
        -15 -> factor 0.85
        -10 -> factor 0.90
        +10 -> factor 1.10
        +15 -> factor 1.15
        +20 -> factor 1.20
    """
    factor = 1.0 + percent / 100.0

    return ImageEnhance.Contrast(image).enhance(factor)


def add_gaussian_noise(image, sigma, rng):
    """
    Add Gaussian noise.

    sigma is expressed directly on the 0-255 pixel scale.

    sigma = 5
    sigma = 10
    sigma = 15
    sigma = 20
    """
    array = np.asarray(image).astype(np.float32)

    noise = rng.normal(
        0,
        sigma,
        array.shape
    )

    noisy = array + noise

    noisy = np.clip(
        noisy,
        0,
        255
    )

    return Image.fromarray(
        noisy.astype(np.uint8),
        mode="RGB"
    )


# ============================================================
# BRIGHTNESS PREVIEW
# ============================================================

def preview_brightness(samples, output_dir):

    values = [
        -20,
        -15,
        -10,
        0,
        10,
        15,
        20
    ]

    grids = []

    for path in samples:

        image = load_image(path)

        panels = []

        for value in values:

            if value == 0:
                result = image
                label = "Original"
            else:
                result = adjust_brightness(
                    image,
                    value
                )
                label = f"{value:+d}%"

            panels.append(
                add_label(
                    result,
                    label
                )
            )

        grid = make_grid(
            panels,
            columns=7
        )

        grid = add_filename_header(
            grid,
            path.name
        )

        grids.append(grid)

    result = stack_grids(grids)

    if result:
        path = output_dir / "01_brightness_comparison.png"
        result.save(path)
        print(f"[SAVED] {path}")


# ============================================================
# CONTRAST PREVIEW
# ============================================================

def preview_contrast(samples, output_dir):

    values = [
        -20,
        -15,
        -10,
        0,
        10,
        15,
        20
    ]

    grids = []

    for path in samples:

        image = load_image(path)

        panels = []

        for value in values:

            if value == 0:
                result = image
                label = "Original"
            else:
                result = adjust_contrast(
                    image,
                    value
                )
                label = f"{value:+d}%"

            panels.append(
                add_label(
                    result,
                    label
                )
            )

        grid = make_grid(
            panels,
            columns=7
        )

        grid = add_filename_header(
            grid,
            path.name
        )

        grids.append(grid)

    result = stack_grids(grids)

    if result:
        path = output_dir / "02_contrast_comparison.png"
        result.save(path)
        print(f"[SAVED] {path}")


# ============================================================
# GAUSSIAN NOISE PREVIEW
# ============================================================

def preview_noise(samples, output_dir):

    sigma_values = [
        0,
        5,
        10,
        15,
        20
    ]

    grids = []

    for path in samples:

        image = load_image(path)

        panels = []

        for sigma in sigma_values:

            if sigma == 0:
                result = image
                label = "Original"
            else:
                # Use fixed seed for repeatability
                rng = np.random.default_rng(
                    RANDOM_SEED + sigma
                )

                result = add_gaussian_noise(
                    image,
                    sigma,
                    rng
                )

                label = f"σ = {sigma}"

            panels.append(
                add_label(
                    result,
                    label
                )
            )

        grid = make_grid(
            panels,
            columns=5
        )

        grid = add_filename_header(
            grid,
            path.name
        )

        grids.append(grid)

    result = stack_grids(grids)

    if result:
        path = output_dir / "03_gaussian_noise_comparison.png"
        result.save(path)
        print(f"[SAVED] {path}")


# ============================================================
# FLIP PREVIEW
# ============================================================

def preview_flips(samples, output_dir):

    grids = []

    for path in samples:

        image = load_image(path)

        transformations = [
            ("Original", image),
            ("Horizontal flip", horizontal_flip(image)),
            ("Vertical flip", vertical_flip(image)),
            (
                "Horizontal + Vertical",
                vertical_flip(horizontal_flip(image))
            )
        ]

        panels = [
            add_label(result, label)
            for label, result in transformations
        ]

        grid = make_grid(
            panels,
            columns=4
        )

        grid = add_filename_header(
            grid,
            path.name
        )

        grids.append(grid)

    result = stack_grids(grids)

    if result:
        path = output_dir / "04_flipping_comparison.png"
        result.save(path)
        print(f"[SAVED] {path}")


# ============================================================
# ROTATION PREVIEW
# ============================================================

def preview_rotation(samples, output_dir):

    angles = [
        -90,
        -45,
        -30,
        -15,
        0,
        15,
        30,
        45,
        90,
        180
    ]

    grids = []

    for path in samples:

        image = load_image(path)

        panels = []

        for angle in angles:

            if angle == 0:
                result = image
                label = "Original"
            else:
                result = rotate_image(
                    image,
                    angle
                )
                label = f"{angle:+d}°"

            panels.append(
                add_label(
                    result,
                    label
                )
            )

        grid = make_grid(
            panels,
            columns=5
        )

        grid = add_filename_header(
            grid,
            path.name
        )

        grids.append(grid)

    result = stack_grids(grids)

    if result:
        path = output_dir / "05_rotation_comparison.png"
        result.save(path)
        print(f"[SAVED] {path}")


# ============================================================
# TRANSLATION PREVIEW
# ============================================================

def preview_translation(samples, output_dir):

    transformations = [
        ("Original", 0, 0),

        ("Left 5%", -5, 0),
        ("Right 5%", 5, 0),
        ("Up 5%", 0, -5),
        ("Down 5%", 0, 5),

        ("Left 10%", -10, 0),
        ("Right 10%", 10, 0),
        ("Up 10%", 0, -10),
        ("Down 10%", 0, 10),
    ]

    grids = []

    for path in samples:

        image = load_image(path)

        panels = []

        for label, dx, dy in transformations:

            if dx == 0 and dy == 0:
                result = image
            else:
                result = translate_image(
                    image,
                    horizontal_percent=dx,
                    vertical_percent=dy
                )

            panels.append(
                add_label(
                    result,
                    label
                )
            )

        grid = make_grid(
            panels,
            columns=5
        )

        grid = add_filename_header(
            grid,
            path.name
        )

        grids.append(grid)

    result = stack_grids(grids)

    if result:
        path = output_dir / "06_translation_comparison.png"
        result.save(path)
        print(f"[SAVED] {path}")


# ============================================================
# COMBINED APPEARANCE PREVIEW
# ============================================================

def preview_combined_appearance(samples, output_dir):

    configurations = [
        ("Mild\nB10 C10 N5", 10, 10, 5),
        ("Proposed\nB15 C15 N10", 15, 15, 10),
        ("Strong\nB20 C20 N20", 20, 20, 20)
    ]

    grids = []

    for path in samples:

        image = load_image(path)

        panels = [
            add_label(image, "Original")
        ]

        for label, brightness, contrast, sigma in configurations:

            rng = np.random.default_rng(
                RANDOM_SEED + sigma
            )

            result = adjust_brightness(
                image,
                brightness
            )

            result = adjust_contrast(
                result,
                contrast
            )

            result = add_gaussian_noise(
                result,
                sigma,
                rng
            )

            panels.append(
                add_label(
                    result,
                    label
                )
            )

        grid = make_grid(
            panels,
            columns=4
        )

        grid = add_filename_header(
            grid,
            path.name
        )

        grids.append(grid)

    result = stack_grids(grids)

    if result:
        path = output_dir / "07_combined_appearance.png"
        result.save(path)
        print(f"[SAVED] {path}")


# ============================================================
# FULL COMBINED PREVIEW
# ============================================================

def preview_full_combinations(samples, output_dir):

    configurations = [
        (
            "Mild",
            10,     # brightness
            10,     # contrast
            5,      # noise
            15,     # rotation
            5,      # translation X
            5       # translation Y
        ),

        (
            "Proposed",
            15,
            15,
            10,
            30,
            10,
            -10
        ),

        (
            "Strong",
            20,
            20,
            20,
            45,
            -10,
            10
        )
    ]

    grids = []

    for path in samples:

        image = load_image(path)

        panels = [
            add_label(
                image,
                "Original"
            )
        ]

        for (
            label,
            brightness,
            contrast,
            sigma,
            rotation,
            tx,
            ty
        ) in configurations:

            rng = np.random.default_rng(
                RANDOM_SEED + sigma + rotation
            )

            result = adjust_brightness(
                image,
                brightness
            )

            result = adjust_contrast(
                result,
                contrast
            )

            result = add_gaussian_noise(
                result,
                sigma,
                rng
            )

            result = rotate_image(
                result,
                rotation
            )

            result = translate_image(
                result,
                tx,
                ty
            )

            panels.append(
                add_label(
                    result,
                    label
                )
            )

        grid = make_grid(
            panels,
            columns=4
        )

        grid = add_filename_header(
            grid,
            path.name
        )

        grids.append(grid)

    result = stack_grids(grids)

    if result:
        path = output_dir / "08_full_combination_comparison.png"
        result.save(path)
        print(f"[SAVED] {path}")


# ============================================================
# MAIN PROCESS
# ============================================================

def process_stage(stage_name, folder):

    print()
    print("=" * 70)
    print(stage_name)
    print("=" * 70)

    samples = choose_samples(
        folder,
        SAMPLES_PER_STAGE
    )

    if not samples:
        print("[ERROR] No images found.")
        return

    print("\nSelected images:")

    for image in samples:
        print(f"  {image}")

    output_dir = (
        OUTPUT_DIR /
        stage_name.lower().replace(" ", "_")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    preview_brightness(
        samples,
        output_dir
    )

    preview_contrast(
        samples,
        output_dir
    )

    preview_noise(
        samples,
        output_dir
    )

    preview_flips(
        samples,
        output_dir
    )

    preview_rotation(
        samples,
        output_dir
    )

    preview_translation(
        samples,
        output_dir
    )

    preview_combined_appearance(
        samples,
        output_dir
    )

    preview_full_combinations(
        samples,
        output_dir
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    process_stage(
        "Stage 1",
        STAGE1_DIR
    )

    process_stage(
        "Stage 2",
        STAGE2_DIR
    )

    print()
    print("=" * 70)
    print("ALL AUGMENTATION PREVIEWS COMPLETE")
    print("=" * 70)
    print()
    print(f"Preview folder:")
    print(OUTPUT_DIR)
    print()
    print("Your original dataset was NOT modified.")


if __name__ == "__main__":
    main()