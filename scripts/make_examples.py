"""Create small, dependency-free demo inputs for the README examples."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1] / "examples"
ROOT.mkdir(exist_ok=True)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (640, 640), (249, 249, 247))
    return image, ImageDraw.Draw(image)


def mug() -> Image.Image:
    image, draw = canvas()
    draw.rounded_rectangle((215, 170, 445, 485), radius=42, fill=(37, 91, 120), outline=(24, 53, 72), width=8)
    draw.ellipse((390, 245, 518, 375), outline=(24, 53, 72), width=24)
    draw.ellipse((233, 135, 427, 205), fill=(218, 224, 219), outline=(24, 53, 72), width=8)
    draw.arc((280, 225, 380, 400), 92, 265, fill=(191, 224, 233), width=8)
    draw.text((275, 433), "BREW", fill=(233, 241, 239))
    return image


def serum() -> Image.Image:
    image, draw = canvas()
    draw.rounded_rectangle((250, 185, 390, 510), radius=20, fill=(215, 167, 89), outline=(89, 65, 35), width=7)
    draw.rectangle((276, 142, 364, 200), fill=(199, 200, 193), outline=(89, 65, 35), width=7)
    draw.rectangle((265, 306, 375, 396), fill=(242, 232, 195), outline=(125, 100, 64), width=4)
    draw.text((287, 332), "GLOW", fill=(76, 67, 48))
    draw.text((282, 359), "SERUM", fill=(76, 67, 48))
    return image


def headphones() -> Image.Image:
    image, draw = canvas()
    draw.arc((160, 125, 480, 465), 180, 360, fill=(35, 40, 48), width=35)
    draw.rounded_rectangle((145, 330, 245, 485), radius=35, fill=(53, 62, 74), outline=(21, 25, 31), width=8)
    draw.rounded_rectangle((395, 330, 495, 485), radius=35, fill=(53, 62, 74), outline=(21, 25, 31), width=8)
    draw.ellipse((166, 355, 224, 442), fill=(143, 159, 175))
    draw.ellipse((416, 355, 474, 442), fill=(143, 159, 175))
    return image


for name, factory in (("mug", mug), ("serum", serum), ("headphones", headphones)):
    factory().save(ROOT / f"{name}.input.png")
