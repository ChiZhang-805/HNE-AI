"""Render the current manuscript for page-by-page layout inspection."""
from pathlib import Path
import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
document = fitz.open(ROOT / "paper/main.pdf")
output = ROOT / "paper/preview"
output.mkdir(exist_ok=True)
for previous in output.glob("page_[0-9][0-9].png"):
    if int(previous.stem.removeprefix("page_")) > len(document):
        previous.unlink()
thumbnails = []
for index, page in enumerate(document):
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    path = output / f"page_{index + 1:02d}.png"
    pixmap.save(path)
    picture = Image.open(path).convert("RGB")
    picture.thumbnail((306, 396))
    tile = Image.new("RGB", (326, 428), "#eeeeee")
    tile.paste(picture, ((326-picture.width)//2, 20))
    ImageDraw.Draw(tile).text((10, 408), str(index+1), fill="black")
    thumbnails.append(tile)
    print(index+1, len(page.get_text()), page.get_text().splitlines()[-3:])
sheet = Image.new("RGB", (326*3, 428*((len(thumbnails)+2)//3)), "#bbbbbb")
for index, tile in enumerate(thumbnails):
    sheet.paste(tile, ((index % 3)*326, (index//3)*428))
sheet.save(output / "contact_sheet.png")
print(f"Rendered {len(document)} pages to {output}")
