from PIL import Image, ImageDraw

def create_frame(text):
    img = Image.new("RGB", (1280, 720), "black")
    draw = ImageDraw.Draw(img)
    draw.text((100, 300), text, fill="white")
    return img
