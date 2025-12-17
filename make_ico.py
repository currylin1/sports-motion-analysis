# make_ico.py
from PIL import Image
img = Image.open("assets/gui.png")
img.save("assets/gui.ico",
         sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
