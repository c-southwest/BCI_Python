import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

# 希望的刺激频率
Fequency = 12
Length = 20

#颜色定义
Black = [0,0,0]
White = [255,255,255]

#刺激所用颜色
Color1 = Black
Color2 = White

# 生成图片
image1 = Image.new('RGB', (1920, 1080), tuple(Color1))
image1.save("img1.png")
image1 = Image.new('RGB', (1920, 1080), tuple(Color2))
image1.save("img2.png")

command = f"ffmpeg -y -loop 1 -f image2 -r {Fequency*2} -i img%d.png -r 60 -t {Length} {Fequency}Hz.mp4"
os.system(command)
