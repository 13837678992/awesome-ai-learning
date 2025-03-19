from PIL import Image
import os

# 手动设置工作目录为项目根目录
# os.chdir('C:/Users/weicheng/study/awesome-ai-learning')

img = Image.open("learn_pytorch/dataSet/train/ants/0013035.jpg")
print(img)