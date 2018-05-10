from PIL import ImageGrab
import win32gui
import time
from pymouse import PyMouse
'''
颜色代号对应关系
    普通 火焰 闪烁
紫色：00  01  02
白色：10  11  12
绿色：20  21  22
黄色：30  31  32
蓝色：40  41  42
红色：50  51  52
橙色：60  61  62
魔方：70
'''


def getjewsimg(box, col, row, lable):
    matriximg = ImageGrab.grab(box)
    imgname = time.time()
    for i in range(len(col)):
        jewimg = matriximg.crop((col[i] * 64, row[i] * 64, col[i] * 64 + 63, row[i] * 64 + 63))
        jewimg.save("jews\\"+lable[i]+"\\"+str(imgname)+str(i)+".jpg", "jpeg")


cols = [2]
rows = [5]
lables = ["70"]
wnd = win32gui.FindWindow(None, "Bejeweled 3")
x1, y1, x2, y2 = win32gui.GetWindowRect(wnd)
box = (x1 * 1.25 + 270, y1 * 1.25 + 75, x2 * 1.25 - 35, y2 * 1.25 - 61)
mousebase = [int(x1*1.25+270), int(y1*1.25+75)]
mouse = PyMouse()
for i in range(100):
    mouse.click(int(mousebase[0] * 0.8) - 20, int(mousebase[1] * 0.8))
    getjewsimg(box, cols, rows, lables)
    time.sleep(0.3)
    if i%10 == 0:
        print(i,"finished")
