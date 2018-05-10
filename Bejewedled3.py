""" 
pymouse是PyUserInput模块的一部分。
需要pip安装pywin32，以及pyhook。
pyhook需要去https://www.lfd.uci.edu/~gohlke/pythonlibs/下载whl文件来安装。
"""
from pymouse import PyMouse
from PIL import ImageGrab
import win32gui
import time
import tensorflow as tf
import numpy as np


class Bejewedled3(object):
    """
    宝石迷阵类
    wnd 窗口句柄
    box 游戏区域
    mousebase 游戏区域左上角
    mouse 鼠标
    matriximg 游戏区域截图
    jewmatrix 宝石方阵
    sess 宝石识别模型对话

    """
    def __init__(self):
        self.wnd = win32gui.FindWindow(None, "Bejeweled 3")
        x1, y1, x2, y2 = win32gui.GetWindowRect(self.wnd)
        self.box = (x1*1.25+270, y1*1.25+75, x2*1.25-35, y2*1.25-61)
        self.mousebase = [int(x1*1.25+270), int(y1*1.25+75)]
        self.mouse = PyMouse()
        modelsaver = tf.train.import_meta_graph(".\\model\\jewsmodel.ckpt.meta")
        self.sess = tf.Session()
        modelsaver.restore(self.sess, ".\\model\\jewsmodel.ckpt")
        self.goon = True
        self.jewmatrix = list()
        self.matriximg = ImageGrab.grab(self.box)
        print("init finished")

    def __del__(self):
        self.sess.close()

    # 将宝石数字矩阵转换为颜色矩阵并输出
    def showclrmatrix(self):
        clrmatrix = list()
        clrlist = ["紫", "白", "绿", "黄", "蓝", "红", "橙", "黑"]
        for row in range(8):
            clrrow = list()
            for col in range(8):
                clrrow.append(clrlist[self.jewmatrix[col][row]])
            clrmatrix.append(clrrow)
        print("宝石阵：")
        for i in range(len(clrmatrix)):
            print(clrmatrix[i])

    # 交换操作
    def swapjew(self, x1, y1, x2, y2):
        time.sleep(0.3)
        self.mouse.click(int((x1*64+31+self.mousebase[0])*0.8), int((y1*64+31+self.mousebase[1])*0.8))
        time.sleep(0.3)
        self.mouse.click(int((x2*64+31+self.mousebase[0])*0.8), int((y2*64+31+self.mousebase[1])*0.8))

    # 截屏，获得排列矩阵#
    def makejewmatrix(self):
        self.matriximg = ImageGrab.grab(self.box)
        # self.matriximg.save("jewimg\\jewmatrix\\"+str(time.time())+".jpg", "jpeg")
        self.jewmatrix = list()
        for col in range(8):
            jewcol = list()
            for row in range(8):
                jewcol.append(self.whichjew(self.matriximg.crop((col*64, row*64, col*64+63, row*64+63))))
            self.jewmatrix.append(jewcol)

    # 判断某个坐标处是什么宝石
    def whichjew(self, jewimg):
        """获取3*3方框平均颜色"""
        jewimg = jewimg.crop((20, 20, 44, 44))
        imgdata = np.array(list(jewimg.getdata())).T
        imgdata.reshape(-1, )
        res = self.sess.run(tf.get_default_graph().get_tensor_by_name("prediction:0"),
                            feed_dict={tf.get_default_graph().get_tensor_by_name("x:0"): imgdata.reshape((-1, 3, 576)),
                                       tf.get_default_graph().get_tensor_by_name("keep_prob:0"): 1})
        return np.argmax(res, 1)[0]

    # 产生可行操作
    def makeactions(self):
        def isblack(mtx, x, y):
            if mtx[x][y] == 7:
                return True
            else:
                return False

        def isysame(mtx, x, y1, y2, y3):
            if y1 < 0 or y1 > 7 or y2 < 0 or y2 > 7 or y3 < 0 or y3 > 7:
                return False
            elif mtx[x][y1] == mtx[x][y2] and mtx[x][y1] == mtx[x][y3]:
                return True
            else:
                return False

        def isxsame(mtx, y, x1, x2, x3):
            if x1 < 0 or x1 > 7 or x2 < 0 or x2 > 7 or x3 < 0 or x3 > 7:
                return False
            elif mtx[x1][y] == mtx[x2][y] and mtx[x1][y] == mtx[x3][y]:
                return True
            else:
                return False

        def tryswap(mtx, x1, y1, x2, y2):
            if x2 < 0 or x2 > 7 or y2 < 0 or y2 > 7:
                return False
            else:
                mtx[x1][y1], mtx[x2][y2] = mtx[x2][y2], mtx[x1][y1]
            if isblack(mtx, x1, y1) or isblack(mtx, x2, y2) \
                    or isysame(mtx, x2, y2-2, y2-1, y2) or isysame(mtx, x2, y2-1, y2, y2+1) \
                    or isysame(mtx, x2, y2, y2+1, y2+2) \
                    or isxsame(mtx, y2, x2-2, x2-1, x2) or isxsame(mtx, y2, x2-1, x2, x2+1) \
                    or isxsame(mtx, y2, x2, x2+1, x2+2) \
                    or isysame(mtx, x1, y1-2, y1-1, y1) or isysame(mtx, x1, y1-1, y1, y1+1) \
                    or isysame(mtx, x1, y1, y1+1, y1+2) \
                    or isxsame(mtx, y1, x1-2, x1-1, x1) or isxsame(mtx, y1, x1-1, x1, x1+1) \
                    or isxsame(mtx, y1, x1, x1+1, x1+2):
                mtx[x1][y1], mtx[x2][y2] = mtx[x2][y2], mtx[x1][y1]
                return True
            else:
                mtx[x1][y1], mtx[x2][y2] = mtx[x2][y2], mtx[x1][y1]
                return False

        matrixcopy = self.jewmatrix.copy()
        ans = False
        for row in range(8):
            for col in range(8):
                if tryswap(matrixcopy, col, row, col+1, row):
                    self.swapjew(col, row, col+1, row)
                    ans = True
                    break
                elif tryswap(matrixcopy, col, row, col, row+1):
                    self.swapjew(col, row, col, row+1)
                    ans = True
                    break
        if not ans:
            print("无解")
            # self.goon = False

    # 开始破解迷阵--ZEN模式
    def run(self):
        while self.goon:
            self.mouse.click(int(self.mousebase[0]*0.8)-20, int(self.mousebase[1]*0.8))
            self.makejewmatrix()
            self.showclrmatrix()
            self.makeactions()
            self.mouse.move(int(self.mousebase[0]*0.8)-20, int(self.mousebase[1]*0.8))
            time.sleep(3)
