""" 
pymouse是PyUserInput模块的一部分。
需要pip安装pywin32，以及pyhook。
pyhook需要去https://www.lfd.uci.edu/~gohlke/pythonlibs/下载whl文件来安装。
"""

import pyHook
import pythoncom
import sys
from Bejewedled3 import Bejewedled3
import threading


# 游戏执行
def bejewrun():
    bejew = Bejewedled3()
    bejew.run()


# q键退出程序
def keyctrl(event):
    if chr(event.Ascii) == 'q':
        sys.exit(0)
    return True

if __name__ == '__main__':
    # 新建线程自动游戏，并设置为守护线程
    bejewthd = threading.Thread(target=bejewrun)
    bejewthd.setDaemon(True)
    bejewthd.start()

    # 监听键盘
    hm = pyHook.HookManager()
    hm.KeyDown = keyctrl
    hm.HookKeyboard()
    pythoncom.PumpMessages()
