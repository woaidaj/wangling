from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import (QImage, QPixmap)
from actconcard import actionControlCard as acc



'''
***************************************************
                    电机运动线程
初始化：传进运动控制卡类
功能：控制电机运动制定的距离。电机运动结束，该线程就结束
***************************************************
'''


class MoveThread(QThread):
    moveEnd = QtCore.pyqtSignal()     # 发出的继续图像处理线程的信号
    msgBoxSend = QtCore.pyqtSignal(str)         # 发出的QMessagebox提示信息

    def __init__(self):
        super().__init__()
        self.hcard = None               # 控制卡实例
        self.maxPulse = 9000            # 物镜最大景深
        self.depth = 0                  # 物镜当前景深（用于承接运控卡编码器脉冲数，设置滑块数据）
        self.distance = 0               # 需要移动的距离（现实距离，不是脉冲数）
        self.pulseEquivalent = 10       # 脉冲当量

    def check(self):
        '''检查是否可以执行运动任务'''
        # 获取电机运动状态，检查电机是否运动
        self.hcard.getInputBit()
        if self.hcard.inValue.value == 0:
            # 电机正在运动，不能执行该线程
            return False
        if self.distance < 0:
            # 向下运动
            if self.depth < abs(self.distance):
                self.msgBoxSend.emit("会运动超过下限，请重新设置")
                return False
            else:
                return True
        else:
            # 向上运动
            if self.depth + self.distance > self.maxPulse:
                self.msgBoxSend.emit("会运动超过上限，请重新设置")
                return False
            else:
                return True

    def run(self):
        if self.check() is True:
            # 设置运动距离
            self.hcard.step = self.distance * self.pulseEquivalent
            # 启动运动
            self.hcard.dMove()
            # 检测电机定位完成
            self.hcard.getInputBit()
            while self.hcard.inValue.value == 0:
                # 循环条件根据电机设置来
                self.hcard.getInputBit()
        self.moveEnd.emit()



if __name__ == '__main__':
    t = MoveThread()
    t.hcard = acc.open(1)
    t.hcard.vel = 10
    t.hcard.acc = 5
    t.hcard.dec = 5
    # pos = c_double(0)
    # t.hcard.getPrfPos2(pos)
    # print(pos.value)
    t.hcard.getPrfPos()
    print(t.hcard.pos.value)
    t.hcard.dMove()
    t.setSliderLens()
    # t.vMove()
    # time.sleep(5)
    t.hcard.close()

