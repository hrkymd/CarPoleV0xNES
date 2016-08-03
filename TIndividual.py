import numpy as np


class TIndividual :
    def __init__(self,n): #コンストラクタ
        # self.z = np.zeros(n)
        # self.theta = np.zeros(n) #theta_jとなる
        self.r_total = 0.0

    def setZ(self, z):
        self.z = z

    def setTheta(self, theta):
        self.theta = theta

    def initr_total(self):
        self.r_total = 0.0

    def addr_total(self, r):
        self.r_total += r



