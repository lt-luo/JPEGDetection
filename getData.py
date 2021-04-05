import numpy as np
import os

def getData(path):
    for file in os.listdir(path):
        print(file)
        data = np.load(path + file)
    return

path = '.\QF75\\'
getData(path)