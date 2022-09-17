import math
import os
import random
import re
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


if __name__ == '__main__':

    data =  pd.read_csv('trainingdata.txt', header=None, sep =',')
    data.columns = ['charge', 'use']
    data.insert(0, column='bias', value=np.zeros(len(data.charge)))
    data = data[data.use < 8]

    plt.scatter(data.charge, data.use)

    print(data.head())

    X =  data.drop(columns=['use'])
    y = data.use

    reg = LinearRegression().fit(X, y)
    reg.score(X, y)

    input_value = float(input())
    if input_value >= 4:
        print(8.00)
    else:
        input_X = np.array([[0, input_value]])
        print(round(reg.predict(input_X)[0], 2))

