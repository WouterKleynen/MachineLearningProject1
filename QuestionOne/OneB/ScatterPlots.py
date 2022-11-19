import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSetFilePath = 'Dataset/InputData.csv'                                                           # Set data File path to that of the assignment data sheet.

data = pd.read_csv(dataSetFilePath).to_numpy()
dataWithoutIDMatrix = data[:, 1:]

id = data[:, 0]                                               
Balance = data[:, 1]                                                
QualMiles = data[:, 2]                                                
cc1Miles = data[:, 3]
cc2Miles = data[:, 4]
cc3Miles = data[:, 5]
BonusMiles = data[:, 6]
BonusTrans = data[:, 7]
FlightMiles12Mo = data[:, 8]
FlightTrans12 = data[:, 9]
DaysSinceEnrolled = data[:, 10]
Award = data[:,11]

plt.scatter(Balance, DaysSinceEnrolled)
plt.show()
