import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess

#filename = sys.argv[1]
filename = 'gait1.csv'
gait = pd.read_csv(filename)
columns = gait.columns
print(columns[0])
if 'ay (m/s^2)' in columns:
    gait = gait.rename(columns={"ay (m/s^2)": "ay"})
# print(gait)

#time decision
gait = gait.loc[(gait['time']>10)&(gait['time']<600)]

#butter filter
b, a = signal.butter(3, 0.05, btype='lowpass', analog=False)
gait['ay'] = signal.filtfilt(b, a, gait['ay'])
#plt.plot(gait['time'], gait['ay'], 'b-')
#plt.show()

#gait['prev'] = gait['gFy'].shift(periods=1)
gait['next'] = gait['ay'].shift(periods=-1)
gait = gait[['time','ay','next']]

#gaitmax = gait.loc[(gait['gFy']>gait['prev'])&(gait['gFy']>gait['next'])]
#gaitmin = gait.loc[(gait['gFy']<gait['prev'])&(gait['gFy']<gait['next'])]
#gaitmax = gaitmax.loc[(gaitmax['time']>60)&(gaitmax['time']<280)]
#gaitmin = gaitmin.loc[(gaitmin['time']>60)&(gaitmin['time']<280)]
#gaitmax['ntime'] = gaitmax['time'].shift(periods=-1)
#gaitmin['ntime'] = gaitmin['time'].shift(periods=-1)


gaitleft = gait.loc[((gait['ay']<0)&(gait['next']>0))]
gaitright =  gait.loc[((gait['ay']>0)&(gait['next']<0))]
gaitleft = gaitleft.reset_index()
gaitright = gaitright.reset_index()

#print(gaitleft)
#print(gaitright)


gaitright['timeN'] = gaitright['time'].shift(periods=-1)
gaitleft['timeN'] = gaitleft['time'].shift(periods=-1)

#ave time of one step = 0.42sec
#ave speed of walking = 178.8 cm/s
#ave length of one step = 76.2 cm


if gaitleft['time'][0] > gaitright['time'][0]:
    steptimeR = gaitleft['time'] - gaitright['time']
    steptimeL = gaitright['timeN'] - gaitleft['time']
else:
    steptimeL = gaitright['time'] - gaitleft['time']
    steptimeR = gaitleft['timeN'] - gaitright['time']

steptimeR.dropna(inplace=True)
steptimeL.dropna(inplace=True)

result = pd.concat([steptimeR,steptimeL], axis=1, sort=False, ignore_index=False)
result = result.rename(columns={"time": "right", 0: "left"})

#result = result.loc[((result['right']<0.42*1.5)&(result['right']>0.42*0.5))|((result['left']<0.42*1.5)&(result['left']>0.42*0.5))]

print(result['right'].mean())
print(result['left'].mean())
print(gait)

#print(result)
