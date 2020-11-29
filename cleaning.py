import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats


#filename = sys.argv[1]
filename = 'fake.csv'
output = filename[0:-4] + 'result.csv' 
gait = pd.read_csv(filename)
columns = gait.columns
if 'ay (m/s^2)' in columns:
    gait = gait.rename(columns={"ay (m/s^2)": "ay"})



#time decision
leng = len(gait['time'])
cut = int(leng*0.1)
gait = gait.loc[(gait['time']>gait['time'].values[cut])&(gait['time']<gait['time'].values[leng-cut])]

#butter filter
b, a = signal.butter(3, 0.05, btype='lowpass', analog=False)
gait['ay'] = signal.filtfilt(b, a, gait['ay'])
#plt.plot(gait['time'], gait['ay'], 'b-')
#plt.show()



gait['prev'] = gait['ay'].shift(periods=1)
gait['next'] = gait['ay'].shift(periods=-1)
gait['timeN'] = gait['time'].shift(periods=-1)
gait = gait[['time','ay','next', 'timeN', 'prev']]
gait.dropna(inplace=True)

#gaitmax = gait.loc[(gait['ay']>gait['prev'])&(gait['ay']>gait['next'])]
#gaitmin = gait.loc[(gait['gFy']<gait['prev'])&(gait['gFy']<gait['next'])]
#gaitmax['ntime'] = gaitmax['time'].shift(periods=-1)
#gaitmin['ntime'] = gaitmin['time'].shift(periods=-1)

#gait['max'] = np.where(((gait['ay']>gait['prev']) & (gait['ay']>gait['next'])), 1, 0)

gaitleft = gait.loc[((gait['ay']<0)&(gait['next']>0))]
gaitright =  gait.loc[((gait['ay']>0)&(gait['next']<0))]
gaitleft = gaitleft.reset_index()
gaitright = gaitright.reset_index()

#print(gaitleft)
#print(gaitright)




gaitright['timeN'] = gaitright['time'].shift(periods=-1)
gaitleft['timeN'] = gaitleft['time'].shift(periods=-1)



#ave speed of walking = 178.8 cm/s https://www.healthline.com/health/exercise-fitness/average-walking-speed#:~:text=What%20Is%20the%20Average%20Walking%20Speed%20of%20an%20Adult%3F&text=The%20average%20walking%20speed%20of%20a%20human%20is%203%20to,age%2C%20sex%2C%20and%20height.
#ave length of one step = 76.2 cm https://www.scientificamerican.com/article/bring-science-home-estimating-height-walk/#:~:text=On%20average%2C%20adults%20have%20a,from%20about%200.41%20to%200.45).
#ave time of one step = 0.42sec calculated from above


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

print(filename)
print("ave a right step(sec): ",result['right'].mean())
print("ave a left step(sec): ",result['left'].mean())

result.to_csv(output, index=False)

## Khoa ttest
# plt.plot(result.index, result['right'], 'b-', label = "right step")
# plt.plot(result.index, result['left'], 'r-', label = "left step")
plt.hist([result['right'], result['left']], bins=50, label=['right', 'left'])
plt.legend(loc='upper right')
plt.show()

## Check for normal distribution and equal variance. If p > 0.05 then it's valid to do t-test
print()

#power exp log sqrt
result['right2'] = np.log(result['right'])
result['left2'] = np.log(result['left'])

result.dropna(inplace=True)

print("test normality right: ",stats.normaltest(result['right2']).pvalue)
print("test normality left: ",stats.normaltest(result['left2']).pvalue)
print("test variance: ",stats.levene(result['right2'], result['left2']).pvalue)


## Compute T-test. Null hypothese: right step and left step have the same step time. If p < 0.05, we reject the null hypothesis
ttest = stats.ttest_ind(result['right2'], result['left2'])
print()
print(ttest)
print()
# print(ttest.statistic)
# print(ttest.pvalue)


#pace test
#distance calculation
gait['speed'] = gait['ay'] * (gait['timeN']-gait['time'])
gait['speedP'] = gait['speed'].shift(periods=1)
gait.dropna(inplace=True)
#gait['distance(cm)'] = gait['speed'] * (gait['timeN']-gait['time'])
gait['distance(cm)'] = gait['speed']**2 - gait['speedP']**2 / (gait['ay']*2)
gait['distance(cm)'] = gait['distance(cm)'] * 100
gaittest = gait.loc[(gait['distance(cm)']>0)]
distanceC = gaittest['distance(cm)'].values.sum()

#time taken calculation
timetaken = gait['time'].values[len(gait['time'])-1] - gait['time'].values[0]

#number of step calculation
result['count'] = 1
step = result['count'].values.sum()


print("steps: ",step)
print("time: ",timetaken)
print("distance: ",distanceC)

distance = 10000
print("pace steps/time(sec): ", step/timetaken)
#for fixed distance
#print("pace steps/distance(m): ", step/distance)
#for calculated distance(device must be placed on foot)
print("pace steps/distance(cm): ", step/distanceC)
#unit is cm
