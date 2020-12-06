import sys
import pandas as pd
import numpy as np
from scipy import signal

def filter(gait):
    data = gait.copy()

    columns = data.columns

    if 'ay (m/s^2)' in columns:
        data = data.rename(columns={"ay (m/s^2)": "ay"})

    #time decision
    leng = len(gait['time'])
    cut = int(leng*0.2)
    data = data.loc[(gait['time']>gait['time'].values[cut])&(gait['time']<gait['time'].values[leng-cut])]


    #butter filter
    b, a = signal.butter(3, 0.05, btype='lowpass', analog=False)
    data['ay'] = signal.filtfilt(b, a, data['ay'])
    return data

def cleaning(data, result):
    gait = data.copy()

    gait['timeN'] = gait['time'].shift(periods=-1)
    gait['gap'] = gait['timeN']-gait['time']
    gait = gait[['time','ay','timeN', 'gap']]
    gait.dropna(inplace=True)
    
    gait['error']= gait['gap']>1
    gaitgap = gait.loc[gait['gap']>1]
    gaitgap = gaitgap.sum()
    remove = gaitgap['gap']
    gait = gait.loc[gait['error']==False]
    
    #distance calculation
    gait['speed'] = gait['ay'] * (gait['timeN']-gait['time'])
    gait['speedP'] = gait['speed'].shift(periods=1)
    gait.dropna(inplace=True)
    gait['distance(cm)'] = gait['speed']**2 - gait['speedP']**2 / (gait['ay']*2)
    gait['distance(cm)'] = gait['distance(cm)'] * 100
    distance = gait['distance(cm)'].values.sum()

    #time taken calculation
    timetaken = (gait['time'].values[len(gait['time'])-1] - gait['time'].values[0]) - remove

    #number of step calculation
    step = result['right'].count()


    print("steps: ",step)
    print("time: ",timetaken)
    print("distance: ",distance)

    fixed_distance = 10000
    print("pace steps/time(sec): ", step/timetaken)
    #for fixed distance
    #print("pace steps/distance(m): ", step/fixed_distance)
    #for calculated distance(device must be placed on foot)
    print("pace steps/distance(cm): ", step/distance)
    #unit is cm
    print("step length(cm): ", distance/step)

if __name__ == "__main__":
    filename = 'soo1.csv'
    output = 'soo1result.csv'
    data = pd.read_csv(filename)
    result = pd.read_csv(output)

    gait = filter(data)
    cleaned_data = cleaning(gait,result)
