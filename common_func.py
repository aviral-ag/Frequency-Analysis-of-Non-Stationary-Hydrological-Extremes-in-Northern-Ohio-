#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt



def annual_max(data):
    data['datetime'] = pd.to_datetime(data['datetime'])

    # find year of each data point
    data['Year'] = data['datetime'].dt.year

    # find the annual maxima in each year
    maxQYear = data.groupby('Year').max()
    # Get just the flow vector for annual maxima as numpy array
    maxQ = np.array(maxQYear['flowrate'])
    # Get just the year vector for annual maxima as numpy array
    yearMaxQ = np.array(maxQYear['datetime'].dt.year)
    return maxQ, yearMaxQ


def findmoments(data):
    
    xbar = np.mean(data)
    std = np.std(data, ddof=1)
    skew = ss.skew(data,bias=False)
    
    return xbar, std, skew





