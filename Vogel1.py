import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss

Data = pd.read_csv('Data/Louisville_USGS_Daily_Updated.csv')
# flow rate unit is CFS

# Make date strings into datetime format so we can calculate dates
Data['datetime'] = pd.to_datetime(Data['datetime'])

# find year of each data point
Data['Year'] = Data['datetime'].dt.year

plt.plot(Data['datetime'], Data['flowrate'])
plt.title('Daily streamflow', fontsize=16)
plt.show()

# find the annual maxima in each year
maxQYear = Data.groupby('Year').max()
# Get just the flow vector for annual maxima as numpy array
maxQ = np.array(maxQYear['flowrate'])
# Get just the year vector for annual maxima as numpy array
fig = plt.figure()
yearMaxQ = np.array(maxQYear['datetime'].dt.year)
plt.bar(yearMaxQ, maxQ)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Flow, CFS', fontsize=16)
plt.title('Annual Maxima Flow', fontsize=16)
fig.set_size_inches([9, 7.25])
fig.savefig('Annual Maxima Flow', dpi=300)
fig.clf()

# Exceedance plot
rank = np.arange(1, len(maxQ)+1)
inflow_sorted=np.sort(maxQ)[::-1]
exceedance=(rank/(len(maxQ)+1))*100
fig = plt.figure()
plt.plot(exceedance, inflow_sorted, linewidth=1)
plt.title('Exceedance Plot of Annual Maxima', fontsize=16)
plt.yscale("log")
plt.grid(linewidth=0.5)
plt.xlabel('Exceedance Probability (%)', fontsize=16)
plt.ylabel('Streamflow (CFS)', fontsize=16)
fig.set_size_inches([9, 7.25])
fig.savefig('Exceedance Plot of Annual Maxima.png', dpi=300)
fig.clf()

# Streamflow over threshold
# find only independent peaks
def findClusters(indices):
    allClusters = []
    subCluster = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] - subCluster[-1] <= 5:
            subCluster.append(indices[i])
        else:
            allClusters.append(subCluster)
            subCluster = [indices[i]]

    allClusters.append(subCluster)

    return allClusters

# Updated find peaks function to get dates of peaks
def findPeaks(clusters, data, dataTime):
    peaks = np.zeros(len(clusters))
    dates = []

    for i in range(len(peaks)):
        peaks[i] = np.max(data[clusters[i]])
        ind = np.where(peaks[i] == data[clusters[i]])[0][0]
        dates.append(dataTime[clusters[i][ind]])
    return peaks, dates

# find all peaks over the threshold
x0 = 0.5 * 10 ** 6
peak_indices = np.where(Data['flowrate'] > x0)[0]

clusters = findClusters(peak_indices)
peaks = findPeaks(clusters, Data['flowrate'], Data['datetime'])

threshold = np.zeros(len(Data['datetime']))
for i in range(len(Data['datetime'])):
    for j in range(len(peaks[1])):
        if Data['datetime'][i] == peaks[1][j]:
            threshold[i] = peaks[0][j]
Data['threshold'] = threshold

fig = plt.figure()
plt.plot(Data['datetime'], Data['threshold'])
plt.ylabel('Flow, CFS', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.title('Peaks over 0.5E6 CFS vs Time', fontsize=16)
fig.set_size_inches([9, 7.25])
fig.savefig('Peaks over 0.5E6 CFS vs Time.png', dpi=300)
fig.clf()

# Function to make QQ plot for normal data with correlation test
def NormalPPCT(data, mu, sigma, title, figname):
    x_sorted = np.sort(data)
    p_observed = np.arange(1, len(data) + 1, 1) / (len(data) + 1)
    x_fitted = ss.norm.ppf(p_observed, mu, sigma)
    rho = np.corrcoef(x_sorted, x_fitted)[0, 1]
    plt.scatter(x_sorted, x_fitted, color='b')
    plt.plot(x_sorted, x_sorted, color='r')
    plt.xlabel('Observations')
    plt.ylabel('Fitted Values')
    plt.title(title)
    plt.savefig(figname)
    plt.clf()
    # Estimate p-value of corelation coefficient
    rhoVector = np.zeros(10000)
    for i in range(10000):
        x = ss.norm.rvs(mu, sigma, len(data))
        rhoVector[i] = np.corrcoef(np.sort(x), x_fitted)[0, 1]
    count = 0
    for i in range(len(rhoVector)):
        if rho < rhoVector[i]:
            count = count + 1
    p_value = 1 - count / 10000
    return rho, p_value

def findMoments(data):
    xbar = np.mean(data)
    std = np.std(data, ddof=1)
    skew = ss.skew(data, bias=False)
    return xbar, std, skew

def fitNormal(data, method):
    assert method == 'MLE' or method == 'MOM', "method must = 'MLE' or 'MOM'"
    xbar, std, skew = findMoments(data)
    if method == 'MLE':
        mu, sigma = ss.norm.fit(data)
    elif method == 'MOM':
        mu = xbar
        sigma = std
    return mu, sigma

def LN2_transform(data):
    mu, sigma = fitNormal(np.log(data), 'MOM')
    rho_log, p_value_log = NormalPPCT(np.log(data), mu, sigma, 'Log Normal PPCT', 'LogNormalPPCT.png')
    return rho_log, p_value_log, mu, sigma
