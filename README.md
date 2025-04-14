# Frequency Analysis of Non-Stationary Hydrological Extremes in Northern Ohio

[📄 Read the Full Report (PDF)](CEE266F_FinalProject_Vogel.pdf)

## 📍 Overview

This project analyzes how extreme hydrological events—specifically floods—are changing over time in Northern Ohio, an area increasingly affected by climate change and urbanization. By applying a statistical method introduced by **Vogel et al. (2011)**, we evaluate the **magnification of future flood risks** and the **reduction in return periods** of extreme events, challenging the assumption of stationarity in traditional flood risk analysis.

## 👩‍🔬 Methodology

The analysis involved:

- Identifying streamflow and precipitation stations with statistically significant **positive trends** (based on the Mann-Kendall and OLS regression tests).
- Fitting **LN2 distributions** to annual maxima data and validating them using **KS tests** and **Q-Q plots**.
- Computing:
  - **Flood and Storm Magnification Factors (MF)** over future intervals (10, 20, 30 years)
  - **Reduced Return Periods (RRP)** for design floods (100-year, 200-year, 500-year, 1000-year)
- Comparing results across seven stations (4 streamflow, 3 precipitation) using β/σ ratios.

## 🗺 Study Area

- **Streamflow stations:** Cuyahoga River, Tymochtee Creek, Portage River, Tuscarawas River
- **Precipitation stations:** John Glenn Int'l Airport, Bucyrus, Pandora

All located in **Northern Ohio**, identified based on trend analysis from **USGS** and **NOAA** datasets spanning 56–82 years.

## 📊 Key Findings

- **All stations exhibited significant increasing trends** in annual maxima streamflow or precipitation.
- **Return periods are decreasing** — for instance, a 100-year flood could become a 32-year event in 20 years.
- **Magnification Factors** range from **1.04 to 1.22** depending on location and forecast period.
- **Uncertainty increases with longer return periods**, especially for 500- and 1000-year events.

## 📌 Limitations & Future Work

- Assumes trends only in the **mean**, not standard deviation.
- Does not test alternative distributions (e.g., GEV, Log-Pearson III).
- Does not isolate specific **anthropogenic influences** (e.g., urban development, infrastructure).
- Calls for **geospatial analysis** integrating land use, precipitation, and hydrology.

## 📂 Project Structure

