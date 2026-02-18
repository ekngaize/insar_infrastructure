import pandas as pd
import numpy as np

def get_velocity_and_displacement(row):
    ts = row.drop(['pid', 'start_year', 'pearson_N1'])
    ts = ts.dropna()
    ts = pd.Series(ts.values, index=pd.to_datetime(ts.index))
    velocity, const = np.polyfit(ts.index.map(pd.Timestamp.toordinal), ts.astype(float).values, 1)
    y_trend = velocity * ts.index.map(pd.Timestamp.toordinal) + const
    total_displacement = y_trend[-1] - y_trend[0]

    return velocity / 6 * 365, total_displacement


def get_velocity_variability(row):
    ts = row.drop(['pid', 'start_year', 'pearson_N1'])
    ts = ts.dropna()
    ts = pd.Series(ts.values, index=pd.to_datetime(ts.index))
    ts = ts.rolling(5).mean().dropna()

    dt = int(365.25 / 6)
    velocity_2y = ts.rolling(window=dt).apply(
        lambda x: (x[-1] - x[0])) # mm/yr

    return velocity_2y.std()


def get_acceleration(row):
    ts = row.drop(['pid', 'start_year', 'pearson_N1'])
    ts = ts.dropna()
    ts = pd.Series(ts.values, index=pd.to_datetime(ts.index))
    ts = ts.rolling(5).mean().dropna()

    dt_years = 6 / 365.25
    acceleration = ts.diff() / dt_years
    return acceleration