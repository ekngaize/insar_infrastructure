import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import itertools


def load_insar_data(filepaths, bbox):
    data = []
    for path in tqdm(filepaths):
        filename = path.name
        df = pd.read_csv(path, usecols=['pid', 'latitude', 'longitude', 'mean_velocity', 'acceleration', 'track_angle', 'temporal_coherence'])
        gdf = parse_insar_data(df, filename, bbox)
        data.append(gdf)

    gdf = gpd.GeoDataFrame(pd.concat(data, axis=0, ignore_index=True), geometry='geometry', crs=4326)
    gdf = remove_duplicated_pid(gdf)

    return gdf


def parse_insar_data(df, filename, bbox):
    df['geometry'] = gpd.points_from_xy(df.longitude, df.latitude)
    df.drop(['longitude', 'latitude'], axis=1, inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
    bbox = bbox.geometry.iloc[0]
    gdf = gdf[gdf.intersects(bbox)]

    gdf['swath'] = filename.split('_')[4]
    gdf['orbite'] = gdf['track_angle'].apply(get_orbite)
    gdf['filename'] = filename

    return gdf


def get_orbite(track_angle):
    if 150 <= track_angle <= 210:
        return 'descending'
    else:
        return 'ascending'


def remove_duplicated_pid(gdf):
    gdf['swath_order'] = gdf['swath'].str[-1]
    gdf = gdf.sort_values(by=['swath_order', 'temporal_coherence'], ascending=[True, False])
    gdf = gdf.drop_duplicates(subset=['pid', 'start_year', 'end_year', 'orbite'], keep='first')
    gdf.drop(['swath_order'], axis=1, inplace=True)

    return gdf


def align_time_series(dfs_list):
    aligned_dfs_index = set([col for df in dfs_list for col in df.columns])
    aligned_dfs_index.remove('index')
    aligned_dfs_index = sorted([pd.to_datetime(col) for col in aligned_dfs_index])
    dfs_reindexed = []

    for df in dfs_list:
        df = df.set_index('index').T
        df.index = pd.to_datetime(df.index)
        df = df.reindex(aligned_dfs_index, fill_value=np.nan)
        dfs_reindexed.append(df.T)

    df_ts = pd.concat(dfs_reindexed, axis=0)

    return df_ts


def resample_and_interpolate(row):
    row.index = pd.to_datetime(row.index)
    row_resampled = row.resample('6D').asfreq()
    raw_interpolated = row_resampled.interpolate(method='linear', limit_area='inside')
    return raw_interpolated


def load_insar_time_series(filepaths, bbox):
    data = []
    time_series = []

    for path in filepaths:
        filename = path.name
        df = pd.read_csv(path)
        date_columns = [col for col in df.columns if col.isdigit() and len(col) == 8]
        columns_to_keep = ['pid', 'latitude', 'longitude', 'track_angle', 'temporal_coherence']
        columns_to_keep.extend(date_columns)
        df = df[columns_to_keep].copy()

        gdf = parse_insar_data(df, filename, bbox)

        df_ts = gdf[date_columns].copy()
        gdf_infos = gdf.drop(columns=date_columns)

        gdf_infos['start_year'] = date_columns[0][:4]
        gdf_infos['end_year'] = date_columns[-1][:4]
        gdf_infos['index'] = gdf_infos['pid'] + gdf_infos['start_year'].astype(str)

        df_ts = df_ts.apply(resample_and_interpolate, axis=1)
        df_ts['index'] = gdf_infos['pid'] + gdf_infos['start_year'].astype(str)

        data.append(gdf_infos)
        time_series.append(df_ts)

    df_ts = align_time_series(time_series)
    df = pd.concat(data, axis=0, ignore_index=True)
    df = df.merge(df_ts.reset_index(), how='inner', on='index')
    df = remove_duplicated_pid(df)

    date_columns = [col for col in df.columns if isinstance(col, pd.Timestamp)]
    df_ts_columns = date_columns.copy()
    df_ts_columns.insert(0, 'start_year')
    df_ts_columns.insert(0, 'pid')
    df_ts = df[df_ts_columns]

    gdf = df.drop(columns=date_columns)
    gdf = gdf.drop(columns='index')
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=4326)

    return gdf, df_ts


def pearson_overlap(s1, s2, min_points=10):
    mask = s1.notna() & s2.notna()
    if mask.sum() < min_points:
        return np.nan

    return pearsonr(s1[mask].astype(float), s2[mask].astype(float))


def pearsonr_by_pid(df):
    results = []
    df = df.set_index('pid')
    for pid, group in df.groupby(level=0):
        if len(group) < 2:
            results.append({
                "pid": pid,
                "pearson_N1": np.nan
            })
            continue

        group_sorted = group.sort_values('start_year')

        for i in range(len(group_sorted) - 1):
            s1 = group_sorted.iloc[i].drop('start_year')
            s2 = group_sorted.iloc[i + 1].drop('start_year')

            r, _ = pearson_overlap(s1, s2, min_points=10)

            results.append({
                "pid": pid,
                "pearson_N1": round(r, 2),
                "start_year": group_sorted.iloc[i]['start_year']
            })

    return pd.DataFrame.from_dict(results)


def combine_series_on_overlap(s1, s2, combine=False):
    # dates communes sans NaN
    mask = s1.notna() & s2.notna()

    delta = s1[mask] - s2[mask]
    delta = delta.mean()

    s2_aligned = s2 + delta

    if not combine:
        return s2_aligned

    s_combined = pd.Series(index=s1.index, dtype=float)
    s_combined[s1.notna() & ~mask] = s1[s1.notna() & ~mask]
    s_combined[s2_aligned.notna()] = s2_aligned[s2_aligned.notna()]

    return s_combined

def align_ts_by_pid(df, combine=False, pearson_threshold=50):
    results = []
    df = df.set_index('pid')

    for pid, group in df.groupby(level=0):
        group_sorted = group.sort_values('start_year')

        if len(group_sorted) == 1:
            s = group_sorted.iloc[0].drop(['pearson_N1', 'start_year'])
            s_infos = group_sorted.iloc[0][['start_year', 'pearson_N1']]
            results.append(pd.concat([s, s_infos]))
            continue

        # First serie
        current_series = group_sorted.iloc[0].drop(['pearson_N1', 'start_year'])
        current_infos = group_sorted.iloc[0][['start_year', 'pearson_N1']]


        results.append(pd.concat([current_series, current_infos]))

        # Propagation
        for idx in range(1, len(group_sorted)):
            next_series = group_sorted.iloc[idx].drop(['pearson_N1', 'start_year'])
            next_infos = group_sorted.iloc[idx][['start_year', 'pearson_N1']]

            if current_infos['pearson_N1'] >= pearson_threshold:
                aligned = combine_series_on_overlap(
                    current_series,
                    next_series,
                    combine=combine
                )
            else:
                aligned = next_series

            results.append(pd.concat([aligned, next_infos]))

            current_series = aligned
            current_infos = next_infos

    df_result = pd.DataFrame(results).reset_index()
    df_result.rename(columns={'index': 'pid'}, inplace=True)
    return df_result
