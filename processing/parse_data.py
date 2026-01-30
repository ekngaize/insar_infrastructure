import pandas as pd
import geopandas as gpd
import numpy as np


def load_insar_data(filepaths, bbox):
    data = []
    for path in filepaths:
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
    df['swath'] = filename.split('_')[4]
    df['start_year'] = int(filename.split('_')[6])
    df['end_year'] = int(filename.split('_')[7])
    df['orbite'] = df['track_angle'].apply(get_orbite)
    df['filename'] = filename
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
    bbox = bbox.geometry.iloc[0]
    gdf = gdf[gdf.intersects(bbox)]

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


def load_insar_time_series(filepaths, bbox):
    data = []
    time_series = []

    for path in filepaths:
        filename = path.name
        df = pd.read_csv(path)
        df_infos = df[['pid', 'latitude', 'longitude', 'track_angle', 'temporal_coherence']].copy()
        date_columns = [col for col in df.columns if col.isdigit() and len(col) == 8]
        df_ts = df[date_columns].copy()

        gdf = parse_insar_data(df_infos, filename, bbox)
        gdf['index'] = gdf['pid'] + gdf['start_year'].astype(str)
        df_ts['index'] = gdf['pid'] + gdf['start_year'].astype(str)

        data.append(gdf)
        time_series.append(df_ts)

    df_ts = align_time_series(time_series)
    df = pd.concat(data, axis=0, ignore_index=True)
    df = df.merge(df_ts, how='inner', on='index')
    df = remove_duplicated_pid(df)

    date_columns = [col for col in df.columns if isinstance(col, pd.Timestamp)]
    df_ts_columns = date_columns.copy()
    df_ts_columns.insert(0, 'start_year')
    df_ts_columns.insert(0, 'pid')
    df_ts = df[df_ts_columns]

    gdf = df.drop(columns=date_columns)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)

    return gdf, df_ts