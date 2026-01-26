import pandas as pd
import geopandas as gpd


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