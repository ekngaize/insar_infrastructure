import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd


def create_hexagon(center_x, center_y, r):
    """Crée un hexagone centré en (x, y) avec rayon r."""
    angles = np.linspace(0, 2 * np.pi, 7)
    return Polygon([(center_x + r * np.sin(a), center_y + r * np.cos(a)) for a in angles])


def generate_hex_grid_from_geometry(gdf_geometry, hex_radius, epsg_code):
    # Charger la géométrie depuis un GeoJSON
    gdf = gdf_geometry.to_crs(epsg_code)  # Reprojeter en mètres si on veut un rayon en mètres

    # Obtenir la bounding box
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    x_min, y_min, x_max, y_max = bounds

    w = np.sqrt(3) * hex_radius
    h = 2 * hex_radius
    v_spacing = 0.75 * h

    cols = int((x_max - x_min) / w) + 2
    rows = int((y_max - y_min) / v_spacing) + 2

    hexes = []

    for row in range(rows):
        for col in range(cols):
            cx = x_min + col * w + (row % 2) * (w / 2)
            cy = y_min + row * v_spacing
            hex_poly = create_hexagon(cx, cy, hex_radius)

            # Only keep whole hexagons that intersect the input geometry
            if gdf.union_all().intersects(hex_poly):
                hexes.append(hex_poly)

    hex_gdf = gpd.GeoDataFrame(geometry=hexes, crs=gdf.crs).to_crs(gdf_geometry.crs)

    return hex_gdf