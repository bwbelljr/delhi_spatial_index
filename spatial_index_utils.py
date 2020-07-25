# Import necessary modules
from itertools import islice
import pickle
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.ops import cascaded_union
from pyproj import CRS

def reproject_gdf(gdf, epsg_code):
    """Reprojects GeoDataFrame to CRS with EPSG code

    Assigns WKT format of projection to EPSG code to GeoDataFrame.

    Args:
    gdf: GeoDataFrame with any geometry (e.g., Point, Line, Polygon)
    epsg_code: EPSG code (integer)

    Returns:
        GeoDataFrame reprojected to new crs (based on EPSG code).
    """
    # Define CRS in WKT format using EPSG code
    target_projection = CRS.from_epsg(epsg_code).to_wkt()

    # Reproject GeoDataFrame to epsg_code
    reprojected_gdf = gdf.to_crs(target_projection)

    # Print message
    print("GeoDataFrame now has the following CRS:\n")
    print(reprojected_gdf.crs)

    return reprojected_gdf

def plot_polygons_and_points(polygon1=None, polygon2=None, points=None):
    """Plots up to two polygon layers and an additional point layer

        Uses matplotlib to plot 1-2 polygon layers and 1 point layer.
        Each layer is optional and the function checks that the GeoDataFrame
        has data before attempting to plot. By default, the first polygon layer
        is plotted in gray, the second polygon layer in red, and the point
        layer in blue. Code adapted from:
        https://automating-gis-processes.github.io

    Args:
        polygon1: GeoDataFrame with polygon geometries
        polygon2: GeoDataFrame with polygon geometries
        points: GeoDataFrame with point geometries

    Returns:
        None
    """

    # Create a figure with one subplot
    fig, ax = plt.subplots(figsize=(14,9))

    # Plot polygon(s)
    if polygon1 is not None:
        polygon1.plot(ax=ax, facecolor='gray')
    if polygon2 is not None:
        polygon2.plot(ax=ax, facecolor='red')

    # Plot points
    if points is not None:
        points.plot(ax=ax, color='blue', markersize=1)

    plt.tight_layout()

def add_polygon_neighbors_column_refugee(polygon_gdf,
    neighbor_colname = "polygon_neighbors", neighbor_id_col='OBJECTID'):
    """In polygon GeoDataframe, add column with neighboring polygons indices

    Finds all the neighboring polygons in a polygon GeoDataFrame based on
    whether polygons intersect bounding box of selected Polygon and adds a new
    column with a list of indices of these neighboring polygons.

    Args:
        polygon_gdf: geopandas GeoDataFrame with Polygon geometry
        neighbor_colname: name of new column that will have the list of indices
            of neighboring polygons. By default, set to 'polygon_neighbors'
        neighbor_id_col: name of the column used as the identifier (or unique
            key) for neighboring polygons. By default, set to 'OBJECTID'

    Returns:
        polygon_with_neighbors_gdf: geopandas GeoDataFrame having an additional
            column with list of indices of neighboring polygons
    """
    # Make copy of polygon_gdf
    polygon_with_neighbors_gdf = polygon_gdf.copy()

    # Store number of rows
    num_rows = len(polygon_gdf)

    # Create new column with column name as neighbor_colname
    # Each value in the new column is set to an empty list
    polygon_with_neighbors_gdf[neighbor_colname] = np.empty((len(polygon_gdf),
                                                    0)).tolist()

    # Iterate over rows of polygon_gdf GeoDataFrame
    for idx, row in polygon_gdf.iterrows():

        # Every 100 rows, print progress update
        if idx%100 == 0:
            print('Record', idx, 'of', num_rows)

        # Set bounds of current polygon
        poly_bounds = row['geometry'].bounds

        # Create bounding box (Shapely Polygon) from bounds
        poly_bbox = box(poly_bounds[0], poly_bounds[1], poly_bounds[2],
                        poly_bounds[3])

        # Iterate over rows again starting at idx+1 index
        # https://stackoverflow.com/questions/38596056/how-to-change-the-starting-index-of-iterrows
        for idx2, row2 in islice(polygon_with_neighbors_gdf.iterrows(), idx+1,
            None):

            # Check if polygon bounding box intersects polygon from other row
            if poly_bbox.intersects(row2['geometry']):

                # Append index of neighbors to the 2 rows
                polygon_with_neighbors_gdf.loc[idx, neighbor_colname].\
                    append(row2[neighbor_id_col])
                polygon_with_neighbors_gdf.loc[idx2, neighbor_colname].\
                    append(row[neighbor_id_col])

    return polygon_with_neighbors_gdf

def add_polygon_neighbors_column(polygon_gdf,
                                 neighbor_colname = "polygon_neighbors",
                                 neighbor_id_col='USO_AREA_U'):
    """In polygon GeoDataframe, add column with indices of neighboring polygons

    Finds all the neighboring polygons in a polygon GeoDataFrame by identifying
    which polygons intersect, overlap, cross, or touch the selected polygon. The
    function adds a new column with a list of indices of these neighboring
    polygons.

    Args:
        polygon_gdf: geopandas GeoDataFrame with Polygon geometry
        neighbor_colname: name of new column that will have the list of indices
            of neighboring polygons. By default, set to 'polygon_neighbors'
        neighbor_id_col: name of the column used as the identifier (or unique
            key) for neighboring polygons. By default, set to 'USO_AREA_U'

    Returns:
        polygon_with_neighbors_gdf: geopandas GeoDataFrame having an additional
        column with list of indices of neighboring polygons
    """
    # Make copy of polygon_gdf
    polygon_with_neighbors_gdf = polygon_gdf.copy()

    # Store number of rows
    num_rows = len(polygon_gdf)

    # Create new column with column name as neighbor_colname
    # Each value in the new column is set to an empty list
    polygon_with_neighbors_gdf[neighbor_colname] = np.empty((len(polygon_gdf),
                                                                0)).tolist()

    # Iterate over rows of polygon_gdf GeoDataFrame
    for idx, row in polygon_gdf.iterrows():

        # Every 100 rows, print progress update
        if idx%100 == 0:
            print('Record', idx, 'of', num_rows)

        # Set current polygon
        poly = row['geometry']

        # Iterate over rows again starting at idx+1 index
        # https://stackoverflow.com/questions/38596056/how-to-change-the-starting-index-of-iterrows
        for idx2, row2 in islice(polygon_with_neighbors_gdf.iterrows(), idx+1,
                                    None):

            # Check if polygon intersects polygon from other row
            if poly.intersects(row2['geometry']) or \
               poly.touches(row2['geometry']) or \
               poly.overlaps(row2['geometry']) or \
               poly.crosses(row2['geometry']):

                # Append index of neighbors to the 2 rows
                polygon_with_neighbors_gdf.loc[idx, neighbor_colname].\
                                            append(row2[neighbor_id_col])
                polygon_with_neighbors_gdf.loc[idx2, neighbor_colname].\
                                            append(row[neighbor_id_col])

    return polygon_with_neighbors_gdf

def create_neighbors_gdf(polygon_gdf, idx=0,
                         neighbor_colname = "polygon_neighbors",
                         neighbor_id_col='USO_AREA_U'):
    """Returns GeoDataFrame with all neighbors of the Polygon located at idx

    This function creates and return a new GeoDataFrame with all neighbors of
    a Polygon located at idx in polygon_gdf.

    Args:
        polygon_gdf: geopandas GeoDataFrame with Polygon geometry
        idx: index of the row to select from polygon_gdf. Defaults to 0 (first
            row).
        neighbor_colname: name of column that will have the list of indices of
            of neighboring polygons. By default, set to 'polygon_neighbors'
        neighbor_id_col: name of the column used as the identifier (or unique
            key) for neighboring polygons. By default, set to 'OBJECTID'

    Returns:
        A GeoDataFrame, neighbor_polygons, that contains all rows of polygon_gdf
        that are neighbors of the polygon at idx.
    """
    # Extract neighbor ID's, which are of OBJECTID
    neighbor_ids = polygon_gdf.iloc[idx, :][neighbor_colname]

    # If no neighbors, return None
    if neighbor_ids == []:
        return None

    # Initialize neighbor_polygons geoDataFrame
    neighbor_polygons = gpd.GeoDataFrame(crs=polygon_gdf.crs)

    for neighbor_id in neighbor_ids:

        # Find row where OBJECTID == neighbor_id
        row = polygon_gdf[polygon_gdf[neighbor_id_col] == neighbor_id]

        # Append row to neighbors_polygons
        neighbor_polygons = neighbor_polygons.append(row)

    return neighbor_polygons

def plot_polygon_and_neighbors(polygon_gdf, idx=0,
                               neighbor_colname = "polygon_neighbors",
                               neighbor_id_col='USO_AREA_U'):
    """Plots Polygon (at idx) with all of its neighbors in polygon_gdf

    This function creates a plot with the selected polygon in red and
    all neighboring polygons in gray

    Args:
        polygon_gdf: geopandas GeoDataFrame with Polygon geometry
        idx: index of the row to select from polygon_gdf. Defaults to 0 (first
            row).
        neighbor_colname: name of column that will have the list of indices of
            of neighboring polygons. By default, set to 'polygon_neighbors'
        neighbor_id_col: name of the column used as the identifier (or unique
            key) for neighboring polygons. By default, set to 'USO_AREA_U'

    Returns:
        None
    """
    # Use idx to select polygon of interest
    selected_polygon = polygon_gdf.loc[[idx],'geometry']

    # Create GeoDataFrame with polygon neighbors
    polygon_neighbors = create_neighbors_gdf(polygon_gdf=polygon_gdf, idx=idx,
                                            neighbor_colname=neighbor_colname,
                                            neighbor_id_col=neighbor_id_col)

    # Plot selected polygon in red and neighbors in gray
    plot_polygons_and_points(polygon1=polygon_neighbors,
        polygon2=selected_polygon)

def add_point_count_column(polygon_gdf, point_gdf, count_colname,
                           join_col='USO_AREA_U'):
    """Add count of services for each polygon to polygon_gdf (GeoDataFrame)

    Calculates number of points in eavch polygon using a spatial join. The
    counts of points within polygon are merged into the polygon GeoDataFrame.
    Code is based on:
    Count intersections: https://automating-gis-processes.github.io

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        point_gdf: GeoDataFrame with point geometries
        count_colname: name of new column that will have count of points in
            polygon
        join_col: unique key that allows merging of polygon_gdf with point
            counts. Default join column is 'USO_AREA_U'

    Returns:
        polygon_gdf_with_point_counts: A GeoDataFrame has polygon_gdf with an
            additional column (join_col) with counts of points in eaqch polygon.
    """

    # Count points within each polygon area
    point_cnt = gpd.sjoin(polygon_gdf, point_gdf).groupby(join_col).size().\
                                                        reset_index()

    # Rename point column to count_colname
    point_cnt = point_cnt.rename(columns={0: count_colname})

    # Merge point count with polygon_gdf data
    # Left join keeps all unique keys from polygon gdf
    polygon_gdf_with_point_counts = polygon_gdf.merge(point_cnt, how='left',
                                                        on=join_col)

    # Fill all NaN values as 0
    polygon_gdf_with_point_counts[count_colname] = \
                        polygon_gdf_with_point_counts[count_colname].fillna(0)

    # Cast point counts as integers
    polygon_gdf_with_point_counts[count_colname] = \
        polygon_gdf_with_point_counts[count_colname].astype(int)

    return polygon_gdf_with_point_counts

def calc_nbr_dist(polygon_gdf, nbr_dist_colname='nbr_dist',
                    centroid_colname='centroid',
                    neighbor_colname = "polygon_neighbors",
                    neighbor_id_col='USO_AREA_U'):
    """Add column with distances to neighbors

    Calculate distances between centroids of polygons and centroids of their
    neighbors and add this as additional column to polygon_gdf

    Args:
        polygon_gdf: geopandas GeoDataFrame with Polygon geometry
        nbr_dist_colname: name of column that will have neighbor id's and
            distances. By default, set to 'nbr_dist'
        centroid_colname: name of column that will have centroid for each
            polygon. By default, set to 'centroid'
        neighbor_colname: name of column that will have the list of indices of
            of neighboring polygons. By default, set to 'polygon_neighbors'
        neighbor_id_col: name of the column used as the identifier (or unique
            key) for neighboring polygons. By default, set to 'USO_AREA_U'

    Returns:
        GeoDataFrame with additional column that includes neighbor id's and
        distances to neighbors as a list of tuples in the following format:
        [(nbr_id1, nbr_dist1), (nbr_id2, nbr_dist1), ...]

    """

    # Make copy of polygon_gdf
    gdf_copy = polygon_gdf.copy()

    # Create new column and initialize with empty list
    gdf_copy[nbr_dist_colname] = np.empty((len(gdf_copy), 0)).tolist()

    # Iterate over rows in GeoDataFrame
    for idx, row in gdf_copy.iterrows():

        # Extract row centroid and list of neighbors
        row_centroid = row[centroid_colname] # Shapely Point object
        neighbor_ids = row[neighbor_colname]

        for neighbor_id in neighbor_ids:
            neighbor_row = gdf_copy[gdf_copy[neighbor_id_col] == neighbor_id]
            # Since neighbor_row['centroid'] is Series, we need
            # .array[0] to extract the Shapely Point object
            neighbor_centroid = neighbor_row[centroid_colname].array[0]
            neighbor_distance = row_centroid.distance(neighbor_centroid)

            gdf_copy.loc[idx, nbr_dist_colname].append((neighbor_id, \
                                                neighbor_distance))

    return gdf_copy

def calc_pcen_mobile(polygon_gdf, count_colname,
                     pcen_mobile_colname,
                     nbr_dist_colname='nbr_dist',
                     pop_colname='population',
                     id_col='USO_AREA_U'):
    """ Calculates and adds column for PCEN_mobile

    Calculates effective number of service points within a
    polygon divided by the population size. This effective number
    not only counts service points within the polygon but also
    service points in neighboring polygons, inversely weighted
    by distance between centroid of selected polygon and centroids
    of its neighbors.

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        count_colname: name of column with count of points in
            polygon
        pcen_mobile_colname: name of column with pcen_mobile number
        nbr_dist_colname: name of column that will have neighbor id's and
            distances. By default, set to 'nbr_dist'
        pop_colname: column name for population
        id_col: column name for ID. Defaults to 'USO_AREA_U'

    Returns:
        GeoDataFrame with pcen_mobile column added.
    """

    # Make copy of polygon_gdf
    gdf_copy = polygon_gdf.copy()

    # Create new column for pcen_mobile
    gdf_copy[pcen_mobile_colname] = 0

    # iterate through GeoDataFrame
    for idx, row in gdf_copy.iterrows():

        # polygon's population
        poly_pop = row[pop_colname]

        # initialize effective service count with polygon's count
        poly_count = row[count_colname]

        # Iterate through each neighbor of the polygon
        for nbr_id, nbr_dist in row[nbr_dist_colname]:

            # Extract service count of neighbor
            nbr_count = gdf_copy[gdf_copy[id_col]==nbr_id][count_colname].array[0]

            # Add this service count (discounted by distance)
            # to effective count of services for polygon
            poly_count += nbr_count * (1/(1+nbr_dist))

        # Divide effective service count by population size
        # and add to the pcen_mobile column
        gdf_copy.loc[idx, pcen_mobile_colname] = poly_count/poly_pop

    return gdf_copy

def calc_service_index(polygon_gdf, pcen_mobile_colname, service_idx_colname):
    """Calculates service index [0, 1] based on PCEN_MOBILE

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        pcen_mobile_colname: name of column with pcen_mobile number
        service_idx_colname: name of column with service index

    Returns:
        GeoDataFrame with service index added
    """
    # Make copy of polygon_gdf
    gdf_copy = polygon_gdf.copy()

    # Calculate min and max of PCEN_mobile
    pcen_min = gdf_copy[pcen_mobile_colname].min()
    pcen_max = gdf_copy[pcen_mobile_colname].max()

    # Create new service index column based on min-max method
    gdf_copy[service_idx_colname] = gdf_copy[pcen_mobile_colname] - pcen_min
    gdf_copy[service_idx_colname] /= pcen_max-pcen_min

    return gdf_copy

def create_service_index(polygon_gdf, point_gdf, service_name, epsg_code):
    """Create service index

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        point_gdf: GeoDataFrame with point geometries
        service_name: name of public service
        epsg_code: EPSG code for point_gdf reprojection

    Returns:
        GeoDataFrame with column '{service_name}_idx' added
        with values between 0 and 1 (inclusive).
    """

    # Define column names to be used
    count_colname = "{}_count".format(service_name)
    pcen_mobile_colname = "{}_pcen".format(service_name)
    service_idx_colname = "{}_idx".format(service_name)

    # Make copy of polygon_gdf
    gdf_copy = polygon_gdf.copy()

    # Reproject point to EPSG 3857
    point_gdf = reproject_gdf(point_gdf, epsg_code)

    # Add number of service points for each polygon
    gdf_copy = add_point_count_column(polygon_gdf=gdf_copy,
                                      point_gdf=point_gdf,
                                      count_colname=count_colname)

    # Calculate and add PCEN_Mobile column
    gdf_copy = calc_pcen_mobile(gdf_copy, count_colname=count_colname,
                                pcen_mobile_colname=pcen_mobile_colname)

    # Calculate and add service index column
    gdf_copy = calc_service_index(gdf_copy,
                                    pcen_mobile_colname=pcen_mobile_colname,
                                    service_idx_colname=service_idx_colname)

    # Drop additional columns
    gdf_copy = gdf_copy.drop(columns=[pcen_mobile_colname, count_colname])

    return gdf_copy
