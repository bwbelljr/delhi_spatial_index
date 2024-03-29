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
from tqdm import tqdm

def create_nbrs_and_dist_columns(colonies):
    """Create two neighbors' columns in GeoDataFrame

    This function computes and adds the two columns that
    have information on bounding box neighbors of each
    polygon: `nbrs_bbox` and `nbrs_dist_bbox`.

    Args:
        colonies: GeoDataFrame with all information needed
            to calculate neighbors columns

    Returns:
        GeoDataFrame with two new columns: `nbrs_bbox` and `nbrs_dist_bbox`
    """

    # Create GeoDataFrame with Bounding Box of each Polygon
    colonies_bbox = create_bbox_gdf(colonies)

    print('Calculate bounding box neighbors column: `nbrs_bbox`')

    # Calculate bounding box neighbors column: `nbrs_bbox`
    colonies_bbox_nbrs = add_polygon_neighbors_column_fast(polygon_gdf=colonies,
                                                right_gdf=colonies_bbox,
                                                id_colname='USO_AREA_U',
                                                neighbor_colname='nbrs_bbox',
                                                barrier_colname='barrier')

    print('Calculate dist from polygons to their neighbors: `nbrs_dist_bbox`')

    # Calculate distances from polygons to their neighbors: `nbrs_dist_bbox`
    colonies_bbox_nbrs = calc_nbr_dist(polygon_gdf=colonies_bbox_nbrs,
                                  nbr_dist_colname='nbrs_dist_bbox',
                                  centroid_colname='centroid',
                                  neighbor_colname='nbrs_bbox',
                                  neighbor_id_col='USO_AREA_U')

    return colonies_bbox_nbrs

def create_exclude_column(colonies_gdf, uso_types_to_drop, area_cutoff_km2):
    """Create exclude column that is True for polygons with certain conditions

    This function adds a binary column `exclude_from_psi` to GeoDataFrame.
    It is True if the area of the polygon < area_cutoff_km2 or the
    'USO_FINAL' column is one of the `uso_area_types`.

    Args:
        colonies_gdf: Colonies shapefile as GeoDataFrame
        uso_types_to_drop: List of string names of USO_FINAL
            to exclude.
        area_cutoff_km2: threshold below which colonies qualify
            for exclusion.

    Returns:
        GeoDataFrame with column `exclude_from_psi`
    """

    exclude_colname = 'exclude_from_psi'

    # Initialize new column
    colonies_gdf[exclude_colname] = False

    # Indicate settlement categories with exclusion criteria
    for uso_type in uso_types_to_drop:

        # Setting exclude=True for USO_FINAL=uso_type
        # We iterate over all indices matching this criteria
        for idx in colonies_gdf[colonies_gdf['USO_FINAL'] == uso_type].index:
            colonies_gdf.loc[idx, exclude_colname] = True

    # Indicate polygon to be excluded based on area cutoff
    # We iterate over all indices matching this criteria
    for idx in colonies_gdf[colonies_gdf['area_km2'] < .0001].index:
        colonies_gdf.loc[idx, exclude_colname] = True

    return colonies_gdf

def generate_colonies_with_exclusions(colonies_pkl_file, columns_to_drop,
    uso_types_to_drop, area_cutoff_km2):
    """Generate colonies shapefile with neighbors' and exclude_from_psi columns

    Args:
        colonies_pkl_file (str): Colonies shapefile pre-processed (e.g., area,
            population, distance to ndmc, barrier, neighbors, etc.) in Pickle
            format.
        columns_to_drop (list): Specifies columns to drop from previously
            processed colonies file.
        uso_types_to_drop (list): List of string names of USO_FINAL
            to exclude.
        area_cutoff_km2 (float): Area cutoff for colonies

    Returns:
        Colonies shapefile (GeoDataFrame object) that has removed
        `columns_to_drop`, all colonies with area <= area_cutoff_km2, and
        bounding box neighbors' columns.
    """
    # open colonies Pickle file
    with open(colonies_pkl_file, 'rb') as f:
        colonies = pickle.load(f)

    # Remove `columns_to_drop`
    colonies = colonies.drop(columns=columns_to_drop)

    # Add exclude column
    colonies = create_exclude_column(colonies_gdf = colonies,
                                     uso_types_to_drop = uso_types_to_drop,
                                     area_cutoff_km2 = area_cutoff_km2)

    # Add two neighbors' columns
    colonies_bbox_nbrs = create_nbrs_and_dist_columns(colonies)

    return colonies_bbox_nbrs

def get_row_index(polygon_gdf, id_colname, id_num):
    """Get row index of GeoDataFrame given a unique id number"""
    return polygon_gdf[polygon_gdf[id_colname] == id_num].index.values[0]

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

def print_invalid_rows(gdf):
    """Print rows with invalid geometries"""
    for i, row in gdf.iterrows():
        if not row['geometry'].is_valid:
            print('not valid index', i, '\n', row)

def gdf_has_duplicate_rows(gdf):
    """Returns True if gdf GeoDataFrame has duplicate rows

    Args:
        gdf: GeoDataFrame

    Returns:
        Boolean value returned based on whether gdf has duplicate
        rows (True) or not (False)

    """

    # Create mask for all duplicate rows
    gdf_duplicate_mask = gdf.duplicated()

    # Calculate # duplicate rows
    number_duplicate_rows = len(gdf[gdf_duplicate_mask])

    # Return True if there are any duplicate rows
    return number_duplicate_rows > 0

def gdf_within_delhi(gdf, delhi_bounds_filepath):
    """Return True if gdf is in Delhi

    Args:
        gdf: GeoDataFrame
        delhi_bounds_filepath: file path for shapefile
            with Delhi's bounding box

    Returns:
        True if geometries are within bounds of Delhi
    """

    delhi_bounds = gpd.read_file(delhi_bounds_filepath)

    # reproject gdf to same CRS as delhi_bounds
    gdf = gdf.to_crs(delhi_bounds.crs)

    gdf_bounds = box(gdf.total_bounds[0], gdf.total_bounds[1],
                     gdf.total_bounds[2], gdf.total_bounds[3])

    # Shapely predicate 'contains' shows if bounding
    # box of shapefile is contained with Delhi's
    # bounding box
    delhi_contains_gdf  = delhi_bounds.contains(gdf_bounds)

    # Extract first element of Series
    # There is only one element since gdf_bounds
    # is a single geometry
    return delhi_contains_gdf[0]

def check_shapefile(gdf, gdf_name, geom_type, delhi_bounds_filepath):
    """Prints information on validity of shapefile

    Checks if shapefile has duplicate rows, rows with invalid
    geometries, rows with None in geometry field, whether
    all geometries are of geom_type, and whether shapefile's extent
    is fully contained within Delhi.

    Args:
        gdf: GeoDataFrame with geometry column named
            as 'geometry'
        gdf_name: name of gdf (e.g., colonies, schools)
        geom_type: string, representing one of 3
            Shapely objects. Possible values are
            Point, Line, and Polygon
        delhi_bounds_filepath: file path for shapefile
            with Delhi's bounding box

    Returns:
        n/a. Just prints statements
    """
    assert geom_type in ['Point', 'Line', 'Polygon'], 'invalid geom_type'
    assert isinstance(gdf_name, str), 'gdf_name is not a string'
    assert 'geometry' in gdf.columns, 'there is no "geometry" column'

    separator = '----------------------------------------------------'

    # Check for duplicate rows
    print(gdf_name, 'has duplicate rows:', gdf_has_duplicate_rows(gdf))

    # Print rows with invalid geometries
    print(separator)
    print('rows with invalid geometries \n')
    print_invalid_rows(gdf)
    print(separator)

    # Check that all geometries are geom_type
    all_geom_type = check_geometries(gdf, geom_type)
    print('all geometries in {} are of type {}: {}'.format(gdf_name,
                                                    geom_type, all_geom_type))
    print(separator)

    # Print all rows where geometry=None
    rows_with_none_geom = gdf[gdf['geometry'] == None]
    print('Rows with None value in geometry column are below')
    print(rows_with_none_geom)
    print(separator)

    # Check that shapefile lies within Delhi
    in_delhi = gdf_within_delhi(gdf, delhi_bounds_filepath)
    print('{} shapefile is contained within Delhi: {}'.format(gdf_name,
                                                        in_delhi))
    print(separator)

    print('Done with shapefile evaluation')


def remove_duplicate_geom(gdf, geom_colname='geometry'):
    """Removes rows with duplicate geometries

    Checks if any rows have duplicate geometries and removes
    them. This is based on Shapely's `object.equals(other)` method
    to compare two geometries. Note that this function currently
    runs in O(n^2) time.

    Args:
        gdf: GeoDataFrame with Shapely objects (e.g., Point,
            Line, or Polygon)
        geom_colname: Name of geometry column. Default is
            'geometry'.

    Returns:
        GeoDataFrame with all rows removed having duplicate
        geometries. Note that this returns a GeoDataFrame
        with a new index (instead of preserving the old index).
    """

    # Original size of gdf
    old_size = len(gdf)

    # Initialize new column `not_duplicate`
    # Assumes that every row is not a duplicate
    gdf['not_duplicate'] = True

    for idx, row in tqdm(gdf.iterrows()):

        if idx % 100 == 0:
            print(f"{idx}/{old_size}")

        row_geom = row[geom_colname]

        # Iterate over rows again starting at idx+1 index
        # https://stackoverflow.com/questions/38596056/how-to-change-the-starting-index-of-iterrows
        for idx2, row2 in islice(gdf.iterrows(), idx+1, None):
            other_geom = row2[geom_colname]

            if row_geom.equals(other_geom):

                # Keeps first occurrence but removes the subsequent ones
                gdf.loc[idx2, 'not_duplicate'] = False

    # Only select rows that are not duplicate
    # In other words, the 'not_duplicate' value is True
    gdf = gdf[gdf['not_duplicate']]

    # Remove 'not_duplicate' column
    gdf = gdf.drop(columns=['not_duplicate'])

    # Reset index
    gdf = gdf.reset_index()

    # Original size of gdf
    new_size = len(gdf)

    print('Original number of rows is {}:'.format(old_size))
    print('New number of rows after deduplication is: {}'.format(new_size))

    return gdf

def check_geometries(gdf, geom_type):
    """ Returns True if all geometries are of geom_type

    Args:
        gdf: GeoDataFrame with any type of geometry (Point, Line, Polygon)
        geom_type: string with either "Point", "Line", or "Polygon"

    Returns:
        Boolean where True means all geometries are of the type specified.
        For this function, Polygon and MultiPolygon are considered valid
        geometries for geom_type='Polygon'
    """

    assert geom_type in ["Point", "Line", "Polygon"], "Input valid geom_type"

    # Create new column with type of each row geometry
    gdf['geom_type'] = type(gdf['geometry'])

    # Find unique values of geometry types
    geom_type_list = gdf.geom_type.unique()

    # Check if each type is of geom_type
    geom_is_geom_type = [geom_type in geom for geom in geom_type_list]

    # Remove 'geom_type' column
    gdf = gdf.drop(columns=['geom_type'])

    # If at least one geometry is not of geom_type, return False
    # Otherwise, return True
    if False in geom_is_geom_type:
        return False
    else:
        return True

def barrier_intersection(colonies_gdf, barrier_gdf, barrier_colname,
    id_colname="USO_AREA_U"):
    """ Add new column indicating intersection with barrier

    Args:
        colonies_gdf: GeoDataFrame with colonies shapefile
        barrier_gdf: GeoDataFrame with barrier (e.g., canal, railway, drain)
        barrier_colname: string, e.g., "canal", "railway", or "drain"
        id_colname: unique ID for colonies_gdf. Default is "USO_AREA_U"

    Returns:
        GeoDataFrame having column (barrier_colname) with boolean value
        indicating whether the Shapefile intersects barrier or now
    """
    # Assume all colonies do not intersect barrier
    colonies_gdf[barrier_colname] = False

    # Spatial (inner) join
    joined_colonies_barrier = gpd.sjoin(colonies_gdf, barrier_gdf, how='inner')

    # List of colonies (by unique id) that intersect barrier
    colony_ids_with_intersection = list(joined_colonies_barrier[id_colname].\
                                    unique())

    for colony_id in colony_ids_with_intersection:
        # Extract index number of row with colony_id
        colony_index = colonies_gdf[colonies_gdf[id_colname] == colony_id].\
                        index.values[0]

        # Add True for intersecting colonies
        colonies_gdf.loc[colony_index, barrier_colname] = True

    return colonies_gdf

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

def add_polygon_neighbors_column_bbox(polygon_gdf,
    neighbor_colname = 'polygon_neighbors', neighbor_id_col='USO_AREA_U',
    barrier_colname='barrier'):
    """Add column with neighboring polygons indices based on bbox

    Finds all the neighboring polygons in a polygon GeoDataFrame based on
    whether polygons intersect bounding box of selected Polygon and adds a new
    column with a list of indices of these neighboring polygons.

    Args:
        polygon_gdf: geopandas GeoDataFrame with Polygon geometry. Assumes that
            GeoDataFrame has column 'USO_FINAL' with USO settlement types.
        neighbor_colname: name of new column that will have the list of indices
            of neighboring polygons. By default, set to 'polygon_neighbors'
        neighbor_id_col: name of the column used as the identifier (or unique
            key) for neighboring polygons. By default, set to 'USO_AREA_U'
        barrier_colname: name of column that indicates if polygon intersects
            a barrier

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
    polygon_with_neighbors_gdf[neighbor_colname] = np.empty((len(polygon_gdf), 0)).tolist()

    # Iterate over rows of polygon_gdf GeoDataFrame
    for idx, row in polygon_gdf.iterrows():

        # Every 100 rows, print progress update
        if idx%100 == 0:
            print('Record', idx, 'of', num_rows)

        # Set bounds of current polygon
        poly_bounds = row['geometry'].bounds

        # Create bounding box (Shapely Polygon) from bounds
        poly_bbox = box(poly_bounds[0], poly_bounds[1], poly_bounds[2], poly_bounds[3])

        # Iterate over rows again starting at idx+1 index
        # https://stackoverflow.com/questions/38596056/how-to-change-the-starting-index-of-iterrows

        # Check if polygon bounding box intersects polygon from other row
        # Append index of neighbors to the respective rows only if
        # they do not have a barrier
        for idx2, row2 in islice(polygon_with_neighbors_gdf.iterrows(), idx+1, None):
            if poly_bbox.intersects(row2['geometry']):
                if not row2['barrier']:
                    polygon_with_neighbors_gdf.loc[idx, neighbor_colname].append(row2[neighbor_id_col])

                if not row['barrier']:
                    polygon_with_neighbors_gdf.loc[idx2, neighbor_colname].append(row[neighbor_id_col])

    return polygon_with_neighbors_gdf

def add_polygon_neighbors_column(polygon_gdf,
                                 neighbor_colname = "polygon_neighbors",
                                 neighbor_id_col='USO_AREA_U',
                                 barrier_colname='barrier'):
    """Add column with indices of neighboring polygons based on intersection

    Finds all the neighboring polygons in a polygon GeoDataFrame by identifying
    which polygons intersect, overlap, cross, or touch the selected polygon. The
    function adds a new column with a list of indices of these neighboring
    polygons.

    Args:
        polygon_gdf: geopandas GeoDataFrame with Polygon geometry. Assumes that
            GeoDataFrame has column 'USO_FINAL' with USO settlement types.
        neighbor_colname: name of new column that will have the list of indices
            of neighboring polygons. By default, set to 'polygon_neighbors'
        neighbor_id_col: name of the column used as the identifier (or unique
            key) for neighboring polygons. By default, set to 'USO_AREA_U'
        barrier_colname: name of column that indicates if polygon intersects
            a barrier
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

               # Append index of neighbors to the respective rows only if
               # they do not have a barrier
               if not row2[barrier_colname]:
                   polygon_with_neighbors_gdf.loc[idx, neighbor_colname].\
                                            append(row2[neighbor_id_col])

               if not row[barrier_colname]:
                   polygon_with_neighbors_gdf.loc[idx2, neighbor_colname].\
                                            append(row[neighbor_id_col])

    return polygon_with_neighbors_gdf

def remove_ids_with_barrier(id_list, polygon_gdf, id_colname, barrier_colname):
    """Remove all unique ids where there is a barrier"""

    new_list = id_list[:]

    for id_num in id_list:
        id_num_idx = get_row_index(polygon_gdf, id_colname, id_num)
        barrier_exists = polygon_gdf.loc[id_num_idx, barrier_colname]
        if barrier_exists:
            new_list.remove(id_num)

    return new_list

def add_polygon_neighbors_column_fast(polygon_gdf, right_gdf, id_colname,
    neighbor_colname, barrier_colname):
    """Add polygon neighbors based on spatial join"""

    # Spatial left join
    # right_gdf can be polygons or bounding boxes
    joined_gdf = gpd.sjoin(polygon_gdf, right_gdf, how='left')

    id_colname_left = id_colname + '_left'
    id_colname_right = id_colname + '_right'

    # Groupby id_colname
    joined_grouped = joined_gdf.groupby(id_colname_left)

    # Make copy of polygon_gdf
    # and create new column for neighbors list
    nbrs_touch_gdf = polygon_gdf.copy()

    # Create new column with column name as neighbor_colname
    # Each value in the new column is set to an empty list
    nbrs_touch_gdf[neighbor_colname] = np.empty((len(nbrs_touch_gdf), 0)).tolist()

    for group in tqdm(joined_grouped.groups):

        # Create list of id numbers that intersect with group
        group_list = list(joined_grouped.get_group(group)[id_colname_right])

        # Because a polygon intersects itself, remove it from the list
        group_list.remove(group)

        # Get index number of group
        group_idx = get_row_index(nbrs_touch_gdf, id_colname, group)

        # Remove ID's where there is a barrier
        group_list = remove_ids_with_barrier(id_list = group_list,
                                polygon_gdf = nbrs_touch_gdf,
                                id_colname = id_colname,
                                barrier_colname = barrier_colname)

        # Insert modified list into nbrs_touch_gdf
        nbrs_touch_gdf.loc[group_idx, neighbor_colname].extend(group_list)

    return nbrs_touch_gdf

def create_bbox_gdf(polygon_gdf):
    """Create GeoDataFrame with bounding box as geometry"""

    # Concatenate polygon_gdf with bounding box columns
    gdf_bbox = gpd.GeoDataFrame(pd.concat([polygon_gdf, polygon_gdf.bounds], axis=1))

    # Initialize bounding box column
    gdf_bbox['bbox'] = None

    # Create bounding box for all geometries
    for idx, row in gdf_bbox.iterrows():
        row_bbox = box(row['minx'], row['miny'], row['maxx'], row['maxy'])
        gdf_bbox.loc[idx, 'bbox'] = row_bbox

    # Remove geometry and bounds columns
    gdf_bbox = gdf_bbox.drop(columns=['geometry', 'minx', 'miny', 'maxx', 'maxy'])

    # Rename bbox as geometry field
    gdf_bbox = gdf_bbox.rename(columns={'bbox':'geometry'})

    # Set CRS to that of polygon_gdf
    gdf_bbox.crs = polygon_gdf.crs

    return gdf_bbox

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

def calc_service_length(small_gdf, poly_geom_colname, line_geom_colname):
    """Calculate length of all (poly)line services in a colony
    Args:

        small_gdf: GeoDataFrame, which is a derived from a groupby
            object based on 'USO_AREA_U'
        poly_geom_colname: name of geometry column for colonies
        line_geom_colname: name of geometry column for (poly)line services-

    Returns:
        Length (kilometers) as a float.
    """

    total_length = 0

    for i, row in small_gdf.iterrows():
        polygon = row[poly_geom_colname]
        line = row[line_geom_colname]
        intersection = polygon.intersection(line)
        length = intersection.length/1000
        total_length += length

    return total_length

def add_service_length_column(polygon_gdf, line_gdf, length_colname,
    id_colname='USO_AREA_U'):
    """Add distance of (poly)line services for each polygon in polygon_gdf

    Calculates distance of (poly)line service within each polygon (e.g., roads).
    The function first does a spatial join between polygon_gdf and line_gdf,
    keeping both geometries. This joined GeoDataFrame is grouped by id_colname.
    Within each group, the length of the intersection of each (poly)line with
    the polygon is added up. This aggregate length is added to polygon_gdf as
    length_colname.

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries. Assumes that its
            geometry column is named 'geometry'.
        line_gdf: GeoDataFrame with (poly)line geometries.  Assumes that its
            geometry column is named 'geometry'.
        length_colname: name of new column that will have distance of service
            (poly)line(s) in polygon
        id_colname: unique key that identifies colonies. Default id column
            name is 'USO_AREA_U'

    Returns:
        A GeoDataFrame has polygon_gdf with an additional column
        (length_colname) with distance of service (poly)line(s) in each polygon.
    """

    polygon_gdf[length_colname] = 0

    # Spatial join removes geometry column from one GeoDataFrame
    # Copy geometry so that it can be used after the spatial join
    line_geom_colname = 'line_geometry'
    line_gdf[line_geom_colname] = line_gdf['geometry']

    # Inner spatial join
    joined = gpd.sjoin(polygon_gdf, line_gdf)

    # Create groupby object based on id_colname
    joined_grouped= joined.groupby(id_colname)

    for name, group in joined_grouped:
        # Compute index of id. Will be used to locate
        # and modify rows of polygon_gdf
        name_index = polygon_gdf[polygon_gdf[id_colname] == name].index.\
                                                                values[0]

        total_road_length = calc_service_length(small_gdf=group,
                                            poly_geom_colname="geometry",
                                            line_geom_colname=line_geom_colname)

        polygon_gdf.loc[name_index, length_colname] = total_road_length

    return polygon_gdf

def create_service_length_index(polygon_gdf, line_gdf, service_name, epsg_code,
    nbr_dist_colname, pcen_denom):
    """ Create service index for services with (poly)lines

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        line_gdf: GeoDataFrame with line geometries
        service_name: name of public service
        epsg_code: EPSG code for point_gdf reprojection
        nbr_dist_colname: name of column that will have neighbor id's and
            distances.
        pcen_denom: String with values "pop", "popdensity", or "one"

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
    line_gdf = reproject_gdf(line_gdf, epsg_code)

    # Add service length for each polygon
    gdf_copy = add_service_length_column(polygon_gdf=gdf_copy,
                                            line_gdf=line_gdf,
                                            length_colname= count_colname)

    # Calculate and add PCEN_Mobile column
    gdf_copy = calc_pcen_mobile(gdf_copy, count_colname=count_colname,
                                pcen_mobile_colname=pcen_mobile_colname,
                                pcen_denom=pcen_denom,
                                nbr_dist_colname=nbr_dist_colname)

    # Calculate and add service index column
    gdf_copy = calc_service_index(gdf_copy,
                                    pcen_mobile_colname=pcen_mobile_colname,
                                    service_idx_colname=service_idx_colname)

    # Drop additional columns
    # gdf_copy = gdf_copy.drop(columns=[pcen_mobile_colname, count_colname])

    return gdf_copy

def create_service_length_index_wards(polygon_gdf, line_gdf, service_name,
    epsg_code, calc_pop_density):
    """ Create service index for services with (poly)lines

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        line_gdf: GeoDataFrame with line geometries
        service_name: name of public service
        epsg_code: EPSG code for point_gdf reprojection
        calc_pop_density: If True, divide by population/area. Else divide
            by population.

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
    line_gdf = reproject_gdf(line_gdf, epsg_code)

    # Add service length for each polygon
    gdf_copy = add_service_length_column(polygon_gdf=gdf_copy,
                                            line_gdf=line_gdf,
                                            length_colname= count_colname,
                                            id_colname="WARD_NO")

    # Calculate and add PCEN_Mobile column
    gdf_copy = calc_pcen_mobile_no_neighbors(polygon_gdf=gdf_copy,
                                            count_colname=count_colname,
                                        pcen_mobile_colname=pcen_mobile_colname,
                                        calc_pop_density=calc_pop_density,
                                        pop_colname='Total Population',
                                        id_col='WARD_NO')

    # Calculate and add service index column
    gdf_copy = calc_service_index(gdf_copy,
                                    pcen_mobile_colname=pcen_mobile_colname,
                                    service_idx_colname=service_idx_colname)

    # Drop additional columns
    # gdf_copy = gdf_copy.drop(columns=[pcen_mobile_colname, count_colname])

    return gdf_copy


def create_service_length_index_buffer(polygon_gdf, line_gdf, service_name,
    calc_pop_density, epsg_code):
    """ Create service index for services with (poly)lines with buffer

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        line_gdf: GeoDataFrame with line geometries
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
    line_gdf = reproject_gdf(line_gdf, epsg_code)

    # Add service length for each polygon
    gdf_copy = add_service_length_column(polygon_gdf=gdf_copy,
                                            line_gdf=line_gdf,
                                            length_colname= count_colname,
                                            id_colname="USO_AREA_U")

    # Calculate and add PCEN_Mobile column
    gdf_copy = calc_pcen_mobile_no_neighbors(polygon_gdf=gdf_copy,
                                            count_colname=count_colname,
                                        pcen_mobile_colname=pcen_mobile_colname,
                                        calc_pop_density=calc_pop_density,
                                        pop_colname='population',
                                        id_col='USO_AREA_U')

    # Calculate and add service index column
    gdf_copy = calc_service_index(gdf_copy,
                                    pcen_mobile_colname=pcen_mobile_colname,
                                    service_idx_colname=service_idx_colname)

    # Drop additional columns
    # gdf_copy = gdf_copy.drop(columns=[pcen_mobile_colname, count_colname])

    return gdf_copy


def calc_nbr_dist(polygon_gdf, nbr_dist_colname='nbr_dist',
                    centroid_colname='centroid',
                    neighbor_colname = "polygon_neighbors",
                    neighbor_id_col='USO_AREA_U'):
    """Add column with distances to neighbors (in kilometers)

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
    with tqdm(total = len(gdf_copy)) as pbar:
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

                # Convert neighbor_distance unit to kilometers
                neighbor_distance = neighbor_distance/1000

                gdf_copy.loc[idx, nbr_dist_colname].append((neighbor_id, \
                                                    neighbor_distance))

            pbar.update(1)

    return gdf_copy

def calc_pcen_mobile(polygon_gdf, count_colname,
                     pcen_mobile_colname,
                     pcen_denom,
                     nbr_dist_colname='nbr_dist',
                     pop_colname='population',
                     area_colname='area_km2',
                     id_col='USO_AREA_U'):
    """ Calculates and adds column for PCEN_mobile

    Calculates effective number of service points within a
    polygon divided by population size, density, or 1. This effective number
    not only counts service points within the polygon but also
    service points in neighboring polygons, inversely weighted
    by distance between centroid of selected polygon and centroids
    of its neighbors. Note that polygons to be excluded get pcen_mobile = -1.

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        count_colname: name of column with count of points in
            polygon
        pcen_denom: If "pop", denominator is Population. If "popdensity",
            denominator is Population density (population/area). If "one",
            denominator=1.
        pcen_mobile_colname: name of column with pcen_mobile number
        nbr_dist_colname: name of column that will have neighbor id's and
            distances. By default, set to 'nbr_dist'
        pop_colname: column name for population, default 'population'
        area_colname: column name for area, default 'area_km2'
        id_col: column name for ID. Defaults to 'USO_AREA_U'

    Returns:
        GeoDataFrame with pcen_mobile column added.
    """

    # Make copy of polygon_gdf
    gdf_copy = polygon_gdf.copy()

    # Create new column for pcen_mobile
    # Note that all excluded polygons will default to this value
    gdf_copy[pcen_mobile_colname] = -1

    # iterate through GeoDataFrame
    for idx, row in gdf_copy.iterrows():

        # For all to be excluded, skip to next row
        #if row['exclude_from_psi']:
        #    continue

        # denominator for PCEN equation is either population or
        # population density (population/area) or 1
        if pcen_denom == 'popdensity':
            denom = row[pop_colname]/row[area_colname]
        elif pcen_denom == 'pop':
            denom = row[pop_colname]
        elif pcen_denom == "one":
            denom = 1

        # initialize effective service count with polygon's count
        poly_count = row[count_colname]

        # Iterate through each neighbor of the polygon
        for nbr_id, nbr_dist in row[nbr_dist_colname]:

            try: #try-except to skip missing (RV) colonies
                # Extract service count of neighbor

                nbr_count = gdf_copy[gdf_copy[id_col]==nbr_id][count_colname].array[0]

                # Add this service count (discounted by distance)
                # to effective count of services for polygon
                poly_count += nbr_count * (1/(1+nbr_dist))

            except:
                pass


        # Divide effective service count by population size
        # and add to the pcen_mobile column
        gdf_copy.loc[idx, pcen_mobile_colname] = poly_count/denom

    return gdf_copy

def calc_pcen_mobile_no_neighbors(polygon_gdf, count_colname,
                     pcen_mobile_colname, calc_pop_density,
                     pop_colname='population', area_colname='area_km2',
                     id_col='USO_AREA_U'):
    """ Calculates and adds column for PCEN_mobile (no neighbors)

    Calculates number of service points within a
    polygon divided by the population size or density.

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        count_colname: name of column with count of points in
            polygon
        pcen_mobile_colname: name of column with pcen_mobile number
        calc_pop_density: If True, calculate population density, else
            population.
        pop_colname: column name for population; defaults to 'population'
        area_colname: column name for area; defaults to 'area_km2'
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

        # polygon's population (density)
        if calc_pop_density:
            denom = row[pop_colname]/row[area_colname]
        else:
            denom = row[pop_colname]

        # initialize effective service count with polygon's count
        poly_count = row[count_colname]

        # Divide service count by population size
        # and add to the pcen_mobile column
        gdf_copy.loc[idx, pcen_mobile_colname] = poly_count/denom

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
    # get first value greater than -1, which is the smallest value
    #pcen_min = sorted(gdf_copy[pcen_mobile_colname].unique())[1]
    pcen_min = gdf_copy[pcen_mobile_colname].min()
    pcen_max = gdf_copy[pcen_mobile_colname].max()

    # initialize service index column with -1, default value for
    # excluded polygons
    gdf_copy[service_idx_colname] = -1

    # Create new service index column based on min-max method
    for idx, row in gdf_copy.iterrows():
        # Exclude polygons
        #if row['exclude_from_psi']:
        #    continue

        result = (row[pcen_mobile_colname] - pcen_min)/(pcen_max-pcen_min)
        gdf_copy.loc[idx, service_idx_colname] = result

    return gdf_copy

def create_service_index(polygon_gdf, point_gdf, service_name, epsg_code,
    pcen_denom, nbr_dist_colname):
    """Create service index

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        point_gdf: GeoDataFrame with point geometries
        service_name: name of public service
        epsg_code: EPSG code for point_gdf reprojection
        pcen_denom: String with values of "pop", "popdensity", or "one"
        nbr_dist_colname: name of column that will have neighbor id's and
            distances.


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
                                pcen_mobile_colname=pcen_mobile_colname,
                                pcen_denom = pcen_denom,
                                nbr_dist_colname=nbr_dist_colname)

    # Calculate and add service index column
    gdf_copy = calc_service_index(gdf_copy,
                                    pcen_mobile_colname=pcen_mobile_colname,
                                    service_idx_colname=service_idx_colname)

    # Drop additional columns
    # gdf_copy = gdf_copy.drop(columns=[pcen_mobile_colname, count_colname])

    return gdf_copy

def create_service_index_wards(polygon_gdf, point_gdf, service_name,
    calc_pop_density, epsg_code):
    """Create service index for Wards (no neigbors)

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        point_gdf: GeoDataFrame with point geometries
        service_name: name of public service
        calc_pop_density: If True, divide service count by pop density.
            Otherwise, divide by population
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

    # Reproject point
    point_gdf = reproject_gdf(point_gdf, epsg_code)

    # Add number of service points for each polygon
    gdf_copy = add_point_count_column(polygon_gdf=gdf_copy,
                                      point_gdf=point_gdf,
                                      count_colname=count_colname,
                                      join_col='WARD_NO')

    # Calculate and add PCEN_Mobile column
    gdf_copy = calc_pcen_mobile_no_neighbors(polygon_gdf=gdf_copy,
                                            count_colname=count_colname,
                                        pcen_mobile_colname=pcen_mobile_colname,
                                        calc_pop_density = calc_pop_density,
                                        pop_colname='Total Population',
                                        id_col='WARD_NO')

    # Calculate and add service index column
    gdf_copy = calc_service_index(gdf_copy,
                                    pcen_mobile_colname=pcen_mobile_colname,
                                    service_idx_colname=service_idx_colname)

    # Drop additional columns
    # gdf_copy = gdf_copy.drop(columns=[pcen_mobile_colname, count_colname])

    return gdf_copy


def create_service_index_buffer(polygon_gdf, point_gdf, service_name,
    calc_pop_density, epsg_code):
    """Create service index for colonies with buffer (no neigbors)

    Args:
        polygon_gdf: GeoDataFrame with polygon geometries
        point_gdf: GeoDataFrame with point geometries
        service_name: name of public service
        calc_pop_density: If True, divide service count by pop density.
            Otherwise, divide by population
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

    # Reproject point
    point_gdf = reproject_gdf(point_gdf, epsg_code)

    # Add number of service points for each polygon
    gdf_copy = add_point_count_column(polygon_gdf=gdf_copy,
                                      point_gdf=point_gdf,
                                      count_colname=count_colname,
                                      join_col='USO_AREA_U')

    # Calculate and add PCEN_Mobile column
    gdf_copy = calc_pcen_mobile_no_neighbors(polygon_gdf=gdf_copy,
                                            count_colname=count_colname,
                                        pcen_mobile_colname=pcen_mobile_colname,
                                        calc_pop_density = calc_pop_density,
                                        pop_colname='population',
                                        id_col='USO_AREA_U')

    # Calculate and add service index column
    gdf_copy = calc_service_index(gdf_copy,
                                    pcen_mobile_colname=pcen_mobile_colname,
                                    service_idx_colname=service_idx_colname)

    # Drop additional columns
    # gdf_copy = gdf_copy.drop(columns=[pcen_mobile_colname, count_colname])

    return gdf_copy


def calc_point_services_wards(polygon_gdf, point_services, calc_pop_density,
    epsg_code):
    """Calculates all point services in Wards"""

    separator = '--------------------------------------------------------'

    for point_service in point_services:
        polygon_gdf = create_service_index_wards(polygon_gdf=polygon_gdf,
                                        point_gdf=point_services[point_service],
                                        service_name=point_service,
                                        calc_pop_density=calc_pop_density,
                                        epsg_code=epsg_code)
        print('{} service index is completed'.format(point_service))
        print(separator)

    print('all point services completed')

    return polygon_gdf

def calc_point_services_buffer(polygon_gdf, point_services, calc_pop_density,
    epsg_code):
    """Calculates all point services in colonies with buffer"""

    separator = '--------------------------------------------------------'

    for point_service in point_services:
        polygon_gdf = create_service_index_buffer(polygon_gdf=polygon_gdf,
                                        point_gdf=point_services[point_service],
                                        service_name=point_service,
                                        calc_pop_density=calc_pop_density,
                                        epsg_code=epsg_code)
        print('{} service index is completed'.format(point_service))
        print(separator)

    print('all point services completed')

    return polygon_gdf

def calc_point_services(polygon_gdf, point_services, epsg_code,
    pcen_denom, nbr_dist_colname):
    """Calculates all point services"""

    separator = '--------------------------------------------------------'

    for point_service in point_services:
        polygon_gdf = create_service_index(polygon_gdf=polygon_gdf,
                                        point_gdf=point_services[point_service],
                                        service_name=point_service,
                                        epsg_code=epsg_code,
                                        pcen_denom = pcen_denom,
                                        nbr_dist_colname=nbr_dist_colname)
        print('{} service index is completed'.format(point_service))
        print(separator)

    print('all point services completed')

    return polygon_gdf

def create_overall_psi(colonies_gdf):
    """Create Overall PSI across all indices (unnormalized and normalized [0,1])"""

    # Create list of all index columns
    idx_columns = [column for column in colonies_gdf.columns if column.endswith('_idx')]

    # Calculate simple average of all index columns and put in `unnorm_psi` column
    colonies_gdf['unnorm_psi'] = colonies_gdf[idx_columns].mean(axis=1)

    # Calculate normalized index [0,1] only for rows that are to be
    # included in the calculation
    colonies_gdf = calc_service_index(colonies_gdf, 'unnorm_psi', 'norm_psi')

    return colonies_gdf

def calc_all_services(polygon_gdf, point_services, line_services, epsg_code,
    pcen_denom, nbr_dist_colname):
    """Calculate all public services indices (point and line)"""

    # Get all point services
    polygon_gdf = calc_point_services(polygon_gdf, point_services, epsg_code,
                    pcen_denom, nbr_dist_colname)


    for line_service in line_services:
        polygon_gdf = create_service_length_index(polygon_gdf,
                                                  line_services[line_service],
                                                  line_service,
                                                  epsg_code,
                                                  nbr_dist_colname,
                                                  pcen_denom)

        print('{} service is completed'.format(line_service))

    polygon_gdf = polygon_gdf.rename(columns={'road_count':'road_length'})

    polygon_gdf = create_overall_psi(polygon_gdf)

    return polygon_gdf

def calc_all_services_wards(polygon_gdf, point_services, line_services,
    epsg_code, calc_pop_density):
    """Calculate all public services indices (point and line) for Wards"""

    # Get all point services
    polygon_gdf = calc_point_services_wards(polygon_gdf, point_services,
                    calc_pop_density, epsg_code)


    for line_service in line_services:
        polygon_gdf = create_service_length_index_wards(polygon_gdf,
                                                  line_services[line_service],
                                                  line_service,
                                                  epsg_code,
                                                  calc_pop_density)

        print('{} service is completed'.format(line_service))

    return polygon_gdf

def calc_all_services_buffer(polygon_gdf, point_services, line_services,
    calc_pop_density, epsg_code):
    """Calculate all public services indices (point and line) for buffer"""

    # Get all point services
    polygon_gdf = calc_point_services_buffer(polygon_gdf, point_services,
                    calc_pop_density, epsg_code)


    for line_service in line_services:
        polygon_gdf = create_service_length_index_buffer(polygon_gdf = polygon_gdf,
                                                  line_gdf = line_services[line_service],
                                                  service_name = line_service,
                                                  calc_pop_density = calc_pop_density,
                                                  epsg_code = epsg_code)

        print('{} service is completed'.format(line_service))

    return polygon_gdf
