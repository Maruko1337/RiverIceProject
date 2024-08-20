import cv2
import numpy as np
import networkx as nx
import os     
import matplotlib.pyplot as plt
import statistics
import sys
import time
import re
import math
from rtree import index
from visFeat import draw_single_features_map
import shapefile
from shapely.geometry import shape, Point, Polygon

# import cupy as cp
# from superpixels import slic  # CUDA-accelerated SLICO implementation
# from normalize_intensity_relabel.py import *


from osgeo import ogr
import pandas as pd
from osgeo import gdal

import geopandas as gpd
from shapely.geometry import Point
# from pyRadar.PolSAR import polSARutils
from skimage.feature.texture import graycomatrix, graycoprops
import pickle
DEFAULT_DRIVER_NAME='ESRI Shapefile'



label_new = True
draw_ground_truth = True
draw_features = True
area_of_interest = "lake"
print(f"generating : {area_of_interest} region")

if area_of_interest == "beauharnois_canal":
    # HARD-CODED paths
    graphs_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/7relabel_graphs/"
    raster_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/output/data/beauharnois_canal/"
    data = {"Ice Coverage": [11111111], "Ice Count": [222222222222], "Total Node": [333333333]}    
    seg_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/9segments/"
    input_path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/sar_jpg/"
    shapefile_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/shapefiles/areas_of_interest/beauharnois_canal/ice/"
    feature_map_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/densePreproc/feature_map/"
    
    # ----------------------------
    ice_shapefile_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/shapefiles/areas_of_interest/lake_stlawrence/ice/"
    water_shapefile_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/shapefiles/areas_of_interest/lake_stlawrence/water/"

else:
    # HARD-CODED paths
    inc_angle_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/incident_angle_sheets/norm_filter/"
    input_path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/0jpg_sar_image/"
    graphs_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/new_label/graphs/"
    seg_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/8segments/"
    raster_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/output/data/lake_stlawrence/"
    feature_map_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/new_label/feature_map/"
    if label_new:
        ice_shapefile_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/shapefiles/areas_of_interest/lake_stlawrence/new_ice/"
        water_shapefile_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/shapefiles/areas_of_interest/lake_stlawrence/new_water/"
        local_feature_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/src/preprocessing/local_features/"
    else:
        ice_shapefile_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/shapefiles/areas_of_interest/lake_stlawrence/ice/"
        water_shapefile_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/shapefiles/areas_of_interest/lake_stlawrence/water/"

        
# Function to  coordinates based on geotransform
def transform_coordinates(lon, lat, geotransform):
    x_transformed= int((lon - geotransform[0]) / geotransform[1])
    y_transformed= int((lat - geotransform[3]) / geotransform[5])
    return x_transformed, y_transformed

def inverse_transform_coordinates(x_transformed, y_transformed, geotransform):
    lon = x_transformed* geotransform[1] + geotransform[0]
    lat = y_transformed* geotransform[5] + geotransform[3]
    return lon, lat

def save_graph(graph_data, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(graph_data, f)

# Extract GLCM texture features for each superpixel
def extract_textural_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    features = {}
    features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    features['energy'] = graycoprops(glcm, 'energy')[0, 0]
    features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    features['entropy'] = -np.sum(glcm * np.log(glcm + 1e-10))  # Add entropy calculation
    features['dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
    features['sum_of_squares_variance'] = graycoprops(glcm, 'ASM')[0, 0]
    return features

def visualize_ice_water_label(point, shapefile_path):
    # Read shapefile using geopandas
    print("Shapefile path:", shapefile_path)
    gdf = gpd.read_file(shapefile_path)

    # Create a GeoDataFrame for the point
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs=gdf.crs)

    # Plot the shapefile features
    ax = gdf.plot(figsize=(10, 10), color='lightblue')
    
    # Plot the point
    point_gdf.plot(ax=ax, color='red', markersize=100, label='Point', aspect=1)

    # Add legend and labels
    plt.legend(['Shapefile Features', 'Point'])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Visualization of Shapefile Features and Point')
    plt.savefig("/home/maruko/projects/def-ka3scott/maruko/seaiceClass/output/graphs/shapefile.jpg") # HARD-CODED path
    plt.close() 
    print("generated shapefile")
    
def get_inc_angle(date, x, y, raster):
    # Split the filename and check if it has the expected structure
    # filtered_angle-S1A_IW_GRDH_1SDV_20170107T224400_20170107T224425_014732_017FAE_9F15.csv
    # print(f"looking for incidence angle for date:{date}")
    # Define a regex pattern to match the date part in the filenames
    
    geotransform = raster.GetGeoTransform()
    # print(f"before coord: ({x}, {y})")
    x, y = inverse_transform_coordinates(x, y, geotransform)
    # print(f"coord: ({x}, {y})")
    
    pattern = re.compile(rf'.*_{date}T.*')
    # print(f"pattern is {pattern}")
    
    # Iterate over all files in the directory
    for filename in os.listdir(inc_angle_folder):
        # print(f"filename is {filename}")
        if filename.endswith('.csv'):  # Check if the file is a CSV file
            match = pattern.search(filename)
            if match:
                parts = filename.split('_')
                # print(f"parts = {parts}")
                if len(parts) > 5:
                    # Extract date from the filename
                    # date_str = parts[6][:8]  # Assuming date is in the 6th part of the split filename
                    # time_str = filename.split('_')[5][9:15]
                    
                    local_feature_path = inc_angle_folder+filename
                    df = pd.read_csv(local_feature_path)
                    # print(f"found date:{date_str}, time:{time_str}")
                    
                    # Compute the Euclidean distance to each point in the DataFrame
                    if x < -70:
                        df['Distance'] = np.sqrt((df['Latitude'] - y)**2 + (df['Longitude'] - x)**2)
                    else:
                        df['Distance'] = np.sqrt((df['Latitude'] - x)**2 + (df['Longitude'] - y)**2)

                    
                    # Find the row with the minimum distance
                    closest_row = df.loc[df['Distance'].idxmin()]
                    found_inc_angle = closest_row['Incidence_Angle']
                    
                    # print(f"point is ({x}, {y}), distance is {df['Distance']}, closest row is {closest_row}")
                    if found_inc_angle is None:
                        print(f"Error: incidence angle is None")
                    # else:
                        # print(f"incidence angle = {found_inc_angle}")
                    
                    # Return the incident angle
                    return found_inc_angle
                else:
                    print(f"parts = {parts} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     else:
        #         # print(f"ERROR: file does not match with the pattern")
        # else:
        #     # print(f"ERROR: file does not ends with csv")
    
    

    
    
# Function to determine ice or water label for a given point (centroid) based on shapefile of ice
def get_ice_water_label(date, raster, intensity_values, std_values):
    # Function to extract coordinates from a Shapely geometry
    def extract_coordinates(geom):
        if geom.geom_type == 'Polygon':
            return list(geom.exterior.coords)
        elif geom.geom_type == 'MultiPolygon':
            coords = []
            for poly in geom:
                coords.extend(list(poly.exterior.coords))
            return coords
        else:
            return []
    
    
    if label_new:
        shapefile_name = "_new_" + date + ".shp"
        ice_shapefile_path = f"{ice_shapefile_folder}ice{shapefile_name}"
        water_shapefile_path =  f"{water_shapefile_folder}water{shapefile_name}"
    else:
        shapefile_name = date + "_Polygon.shp"
        ice_shapefile_path =  os.path.join(ice_shapefile_folder, shapefile_name)
        water_shapefile_path =  os.path.join(water_shapefile_folder, shapefile_name)
   
    # driver = ogr.GetDriverByName(DEFAULT_DRIVER_NAME)
    
    # ice_shapefile = driver.Open(ice_shapefile_path, 0)  
    
    ice_shapefile = shapefile.Reader(ice_shapefile_path)
    
    print(f"ice shapefile path {ice_shapefile_path}") 
    # ice_layer = ice_shapefile.GetLayer()
    ice_label_list = []
    
    
    
    # water_shapefile = driver.Open(water_shapefile_path, 0)    
    # print(f"water shapefile path {water_shapefile_path}") 
    # water_layer = water_shapefile.GetLayer()
    
    water_shapefile = shapefile.Reader(water_shapefile_path)
    
    water_label_list = []
    
    
    

    # Function to transform polygon boundaries
    def transform_polygon(polygon, geotransform):
        transformed_coords = [transform_coordinates(lon, lat, geotransform) for lon, lat in polygon.exterior.coords]
        return Polygon(transformed_coords)

    
    geotransform = raster.GetGeoTransform()
    for shaperec in ice_shapefile.shapeRecords():
        
        geom = shape(shaperec.shape.__geo_interface__)
        transformed_geom = transform_polygon(geom, geotransform)
        # print(f"polygon: {transformed_geom}")
        ice_label_list.append(transformed_geom)
    
    for shaperec in water_shapefile.shapeRecords():
        geom = shape(shaperec.shape.__geo_interface__)
        transformed_geom = transform_polygon(geom, geotransform)
        
        water_label_list.append(transformed_geom)
    
    

    # for ice_feature in ice_layer:
        
    
    #     geometry = ice_feature.GetGeometryRef()
    #     min_lon, max_lon, min_lat, max_lat = geometry.GetEnvelope()
        
    #     geotransform = raster.GetGeoTransform()
    #     min_lon = int((min_lon - geotransform[0]) / geotransform[1])
    #     max_lon = int((max_lon - geotransform[0]) / geotransform[1])
    #     max_lat = int((max_lat - geotransform[3]) / geotransform[5])
    #     min_lat = int((min_lat - geotransform[3]) / geotransform[5])
        
    #     max_lat, min_lat = (min_lat, max_lat) if max_lat < min_lat else (max_lat, min_lat)
    #     max_lon, min_lon = (min_lon, max_lon) if max_lon < min_lon else (max_lon, min_lon)

        
    #     bias_v = 0
    #     bias_h = 0
    #     ice_label_list.append([max_lat + bias_v, min_lat - bias_v, max_lon + bias_h, min_lon - bias_h])
    
    # for water_feature in water_layer:
        
    
    #     geometry = water_feature.GetGeometryRef()
    #     min_lon, max_lon, min_lat, max_lat = geometry.GetEnvelope()
        
    #     geotransform = raster.GetGeoTransform()
    #     min_lon = int((min_lon - geotransform[0]) / geotransform[1])
    #     max_lon = int((max_lon - geotransform[0]) / geotransform[1])
    #     max_lat = int((max_lat - geotransform[3]) / geotransform[5])
    #     min_lat = int((min_lat - geotransform[3]) / geotransform[5])
        
    #     max_lat, min_lat = (min_lat, max_lat) if max_lat < min_lat else (max_lat, min_lat)
    #     max_lon, min_lon = (min_lon, max_lon) if max_lon < min_lon else (max_lon, min_lon)

        
    #     bias_v = 1
    #     bias_h = 1
    #     water_label_list.append([max_lat + bias_v, min_lat - bias_v, max_lon + bias_h, min_lon - bias_h])
    
    
    return ice_label_list, water_label_list


def get_local_feature(date):
    # print(f"looking for time for date:{date}")
    # Define a regex pattern to match the date part in the filenames
    pattern = re.compile(rf'.*_{date}T.*')
    # print(f"pattern is {pattern}")
    
    # Iterate over all files in the directory
    for filename in os.listdir(inc_angle_folder):
        # print(f"filename: {filename}")
        if filename.endswith('.csv'):  # Check if the file is a CSV file
            match = pattern.search(filename)
            
            if match:
                parts = filename.split('_')
                if len(parts) > 5:
                    # Extract date from the filename
                    date_mon_str = filename.split('_')[5][:8] # Assuming date is in the 6th part of the split filename
                    time_str = filename.split('_')[5][9:15]
                    # print(f"found date:{date_mon_str}, time:{time_str}")
                    for date_mon in os.listdir(local_feature_folder):
                        if date_mon_str.startswith(date_mon[:6]):
                            print(f"{date_mon_str} startswith({date_mon[:6]}):")
                            local_feature_path = local_feature_folder+date_mon
                            df = pd.read_csv(local_feature_path)
                            date_str_formatted = f"{int(date[6:])}"
                            # print(f"date formatted type = {type(date_str_formatted)}")
                            hour = int(time_str[:2])
                            time_str_formatted = f"{hour}:00"
                            # print(f"time formatted = {time_str_formatted}")
                            
                            # Initialize an empty list to collect the matching rows
                            matching_rows = []

                            # Loop through each row in the DataFrame
                            for index, row in df.iterrows():
                                # print(f"day is {row['Day']} type is {type(str(row['Day']))}, time is {row['Time (LST)']}")
                                # print(f"day match: {str(row['Day']) == date_str_formatted}, time match: {row['Time (LST)'] == time_str_formatted}")
                                if str(row['Day']) == date_str_formatted and row['Time (LST)'] == time_str_formatted:
                                    # print(f"found")
                                    matching_rows.append(row)
                                    break

                            # Convert the list of matching rows back to a DataFrame
                            filtered_df = pd.DataFrame(matching_rows)

                            # print(filtered_df)
                            # Filter rows based on the Date/Time (LST) and Time (LST) columns
                            # filtered_df = df[
                            #     (str(df['Day']) == date_str_formatted) & 
                            #     (df['Time (LST)'] == time_str_formatted)
                            # ]
                            print(filtered_df)
                            # Check if any rows are returned
                            if not filtered_df.empty:
                                # Return the value of "Rel Hum (%)"
                                return filtered_df['Rel Hum (%)'].values[0], filtered_df['Temp (°C)'].values[0], filtered_df['Wind Spd (km/h)'].values[0]
                            else:
                                print("Error: filtered df is empty")
                                return None
                        # else:
                            # print(f"{date_mon_str} does not startswith({date_mon[:6]}):")
                # else:
                #     print(f"length of filename {filename} is less than 6")
            # else:
            #     print(f"filename {filename} is not matched with pattern")
                            
                    
                    
    
def include_point(point, label_list):
    # x = point.x
    # y = point.y
    # for label_range in label_list:
    #     if y <= label_range[0] and y >= label_range[1] and x <= label_range[2] and x >= label_range[3]:
    #         return True
    
    point = Point((point[1], point[0]))
    # print(f"point = {point}")
    for polygon in label_list:
        if polygon.contains(point):
            # print(f"point {point} is inside")
            return True
    
    
    return False

def read_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image at {image_path}")
        return image
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_intensities(image, date, num_superpixels, segments):
    if area_of_interest != "lake":
        band4_path = f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/sar_jpg/{date}_Band4.jpg' # HARD-CODED path
        band5_path = f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/sar_jpg/{date}_Band5.jpg' # HARD-CODED path

        
        band4_image = read_image(band4_path)
        
        band5_image = read_image(band5_path)
        
        if band4_image is None or band5_image is None:
            print("Error: Unable to read the image.")
        else:
            print("Image successfully loaded.")
        
        b4_values = []
        b5_values = []
    
    # Calculate intensity for each superpixel
    intensity_values = []
    std_values = []
    
    for i in range(num_superpixels):
        mask = np.zeros_like(image)
        mask[segments.getLabels() == i] = 255  # Create a mask for each superpixel
        intensity = np.mean(image[segments.getLabels() == i])  # Calculate average intensity
        std = np.std(image[segments.getLabels() == i])
        
        
        if area_of_interest != "lake":
            b4 = np.mean(band4_image[segments.getLabels() == i])  # Calculate average intensity
            b5 = np.mean(band5_image[segments.getLabels() == i])  # Calculate average intensity
            if math.isnan(b4) or math.isnan(b5):
                print(f"intensity is nan in pixel {i}")
                b4 = 0.001 
                b5 = 0.001 
                
            b4_values.append(b4)
            b5_values.append(b5)
        
        
        # print(f"intensity = {intensity}")
        if math.isnan(intensity):
            print(f"intensity is nan in pixel {i}")
            intensity = 0.001   
            
            
        if math.isnan(std):
            print(f"std is nan in pixel {i}")
            std = 0.001  
            
        intensity_values.append(intensity)
        std_values.append(std)
        

    # Normalize intensity values
    # print(f"intensity_values = {intensity_values}")
    print(f"np.min(intensity_values) = {np.min(intensity_values)}")
    print(f"np.max(intensity_values) = {np.max(intensity_values)}")
    
    
    intensity_values = (intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values))
    
    # remove the outliers
    # the value 0.35 is based on experiment
    intensity_values = [0 if intensity > 0.35 else intensity for intensity in intensity_values]

    intensity_values = (intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values))
    

    mean_int = statistics.mean(intensity_values)
    print(f"mean intensity = {mean_int}")
    
    # Normalize the intensity to avoid unbalanced classes
    if mean_int > 0.06:
        print(f"mean_int / 0.2= {mean_int / 0.2}")
        intensity_values =  intensity_values / (mean_int / 0.03)
    elif mean_int < 0.03:
        print(f"mean_int < 0.02, ")
        intensity_values =  intensity_values * (0.03 / mean_int)

        
    std_values = (std_values - np.min(std_values)) / (np.max(std_values) - np.min(std_values))
    
    std_values = [0 if std > 0.9 else std for std in std_values]
    std_values = (std_values - np.min(std_values)) / (np.max(std_values) - np.min(std_values))
    
    
    if area_of_interest != "lake":
    
        print(f"b4_values = {b4_values}")
        print(f"np.min(b4_values) = {np.min(b4_values)}")
        print(f"np.max(b4_values) = {np.max(b4_values)}")
        print(f"b5_values = {b5_values}")
        print(f"np.min(b5_values) = {np.min(b5_values)}")
        print(f"np.max(b5_values) = {np.max(b5_values)}")
        b4_values = (b4_values - np.min(b4_values)) / (np.max(b4_values) - np.min(b4_values))
        b5_values = (b5_values - np.min(b5_values)) / (np.max(b5_values) - np.min(b5_values))
        
        b4_values = [0 if intensity > 0.9 else intensity for intensity in b4_values]
        b4_values = (b4_values - np.min(b4_values)) / (np.max(b4_values) - np.min(b4_values))

        b5_values = [0 if intensity > 0.9 else intensity for intensity in b5_values]
        b5_values = (b5_values - np.min(b5_values)) / (np.max(b5_values) - np.min(b5_values))

        print(f"intensity is : {intensity_values}")
        print(f"b4_values is : {b4_values}")
        print(f"b5_values is : {b5_values}")
        return intensity_values, b4_values, b5_values
    
    return intensity_values, std_values


# Step 1: Superpixel Segmentation and Node Creation
def create_superpixel_graph(image, date, raster_path):
    # Perform superpixel segmentation (e.g., SLIC)
    region_size = 7 # initially = 10
    segments = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size)
    # print(f"end slic")
    # segments = slic(image, region_size=region_size)

    segments.iterate()

    num_superpixels = segments.getNumberOfSuperpixels()
    superpixel_centers = {}  # Dictionary to store superpixel centers
    graph = nx.Graph()  # Create an empty graph
    # print(f"graph created")
    
    if area_of_interest != "lake":
        intensity_values, b4_values, b5_values = get_intensities(image, date, num_superpixels, segments)
    else:
        intensity_values, std_values = get_intensities(image, date, num_superpixels, segments)
    
    
    raster = gdal.Open(raster_path)
    ice_count = 0
    water_count = 0
    total_node = 0
    start_time = time.time()
    
    ice_list, water_list = get_ice_water_label(date, raster, intensity_values, std_values )
    hum, temp, wind = get_local_feature(date)
    
    valid_centers = []
    
    mask = []

    # add_node_feat(intensity_values, segments, graph, layer, raster)
    # Check if centroid of superpixel falls within ice or water region
    
    # print(f"number of nodes: {len(intensity_values)}")
    for i, intensity in enumerate(intensity_values):
        
        argwhere_result = np.argwhere(segments.getLabels() == i)
        if argwhere_result.size > 0:
            
            center = np.mean(argwhere_result, axis=0)
            superpixel_centers[i] = center
            valid_centers.append(center)  # Add this center to the list of valid centers
            # texture_features = extract_textural_features(image[segments.getLabels() == i])
        else:
            if valid_centers:
                superpixel_centers[i] = valid_centers[-1]
                # texture_features = extract_textural_features(image[segments.getLabels() == i - 1])
            else:
                print("no valid center exist")
                center = None
                continue
        
        
        
        point = Point(center[1], center[0])  # Create shapely Point from centroid
        # visualize_ice_water_label(point, shapefile_path)
        # print(f"start getting inc angle")
        # start_inc_angle_time = time.time()
        
        inc_angle = get_inc_angle(date, center[0], center[1], raster)
        # print(f"time to generate incidence angle is {time.time() - start_inc_angle_time}")
        
        intensity_threshold = 0.1
        # ice == 1, water == 0
        # print(f"intensity < threshold:{intensity_threshold} = {intensity}-----------------------------------")
        if intensity > intensity_threshold:
            if include_point(center, ice_list):
                ice_water_label = 1
                ice_count += 1
                mask.append(1)
                # print(f"point {point} is ice")
            elif include_point(center, water_list):
                ice_water_label = 1
                ice_count += 1
                mask.append(0)
            else:
                ice_water_label = 1
                mask.append(1)
        else:
            if include_point(center, water_list):
                ice_water_label = 0
                water_count += 1
                mask.append(1)
                # print(f"point {point} is ice")
            elif include_point(center, ice_list):
                ice_water_label = 1
                water_count += 1
                mask.append(0)
            else:
                ice_water_label = 0
                water_count += 1
                mask.append(1)
            
    
        # ice_water_label = get_ice_water_label(point, shapefile, raster, spatial_index)  # Determine ice or water label for the superpixel
        # if ice_water_label == "ice" : ice_count += 1
        total_node += 1
        
        # Add superpixel as a node with intensity, label, and texture features
        # graph.add_node(i, pos=(center[1], center[0]), intensity=intensity, std=std_values[i], label=ice_water_label, **texture_features)
        if area_of_interest != "lake":
            graph.add_node(i, pos=(center[1], center[0]), intensity=intensity, b4 = b4_values[i], b5 = b5_values[i], label=ice_water_label)
        else:
            graph.add_node(i, pos=(center[1], center[0]), intensity=intensity, std = std_values[i], incidence_angle=inc_angle, temperature = temp, humidity = hum, wind_spd = wind, label=ice_water_label, mask = mask[i])

        
        
    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"finished date: {date} with total nodes = {total_node}, ice node = {ice_count}, ice coverage = {ice_count/total_node}")
    return graph, superpixel_centers, segments

# Step 2: Connecting Adjacent Superpixels
def connect_adjacent_superpixels(graph, superpixel_centers, segments):
    height, width = segments.getLabels().shape[:2]
    labels = segments.getLabels()
    
    for y in range(height):
        for x in range(width):
            current_label = labels[y, x]
            if x > 0:
                left_label = labels[y, x - 1]
                if left_label != current_label:
                    graph.add_edge(current_label, left_label)
            if y > 0:
                top_label = labels[y - 1, x]
                if top_label != current_label:
                    graph.add_edge(current_label, top_label)

# Step 3: Remove Nodes Representing Land Areas (Example)
def remove_land_nodes(graph, superpixel_centers, image):
    # Example: Remove nodes representing land areas based on color thresholding
    for i, center in superpixel_centers.items():
        x, y = int(center[1]), int(center[0])
        if image[y, x, 0] > 200 and image[y, x, 1] > 200 and image[y, x, 2] > 200:
            graph.remove_node(i)


def generateNodeEdge(input_file, raster_path, date):
    # Load image
    image = cv2.imread(input_file)
    
    print("start generate step 1")
    # Step 1: Superpixel Segmentation and Node Creation
    graph, superpixel_centers, segments = create_superpixel_graph(image, date, raster_path)

    # Visualize superpixel segmentation
    
    if area_of_interest == "lake":
        output_path1 = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/1SuperpixelSeg/" + date + ".jpg" # HARD-CODED path
    else:
        output_path1 = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/densePreproc/1SuperpixelSeg/" + date + ".jpg" # HARD-CODED path
    print(f"start draw")
    # Draw superpixel contours
    contours = cv2.findContours(segments.getLabelContourMask(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)  # Draw all contours in red color with thickness 2
    cv2.imwrite(output_path1, image)
    
    print("start generate step 2")

    # Step 2: Connecting Adjacent Superpixels
    connect_adjacent_superpixels(graph, superpixel_centers, segments)

    # Visualize graph with superpixels
    
    if area_of_interest == "lake":
        output_path2 = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/2ConnetAdjSuperpixel/" + date + ".jpg" # HARD-CODED path
    else:
        output_path2 = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/densePreproc/2ConnetAdjSuperpixel/" + date + ".jpg" # HARD-CODED path
        # output_path2 = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/denser/2ConnetAdjSuperpixel/" + date + ".jpg" # HARD-CODED path
    
    plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100))  # Set figure size based on image dimensions
    pos = nx.get_node_attributes(graph, 'pos')  # Get node positions
    # Flip y-coordinates to correct orientation
    pos = {node: (x, image.shape[0] - y) for node, (x, y) in pos.items()}
       
    nx.draw(graph, pos, node_size=2)
    plt.title('Graph with Superpixels')
    plt.savefig(output_path2)
    plt.close()

    print("start generate step 3")
    # Step 3: Remove Nodes Representing Land Areas (Example)
    remove_land_nodes(graph, superpixel_centers, image)

    # Visualize final graph without land nodes
    if area_of_interest == "lake":
        output_path3 = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/3RemoveLandNode/" + date + ".jpg" # HARD-CODED path
    else: 
        output_path3 = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/densePreproc/3RemoveLandNode/" + date + ".jpg" # HARD-CODED path
        # output_path3 = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/denser/3RemoveLandNode/" + date + ".jpg" # HARD-CODED path

    # Draw the graph on a blank canvas
    blank_image = np.zeros_like(image)
    plt.figure(figsize=(blank_image.shape[1] / 100, blank_image.shape[0] / 100))  # Set figure size based on image dimensions
    pos = nx.get_node_attributes(graph, 'pos')  # Get node positions
    # Flip y-coordinates to correct orientation
    pos = {node: (x, image.shape[0] - y) for node, (x, y) in pos.items()}
    
    nx.draw(graph, pos, node_size=2)
    
    plt.title('Final Graph without Land Nodes')
    plt.savefig(output_path3, bbox_inches='tight')
    plt.close()
    return graph
    # plt.show()

def load_image(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File does not exist - {image_path}")
        return None

    # Attempt to read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}. Please check the file format and integrity.")
        return None
    
    print(f"successfully load image {image_path}")
    return image

count = 0
total_file_count = 0

# Save the current stdout
original_stdout = sys.stdout

# Specify the file path where you want to save the printed output
output_file_path = "Node_generate_info.txt"

# Open the file in write mode
with open(output_file_path, "w") as f:
    # Redirect stdout to the file
    sys.stdout = f
    
    print("hello")
    # Iterate through all files in the folder
    for file_name in os.listdir(input_path):
        f.flush()
        # if file_name.endswith("jpg"):
        modify_list = ["20170313"]
        modify_list2 = ["20210316", "20210220", "20210127", "20210115", "20210103", "20200321", "20200309", "20200226", "20200202", "20200109", "20190327"]
        for date in modify_list:
            if not file_name.startswith(date):
                print(f"start generating date {file_name}!!!!!!!!!!!!")

                date = file_name.split("_", 1)[-1]
                base_name, extension = os.path.splitext(file_name)
                date = re.search(r'\d{8}', base_name)
                date = date.group(0)
                # if not os.path.exists(os.path.join(graphs_folder, base_name)):
                print(f"generating {base_name}")
                
                input_file = os.path.join(input_path, file_name)
                # output_file = os.path.join(output_path, output_name)
                
                
                
                raster_name = date + ".nc"
                
                total_file_count += 1
                
                
                
                raster_path = os.path.join(raster_folder, raster_name)
                
                if area_of_interest == "beauharnois_canal":
                    shapefile_name = base_name + "_Polygon.shp"
                    shapefile_path =  os.path.join(shapefile_folder, shapefile_name)
                    print(shapefile_path)
                    # Check if the shapefile exists
                    if not os.path.exists(shapefile_path):
                        print(f"Shapefile {shapefile_name} not found. Skipping.")
                        count += 1
                        continue
                    graph_data = generateNodeEdge(input_file, shapefile_path, raster_path, base_name)
                    input_file = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/sar_jpg/20170107.jpg" # HARD-CODED path
                else:
                    base_name = file_name.split("_", 1)[0]
                    if label_new:
                        shapefile_name = "ice_new_" + base_name + ".shp"
                    else:
                        shapefile_name = base_name + "_Polygon.shp"
                    
                    shapefile_path =  os.path.join(ice_shapefile_folder, shapefile_name)
                    # Check if the shapefile exists
                    if not os.path.exists(shapefile_path):
                        print(f"Shapefile {shapefile_name} not found. Skipping.")
                        count += 1
                        continue
                    graph_data = generateNodeEdge(input_file, raster_path, base_name)
                    input_file = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/0jpg_sar_image/20170112_Band1.jpg" # HARD-CODED path
                
                
                
                
                save_graph(graph_data, graphs_folder, base_name)
                
                plt.close() 
                
                # Get node positions
                # pos = nx.get_node_attributes(G, 'pos')
                
                if not os.path.exists(input_file):
                    print(f"Error: File does not exist - {input_file}")
                else:
                    print("image exist")
                # Attempt to read the image
                image = cv2.imread(input_file)
                if image is None:
                    print(f"Failed to load image at {input_file}. Please check the file format and integrity.")
                else:
                    print(f"image is not none")
                # image = load_image(input_file)
                
                if draw_features:
                    print(f"start draw features")
                    draw_single_features_map(graph_data, image, feature_map_folder, base_name)
                    print(f"feature of date{base_name} is drawn")
                
                if draw_ground_truth:
                    labels = nx.get_node_attributes(graph_data, 'label')
                    mask = nx.get_node_attributes(graph_data, 'mask')
                    # print(f"label = {labels}")
                    node_colors = []
                    for name, label in labels.items():
                        # print(f"label is {label}")
                        # print(f"mask name = {mask[name]}")
                        if mask[name] == 0:
                            node_colors.append('gray')
                            # print("gray is drawn")
                        elif label == 1:
                            node_colors.append('yellow')
                        # elif mask[name] == 0:
                        #     node_colors.append('gray')
                        #     print("gray is drawn")
                        else:
                            node_colors.append('blue')
                        
                    # node_colors = ['blue' if label == 'water' else 'red' for label in labels]

                    ori_pos = nx.get_node_attributes(graph_data, 'pos')

                    # Flip y-coordinates to correct orientation
                    pos = {node: (x, image.shape[0] - y) for node, (x, y) in ori_pos.items()}
                    
                    # Plot the graph
                    plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100))
                    nx.draw(graph_data, pos, with_labels=False, node_color=node_colors, node_size=2)
                    plt.savefig(f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/new_label/ground_truth_int/{date}') # HARD-CODED path
                    
                    print(f"image {date} saved ---------------------------------------------------")
                    plt.close()  # Close the plot to prevent it from being displayed
                    print(f"ground truth of date{base_name} is drawn")
                    # break
                    
                    
                    

                
                print(f"finish generate {file_name}")
                # break

    print(f"number of not found shape file: {count}")
    print(f"number of total file: {total_file_count}")
    # Now all printed output will be written to the file
    print("Printed output will be saved to printed_output.txt")
    # process_graph_folder(graphs_folder)

    # Restore the original stdout
    sys.stdout = original_stdout