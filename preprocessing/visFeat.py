import cv2
import numpy as np
import networkx as nx
import os     
import matplotlib.pyplot as plt
# import sys
# import time
# from rtree import index
# import netCDF4

# from osgeo import ogr
# import pandas as pd
# from osgeo import gdal

# import geopandas as gpd
# from shapely.geometry import Point
# from pyRadar.PolSAR import polSARutils
# from skimage.feature.texture import graycomatrix, graycoprops
import pickle



def read_graphs_from_folder(folder_path):
    graphs = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)
            graphs.append(graph)
    return graphs


def draw_features_map(graph_data, feature_map_folder, graph_folder):
    input_file = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/sar_jpg/20170107.jpg"
    image = cv2.imread(input_file)
    
    for nFeat, feature_name in enumerate(os.listdir(feature_map_folder)):
        print(f"generating feature:{feature_name}")
        if feature_name == "intensity" or feature_name == "std":
            print(feature_name)
            for i, file_name in enumerate(os.listdir(graph_folder)):
                # if file_name == "20170112":
                    print(f"graph:{file_name}")
                    graph = graph_data[i]
                    ori_pos = nx.get_node_attributes(graph, 'pos')

                    # Flip y-coordinates to correct orientation
                    pos = {node: (x, image.shape[0] - y) for node, (x, y) in ori_pos.items()}
                    
                    # Plot the graph
                    plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100))
                    
                    intensities = nx.get_node_attributes(graph, feature_name)
                    # print(f"label = {labels}")
                    intensity_values = [intensity for node, intensity in intensities.items()]
                    print(intensity_values)
                    # intensity_values = [0 if intensity > 0.5 else intensity for intensity in intensity_values]

                    # Normalize intensity values
                    normalized_values = (intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values))

                    # Create a colormap
                    cmap = plt.get_cmap('Reds')
                        
                    # Plot the graph
                    plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100))
                    nx.draw_networkx_nodes(graph, pos, node_color=normalized_values, node_size=2, cmap=cmap)
                    plt.savefig(f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/densePreproc/feature_map/{feature_name}/{file_name}')
                    # plt.savefig(f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/denser/feature_map/{feature_name}/{file_name}')
                    print(f"image {file_name} saved in {feature_name}---------------------------------------------------")
                    plt.close()  # Close the plot to prevent it from being displayed
                    break
                    


# graphs_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/4Graphs/"
graphs_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/densePreproc/4Graphs"
feature_map_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/densePreproc/feature_map/"
# graphs_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/denser/4Graphs"
# feature_map_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/denser/feature_map/"

graphs = read_graphs_from_folder(graphs_folder)


# draw_features_map(graphs, feature_map_folder, graphs_folder)



def draw_single_features_map(graph, image, feature_map_folder, file_name):
    input_file = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/0jpg_sar_image/20170112_Band1.jpg"
    image = cv2.imread(input_file)
    
    for nFeat, feature_name in enumerate(os.listdir(feature_map_folder)):
        print(f"generating feature:{feature_name}")
        if feature_name == "intensity" or feature_name == "std":
            print(feature_name)
            
            print(f"graph:{file_name}")
            ori_pos = nx.get_node_attributes(graph, 'pos')

            # Flip y-coordinates to correct orientation
            pos = {node: (x, image.shape[0] - y) for node, (x, y) in ori_pos.items()}
            
            # Plot the graph
            plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100))
            
            intensities = nx.get_node_attributes(graph, feature_name)
            # print(f"label = {labels}")
            intensity_values = [intensity for node, intensity in intensities.items()]
            print(intensity_values)
            # intensity_values = [0 if intensity > 0.5 else intensity for intensity in intensity_values]

            # Normalize intensity values
            normalized_values = (intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values))

            # Create a colormap
            cmap = plt.get_cmap('Reds')
                
            # Plot the graph
            plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100))
            nx.draw_networkx_nodes(graph, pos, node_color=normalized_values, node_size=2, cmap=cmap)
            save_path = f'{feature_map_folder}{feature_name}/{file_name}'
            plt.savefig(save_path)
            # plt.savefig(f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/denser/feature_map/{feature_name}/{file_name}')
            print(f"image {file_name} saved in {save_path}---------------------------------------------------")
            plt.close()  # Close the plot to prevent it from being displayed
            # break
            
