
import os
import networkx as nx
import numpy as np
import cv2
import pickle

from visFeat import draw_single_features_map

def find_global_min_max(input_folder):
    global_min = float('inf')
    global_max = float('-inf')

    for filename in os.listdir(input_folder):
        graph_path = os.path.join(input_folder, filename)
        
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
            
        for node in graph.nodes:
            intensity = graph.nodes[node]['intensity']
            if intensity < global_min:
                global_min = intensity
            if intensity > global_max:
                global_max = intensity

    return global_min, global_max

def normalize_intensity(graphs, global_min, global_max):
    range_intensity = global_max - global_min
    
    for graph in graphs:
        for node in graph.nodes:
            if range_intensity != 0:
                normalized_intensity = (graph.nodes[node]['intensity'] - global_min) / range_intensity
            else:
                normalized_intensity = graph.nodes[node]['intensity']  # if all intensities are the same
            graph.nodes[node]['intensity'] = normalized_intensity
            graph.nodes[node]['label'] = 1 if normalized_intensity > 0.9 else 0

def process_graph_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    global_min, global_max = find_global_min_max(input_folder)
    print(f"min is {global_min}, max is {global_max}")

    for filename in os.listdir(input_folder):
        print(f"processing {filename}")
        input_graph_path = os.path.join(input_folder, filename)
        
        with open(input_graph_path, 'rb') as f:
            graph = pickle.load(f)
            
        normalize_intensity([graph], global_min, global_max)
        
        output_graph_path = os.path.join(output_folder, filename)
        with open(output_graph_path, 'wb') as f:
            pickle.dump(graph, f)


def modify_intensity(grpah):
    
input_folder = '/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/new_label/graphs/'
output_folder = '/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/new_label/graphs/'
# process_graph_folder(input_folder, output_folder)

# folder_path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/new_label/graphs/"
# normalize_intensity_relabel(folder_path)


input_file = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/0jpg_sar_image/20170112_Band1.jpg"

image = cv2.imread(input_file)

base_name = "20210316"
feature_map_folder = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/new_label/feature_map/"
input_graph_path = input_folder + base_name
with open(input_graph_path, 'rb') as f:
    graph = pickle.load(f)
            
modify_intensity(graph)
draw_single_features_map(input_folder, image, feature_map_folder, base_name)
