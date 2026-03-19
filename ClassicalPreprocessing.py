"""
This module imports the dataframe (excel) and loads data.
It creates a dict representation of the adjacency matrix as:
{place_i: {neighbour_1: cost_1, neighbour_2: cost_2, ...}, ...}

Prepare the adjacency matrix with penalties 100 for unconnected


It gives a way to generate the adjacency matrix given the dict

    generate_adjacency_matrix(ClassicalPreprocessing.adjacency_matrix)

A way to generate the limited adjacency dict, ex.

    filtered_dict = limited_adjacency(max_gear="Urban", time="Summer")

And a way to reindex the limited adjacency which is required to make its adj. matrix

    relabeled_filt_dict, relabeling = reindex_dict(filtered_dict)

And a way to draw them with

    draw_benasque_graph(adjacency_dict)
"""

import numpy as np
import pandas as pd


Nplaces = 25
benasque_data = pd.read_excel("Hackathon - data.xlsx")[:Nplaces]
# some are nan otherwise

place_to_index = {place: idx+1 for idx, place in enumerate(benasque_data["Place"])}
index_to_place = {idx+1: place for idx, place in enumerate(benasque_data["Place"])}


# Adjacency notation:
# adjacency[i] = {neighbour1:  cost1, neighbour2: cost2, ...]
# costs in minutes
adjacency_dict_given = {
				  1: {20: 4*60+50, 24: 4*60+35}, 
				  2: {8: 4*60+20}, 
				  3: {4: 1*60+25, 5: 45, 8: 0, 9: 0, 16: 3*60+10, 17: 0, 20: 0, 25: 1*60},
				  4: {5: 1*60+30, 7: 0, 8: 0, 9: 0, 11: 1*60+10, 17:0, 20: 0},
				  5: {8: 0, 9: 0, 17:0, 20: 0},
				  6: {9: 2*60+25, 12: 1*60},
				  7: {},
				  8: {15: 3*60+35, 20: 0},
				  9: {20: 0},
				  10: {20: 1*60, 21: 1*60+50},
				  11: {},
				  12: {},
				  13: {20: 4*60+20},
				  14: {19: 3*60+40, 21: 5*60+50},
				  15: {},
				  16: {18: 2*60+30},
				  17: {},
				  18: {},
				  19: {},
				  20: {21: 4*60+20, 22: 3*60+5},
				  21: {24: 1*60+35},
				  22: {23: 1*60+15},
				  23: {24: 50},
				  24: {},
				  25: {}}

## Add opposite paths (if i -> j,  j -> i)
adjacency_dict = {i: {} for i in adjacency_dict_given}

for place_i in adjacency_dict_given:
	for place_j in adjacency_dict_given[place_i]:
		adjacency_dict[place_i][place_j] = adjacency_dict_given[place_i][place_j]
		adjacency_dict[place_j][place_i] = adjacency_dict_given[place_i][place_j]



# CONVERSION FUNCTIONS BETWEEN DIFFERENT REPRESENTATIONS OF ADJACENCY

############
# We build adjacency matrix representation, with -1 to indicate non connected
############
def generate_adjacency_matrix(adjacency_dict):
    Nplaces = len(adjacency_dict)
    adjacency_matrix = np.zeros((Nplaces,Nplaces), dtype=np.float32)

    for i in range(Nplaces):
        for j in range(i, Nplaces):
            adjacency_matrix[i,j] = adjacency_dict[i+1].get(j+1,-1)
            adjacency_matrix[j,i] = adjacency_dict[i+1].get(j+1,-1)

    return adjacency_matrix



############
##### WE BUILD EDGE LIST
############


def generate_edge_list(adjacency_dict_given):
    # [(i, j, weight), ...]
    edge_list = []

    for place_i in adjacency_dict_given:
        for place_j in adjacency_dict_given[place_i]:
            edge_list.append((place_i, place_j, adjacency_dict_given[place_i][place_j]))

    return edge_list


def adjacency_dict_from_edge_list(edge_list):
    res_adjacency_dict = {i: {} for i in set([i for i, j, w in edge_list])} # + [j for i, j, w in edge_list])}  we don't need to check j due to symmetry
    for i, j, w in edge_list:
        res_adjacency_dict[i][j] = w
    return res_adjacency_dict




#### GRAPH DRAWING

# Graph using networkx
import networkx as nx
import matplotlib.pyplot as plt

def draw_benasque_graph(adjacency_dict):
    bias = 60 # bias to account for 0 distance

    edge_list = generate_edge_list(adjacency_dict)

    G = nx.Graph()
    for u, v, weight in edge_list:
        G.add_edge(u, v, weight=weight+bias)

    G = nx.relabel_nodes(G, index_to_place)

    pos = nx.spring_layout(G)
    colors = ["r" for node in G.nodes()]

    print()

    def draw_graph(G, colors, pos):
        plt.figure(figsize=(20, 14))
        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
        edge_labels = {edge: weight-bias for edge, weight in nx.get_edge_attributes(G, "weight").items()}
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    draw_graph(G, colors, pos)
    
    plt.show()





#### FILTERING



def get_gear(place_index, time):
    if time == "Summer":
        return benasque_data.loc[benasque_data["Index"] == place_index, "Summer gear"].values[0]
    else:
        return benasque_data.loc[benasque_data["Index"] == place_index, "Winter gear"].values[0]
    

gear_tiers = {"Urban": 0, "Trail": 1, "Mountain": 2, "Snow": 3}
def gear_check(gear, max_gear):
    return gear_tiers[gear] <= gear_tiers[max_gear]



def limited_adjacency(adjacency_dict=adjacency_dict, max_gear = "Urban", time = "Summer"):
    # Filter out nodes that exceed the maximum difficulty
    edge_list = generate_edge_list(adjacency_dict)
    filtered_edge_list = [(i, j, w) for i, j, w in edge_list if 
                          (gear_check(get_gear(i, time), max_gear) and gear_check(get_gear(j, time), max_gear))]
    
    return adjacency_dict_from_edge_list(filtered_edge_list)




#### RELABEL INDEXING

# filtered_adjacency = limited_adjacency(adjacency_dict, max_gear="Urban", time="Summer")

def reindex_dict(adjacency_dict):
    relabeling = {v: i+1 for i, v in enumerate(adjacency_dict)}
    return \
        {relabeling[old]: {relabeling[oldn]: weight for oldn, weight in neighbours.items()} for old, neighbours in adjacency_dict.items()},\
        relabeling

# for i in filtered_adjacency:
#     print(index_to_place[i])

# print(generate_adjacency_matrix(reindex_dict(filtered_adjacency)[0]))
# draw_benasque_graph(filtered_adjacency)



def generate_adjacency_matrix_hours_penalized(adjacency_dict):
    Nplaces = len(adjacency_dict)
    adjacency_matrix = np.zeros((Nplaces,Nplaces), dtype=np.float32)

    for i in range(Nplaces):
        for j in range(i, Nplaces):
            adjacency_matrix[i,j] = adjacency_dict[i+1].get(j+1,100*60)/60
            adjacency_matrix[j,i] = adjacency_dict[i+1].get(j+1,100*60)/60

    return adjacency_matrix


def prepare_adjacency_matrix(max_gear="Urban", time="Summer"):
    filtered_adjacency = limited_adjacency(adjacency_dict, max_gear=max_gear, time=time)
    filtered_adjacency, relabeling  = reindex_dict(filtered_adjacency)
    return generate_adjacency_matrix_hours_penalized(filtered_adjacency), {new-1: old for old, new in relabeling.items()}