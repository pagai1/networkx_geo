#!/usr/bin/python
# This script reads a edgelist of a deezer-database-extract
# 
import csv
import networkx as nx
import matplotlib.pyplot as plt
import time
import json
import signal
# for heuristic function
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.realpath(__file__),"../../networkx_modules")))
from helpers.generalStuff import *
from helpers.networkx_load_n_save import *
from helpers.search_functions import *
from algoPackage.pageRank import *
from algoPackage.simRank import *
from algoPackage.hits import *
from algoPackage.shortestPath import *
from algoPackage.jaccard_coefficient import *
from algoPackage.degree_centrality import *

from builtins import len
from networkx.algorithms.coloring.greedy_coloring_with_interchange import Node
from networkx.classes.function import get_node_attributes
from networkx.readwrite import json_graph;
from _operator import itemgetter
from matplotlib.pyplot import plot

#def to_ms(time):
#    return ("%.3f" % time)

def cleanupAll(tmpfilepath):
    print("CLEANING UP.")
    os.remove(tmpfilepath)

def draw_graph(G):
    pos = nx.spring_layout(G)
    #edge_labels = dict([((n1, n2), f"{n3['weight']") for n1, n2, n3 in G.edges])
#    nx.draw_networkx(G, pos, edge_color="blue")
    #### FIRST WAY TO GET EDGELABELS
   # edge_labels = dict([((n1, n2), G[n1][n2]["weight"]) for n1, n2 in G.edges])
   # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, connectionstyle="arc3,rad=0.1", edge_color="red", label=nx.get_edge_attributes(G,'weight'))
    
    #### SECOND WAY TO GET EDGELABELS
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G,'weight'), label_pos=0.25, horizontalalignment="left")
        
    #nx.draw_kamada_kawai(G,with_labels=True)
    #plt.plot()
    plt.show()

#
# MAIN
#
#
#edgelistfile='/home/pagai/graph-data/deezer_clean_data/both.csv'
## edgelist    
edgelistfile='/home/pagai/graph-data/OSRM/final_edgelist.csv'
datafile='/home/pagai/graph-data/OSRM/final_semicolon.txt'
tmpfilepath = "/tmp/tmpfile.csv"
limit = 0
seclimit=1
operatorFunction="eq"
verbose=False
drawit=False
#catchable_sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
#for sig in catchable_sigs:
#    signal.signal(sig, tmpfilepath)  # Substitute handler of choice for `print`

if (len(sys.argv) == 1):
    if (verbose):
        print("NOTHING WAS GIVEN")
    limit = "all"
elif (len(sys.argv) == 2):
    limit = sys.argv[1]
    if (verbose):
        print("LOADING " + str(limit) + " LINES FROM " + edgelistfile)
elif (len(sys.argv) == 3):
    limit = sys.argv[1] 
    seclimit = sys.argv[2]
    if (verbose):
        print("LOADING " + str(limit) + " LINES FROM " + edgelistfile + " AND " + str(seclimit) + " DEGREE.")
elif (len(sys.argv) == 4):    
    limit = sys.argv[1] 
    seclimit = int(sys.argv[2])
    operatorFunction=sys.argv[3]
    if (verbose):
        print("LOADING " + str(limit) + " LINES FROM " + edgelistfile + " AND DEGREE " + operatorFunction + " " + str(seclimit))    

if limit != "all":
    cleanup = True
# get number of lines of file
    with open(edgelistfile) as f:
        allLines = [next(f) for x in range(int(limit))]
        tmpFile = open(tmpfilepath, 'w+')
        for line in allLines:
            tmpFile.write(line)
    tmpFile.close()
    edgelistfile = tmpfilepath
    

## CREATING GRAPH
start_time = time.time()
G = nx.read_weighted_edgelist(edgelistfile, comments="no comments", delimiter=",", create_using=nx.DiGraph(), nodetype=str)
for edge in (G.edges()):
    G.edges()[edge]['label'] = "HAS_ROAD_TO"
## ADDING PROPERTIES
with open(datafile, 'r') as data:
    reader = csv.reader(data, delimiter=';')
    for row in reader:
        G.nodes[row[0]]['name'] = row[1]
        G.nodes[row[0]]['y'] = row[2]
        G.nodes[row[0]]['x'] = row[3]
        

#for node in (G.nodes())
#G = nx.read_edgelist(edgelistfile, comments="no comments", delimiter=",", create_using=nx.DiGraph(), nodetype=str)

if (verbose):
    print("Load of " + limit + " finished in: " + to_ms(time.time() - start_time) + " s.")
    print(nx.info(G))
    print(G.nodes(data=True))
    print(G.edges(data=True))

############ ALGOS #############

#### SHORTEST PATH
#algo_shortest_path(G)
#algo_all_pairs_dijkstra(G,verbose=True,inputWeight='weight')
#algo_all_pairs_bellman_ford_path(G,verbose=True,inputWeight='weight')

#all_pairs_shortest_path(G)

#algo_all_pairs_shortest_path(G,verbose=False,inputWeight='weight')
#draw_all_shortest_path_for_single_node(G,"1")
#all_shortest_path_for_single_node(G,"12")


#### SHORTESTPATH ASTAR
#algo_all_pairs_shortest_path_astar(G,verbose=verbose)

#### PAGERANK

#algo_pagerank(G, "default",  weightInput=None, verbose=True)
#algo_pagerank(G, "numpy", weightInput=None, verbose=True)
algo_pagerank(G, "scipy", weightInput=None, verbose=True, maxLineOutput=10)

#### SIMRANK
#algo_simRank(G,verbose=True,max_iterations=1)

#### DEGREE CENTRALITY
# Degree Centrality - own
#verbose=True
#peng = sorted(G.degree, key=lambda x: x[1], reverse=True)
#if (verbose):
#    for bums in peng:
#        print(bums)

# Degree Centrality - native
#algo_degree_centrality(G, verbose=False)

#### HITS

#get_hits(G)

if (drawit):
    draw_graph(G)
