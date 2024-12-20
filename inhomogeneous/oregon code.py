# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:53:26 2024

@author: dell
"""


from datetime import datetime, timedelta

import time
import networkx as nx
import matplotlib.pyplot as plt
import math
import random

# CDF distance between two lists A and B
def CDFdist(A, B):
    if len(A) == 0 or len(B) == 0:
        return 10000
    L = []
    inc = [1/len(A), -1/len(B)]
    for i in A:
        L.append([i,0])
    for i in B:
        L.append([i,1])
    L.sort()
    x = 0
    pre = 0
    ans = 0
    for pair in L:
        ans += abs(x)*(pair[0]-pre)
        x += inc[pair[1]]
        pre = pair[0]
    return ans

# Balls-into-bins distance between two lists A and B, with the width of bins being r
def BINdist(A, B, r):
    if len(A) == 0 or len(B) == 0:
        return 10000
    L = []
    inc = [1, -1]
    for i in A:
        L.append([i-r, 0])
        L.append([i+r, 1])
    for i in B:
        L.append([i-r, 1])
        L.append([i+r, 0])
    L.sort()
    x = 0
    pre = 0
    ans = 0
    for pair in L:
        ans += abs(x)*(pair[0]-pre)
        x += inc[pair[1]]
        pre = pair[0]
    return ans

# Given a vertex i in the first graph, return the vertex v in the second graph 
# with the minimum CDF distance. Here vertices are the vertex set of the second
# graph. Profile1, Profile2 are degree profiles of the first and the second 
# graph respecticely.   
def matchCDF(vertices, Profile1, Profile2, i):
    arg = []
    cur_min = 10**5
    arg_min = -1
    for k in vertices:
        new = CDFdist(Profile1[i], Profile2[k])
        if new < cur_min:
            cur_min = new
            arg_min = k
            arg += [k]
    return arg_min

# Given a vertex i in the first graph, return the vertex v in the second graph 
# with the minimum Balls-into-bins distance. Here vertices are the vertex set 
# of the second graph. Profile1, Profile2 are degree profiles of the first and 
# the second graph respecticely. The width of bins are r.
def matchBIN(vertices, Profile1, Profile2, i, r):
    cur_min = 10**5
    arg_min = -1
    for k in vertices:
        new = BINdist(Profile1[i], Profile2[k], r)
        if new < cur_min:
            cur_min = new
            arg_min = k
    return arg_min

def permute(G,vertices, perm):
    G_perm = nx.Graph()
    G_perm.add_nodes_from(vertices)
    # Permute each edge.
    for a, b in G.edges():
        new_a = perm[a]
        new_b = perm[b]
        G_perm.add_edge(new_a, new_b)
    return G_perm

# Given a graph G, generate the degree profiles of G. 
def profile(G, vertices, radius):
    # The degrees of verties in G.
    dG = dict(nx.degree(G))
    rootdG = {i: math.sqrt(f_i) for i, f_i in dG.items()}
    # In ProfileG, we take the square roots of degrees.
    ProfileG = {i:[rootdG[j] for j in G.neighbors(i)] for i in vertices}
    # In ProfilerefG, we do not take square roots.
    ProfileGref = {i:[dG[j] for j in G.neighbors(i)] for i in vertices}
    ProfileGdisc = {i:[rootdG[j]//(2*radius) for j in G.neighbors(i)] for i in vertices}
    return ProfileG, ProfileGref, ProfileGdisc


# Starting date (March 31)
start_date = datetime(2001, 3, 31)
# Generate 7 consecutive Sundays, each as a string in "month"+"date" format
days = [(start_date + timedelta(weeks=i)).strftime("%m%d") for i in range(9)]
# The list of all nine graphs
graphs = []
# Read all the files
for i in range(9):
    file = open('oregon1_01' + days[i] + '.txt', 'r')
    content = file.readlines()
    file.close()
    
    # Translate a file into a graph
    l = len(content)
    G = nx.Graph()
    # Each line in file represents an edge
    for i in range(4,l):
        # Add an edge into G
        a,b = map(int, content[i].split('\t'))
        G.add_edge(a,b)
    graphs.append(G)

        
vertex_sets = [set(G.nodes()) for G in graphs]
# Generate common vertices of all nine graphs
common_vertices = set.intersection(*vertex_sets)
# The subgraph of the first subgraph on the common vertices
graphs[0] = graphs[0].subgraph(common_vertices)
# The degrees of the vertices in the above subgraph
dg = dict(nx.degree(graphs[0]))

# Consider the subgraphs generated by some common vetices. The number of common
# vertices is denoted by threshold.
threshold = 10000
topv = [key for key, value in sorted(dg.items(), key=lambda item: item[1], reverse=True)[:threshold]]
# Generate the subgraphs of these common vertices
for i in range(9):
    graphs[i] = graphs[i].subgraph(topv)


# Consider the subgraphs FURTHER generated by some HIGH-DEGREE VERTICES among 
# the above common vertices(whose number is threshold). The number of high-degree
# vertices is denoted by bar.
bar = 1000


# The set of random permutations. We set the first element to be identity, and 
# the indices will match those of graphs.
Perm = [dict(zip(topv, topv))]
# Generate random permutations
for G in graphs[1:]:
    # A random permuted list of topv
    vperm = random.sample(topv, threshold)
    # Generate the random permutation 
    perm = dict(zip(topv, vperm))
    Perm.append(perm)
    

# The radius of discrete bins
radius = 0.5

     
# The degree profile for the nine graphs
Profile = []
# Generate the degree profiles of randomly permuted graphs.
for i in range(9):
    # Permute the i-th graph using the i-th permutation.
    G = graphs[i]
    G_perm = permute(G, topv, Perm[i])
    ProfileG, ProfileGref, ProfileGdisc = profile(G_perm, topv, radius)
    Profile.append([ProfileG, ProfileGref, ProfileGdisc])


# Two plots regarding the whole graph and the high-degree subgraph
fig, ax_top = plt.subplots(); fig, ax_high = plt.subplots()
ax_top.set_title("Oregon whole graph"); ax_high.set_title("Oregon high-degree subgraph")
ax_top.set_xlabel("Dates"); ax_top.set_ylabel("Fraction of correctly matched vertices")
ax_high.set_xlabel("Dates"); ax_high.set_ylabel("Fraction of correctly matched vertices")

# Testing the best width for balls-into-bins algorithm.
# The list of different widths.
Width = [0.5, 1, 2]

# Run the algorithm for each width
for r in Width:
    # Record the starting time
    t_start = time.time()
    # The following is the similar as before.
    # Results for BIN distances of all vertices and high-degree vertices
    # (with width being r)
    BINresults_top = []; BINresults_high = []
    # Obtain the distances    
    for i in range(9):
        cnt_top = 0; cnt_high = 0
        for j in range(threshold):
            u = topv[j]
            # Width = r
            v = matchBIN(topv, Profile[0][0], Profile[i][0], u, r)
            if v == Perm[i][u]:
                cnt_top += 1
                if j < bar:
                    cnt_high += 1
        # Fraction of vertices and high-degree vertices being matched
        BINresults_top.append(cnt_top/threshold); BINresults_high.append(cnt_high/threshold); 
    # First coordinate is the date, second coordinate is the fraction.
    print(f"The results of the whole graph for r={r} are", BINresults_top)
    ax_top.plot(days, BINresults_top, marker="o", label = f"r={r}")
    print(f"The results of the hd-subgraph for r={r} are", BINresults_high)
    ax_high.plot(days, BINresults_high, marker="o", label = f"r={r}")
    # Record the ending time
    t_end = time.time()
    print(f"The time for r={r} is {t_end - t_start}")
    print()


# Record the starting time
t_start = time.time()
# Results for CDF distances of all vertices and high-degree vertices.
CDFresults_top = []; CDFresults_high = []
# Obtain the distances    
for i in range(9):
    cnt_top = 0; cnt_high = 0
    for j in range(threshold):
        u = topv[j]
        v = matchCDF(topv, Profile[0][0], Profile[i][0], u)
        if v == Perm[i][u]:
            cnt_top += 1
            if j < bar:
                cnt_high += 1
    # Fraction of vertices and high-degree vertices being matched
    CDFresults_top.append(cnt_top/threshold); CDFresults_high.append(cnt_high/threshold)
# First coordinate is the date, second coordinate is the fraction.
print("The results of the whole graph for cdf are", CDFresults_top)
ax_top.plot(days, CDFresults_top, marker="x", label = "CDF")
print("The results of the hd-subgraph for cdf are", CDFresults_high)
ax_high.plot(days, CDFresults_high, marker="x", label = "CDF")
# Record the ending time
t_end = time.time()
print(f"The time for cdf is {t_end - t_start}")
print()


# Record the starting time
t_start = time.time()
# Results for CDF distances of all vertices and high-degree vertices.
CDFrefresults_top = []; CDFrefresults_high = []
# Obtain the distances    
for i in range(9):
    cnt_top = 0; cnt_high = 0
    for j in range(threshold):
        u = topv[j]
        v = matchCDF(topv, Profile[0][1], Profile[i][1], u)
        if v == Perm[i][u]:
            cnt_top += 1
            if j < bar:
                cnt_high += 1
    # Fraction of vertices and high-degree vertices being matched
    CDFrefresults_top.append(cnt_top/threshold); CDFrefresults_high.append(cnt_high/threshold)
# First coordinate is the date, second coordinate is the fraction.
print("The results of the whole graph for ref are", CDFrefresults_top)
ax_top.plot(days, CDFrefresults_top, marker="d", label = "ref(=(DP))")
print("The results of the hd-subgraph for ref are", CDFrefresults_high)
ax_high.plot(days, CDFrefresults_high, marker="d", label = "ref(=(DP))")
# Record the ending time
t_end = time.time()
print(f"The time for ref(=DP) is {t_end - t_start}")
print()

# Record the starting time
t_start = time.time()
# Results for CDF distances of all vertices and high-degree vertices.
discresults_top = []; discresults_high = []
# Obtain the distances    
for i in range(9):
    cnt_top = 0; cnt_high = 0
    for j in range(threshold):
        u = topv[j]
        v = matchCDF(topv, Profile[0][2], Profile[i][2], u)
        if v == Perm[i][u]:
            cnt_top += 1
            if j < bar:
                cnt_high += 1
    # Fraction of vertices and high-degree vertices being matched
    discresults_top.append(cnt_top/threshold); discresults_high.append(cnt_high/threshold)
# First coordinate is the date, second coordinate is the fraction.
print(f"The results of the whole graph for disc({radius}) are", discresults_top)
ax_top.plot(days, discresults_top, marker="+", label = f"disc({radius})")
print(f"The results of the hd-subgraph for disc({radius}) are", discresults_high)
ax_high.plot(days, discresults_high, marker="+", label = f"disc({radius})")
# Record the ending time
t_end = time.time()
print(f"The time for disc({radius}) is {t_end - t_start}")


ax_top.legend(); ax_high.legend()
plt.show()

"""

# Results for BIN distances with width being 0.5
BINresults_1 = []
# Obtain the distances    
for i in range(9):
    cnt = 0
    for j in topv:
        # Width = 0.5
        k = matchBIN(topv, Profile[0][0], Profile[i][0], j, 0.5)
        if k == j:
            cnt += 1
    # Fraction of vertices being matched
    BINresults_1.append(cnt/threshold)
# First coordinate is the date, second coordinate is the fraction.
ax.plot(days, BINresults_1, marker="o", label = "r=0.5")
t3 = time.time()
print(BINresults_1)
print(t3-t2)
          
# Results for BIN distances with width being 1
BINresults_2 = []
# Obtain the distances    
for i in range(9):
    cnt = 0
    for j in topv:
        # Width = 1
        k = matchBIN(topv, Profile[0][0], Profile[i][0], j, 1)
        if k == j:
            cnt += 1
    # Fraction of vertices being matched
    BINresults_2.append(cnt/threshold)
# First coordinate is the date, second coordinate is the fraction.
ax.plot(days, BINresults_2, marker="o", label = "r=1")
t4 = time.time()
print(BINresults_2)
print(t4-t3)

ax.legend()
"""
    
    
        



    
    
    