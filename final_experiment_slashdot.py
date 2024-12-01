# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:43:45 2024

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:41:03 2024

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:53:26 2024

@author: dell
"""


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

# Given a graph and a permutation, generate the permuted graph.
def permute(G,vertices, perm):
    G_perm = nx.Graph()
    G_perm.add_nodes_from(vertices)
    # Permute each edge.
    for a, b in G.edges():
        new_a = perm[a]
        new_b = perm[b]
        G_perm.add_edge(new_a, new_b)
    return G_perm
    
# Given a graph G, generate the degree profiles of G. The variable vertices
# denote the vertex set of G
def profile(G, vertices, radius):
    # The degrees of verties in G.
    dG = dict(nx.degree(G))
    rootdG = {i: math.sqrt(f_i) for i, f_i in dG.items()}
    # In ProfileG, we take the square roots of degrees.
    ProfileG = {i:[rootdG[j] for j in G.neighbors(i)] for i in vertices}
    # In ProfilerefG, we do not take square roots.
    ProfileGref = {i:[dG[j] for j in G.neighbors(i)] for i in vertices}
    # In ProfiledicsG, we consider discrete bins.
    ProfileGdisc = {i:[rootdG[j]//(2*radius) for j in G.neighbors(i)] for i in vertices}
    return ProfileG, ProfileGref, ProfileGdisc

# Given a parent graph, generate two independently subsampled graph,
# where in each subsampling process, each edge is independently maintained with
# probability sub_prob 
def subsample(ParGraph, sub_prob, perm, radius):
    vertices = list(ParGraph.nodes)
    G = nx.Graph()
    # G has the same vertex set as ParGraph
    G.add_nodes_from(vertices)
    for a, b in ParGraph.edges:
        # Subsampling each edge
        p = random.random()
        if p < sub_prob:
            G.add_edge(a, b)
    # Degree profiles of G
    ProfileG, ProfileGref, ProfileGdisc = profile(G, vertices, radius)
    H = nx.Graph()
    # H has the same vertex set as ParGraph
    H.add_nodes_from(vertices)
    for a, b in ParGraph.edges:
        # Subsampling each edge
        p = random.random()
        if p < sub_prob:
            H.add_edge(a, b)
    # Permute H
    H_perm = permute(H, vertices, perm)
    # Degree profiels of the oermuted H
    ProfileH, ProfileHref, ProfileHdisc = profile(H_perm, vertices, radius)
    return ProfileG, ProfileGref, ProfileGdisc, ProfileH, ProfileHref, ProfileHdisc

t0 = time.time()

# Read the file
file = open('Slashdot0902.txt', 'r')
content = file.readlines()
file.close()

# The number of vertices 
num = 75    
# Translate a file into a graph
l = len(content)
Par = nx.Graph()
# Each line in file represents an edge

for i in range(4,l):
    # Add an edge into G
    a,b = map(int, content[i].split('\t'))
    if a < num and b < num and a != b:
        Par.add_edge(a,b)

# Vertex set of Par.
vertices = list(range(num))


# The list of all the delta's. The subsampling probability will equal 1-delta
deltalist = [0.025*i for i in range(17)]

# The number of independent repetitions
rep = 10

# The radius of discrete bins
radius = 0.5

# The lists of random permutations. Different algorithms will be applied on the 
# same graphs with same random permutations
Perm = [[0 for _ in range(rep)] for _ in range(17)]
# The lists of ProfileG, ProfileGref, ProfileH, ProfileHref
PG = [[0 for _ in range(rep)] for _ in range(17)]
PGref = [[0 for _ in range(rep)] for _ in range(17)]
PGdisc = [[0 for _ in range(rep)] for _ in range(17)]
PH = [[0 for _ in range(rep)] for _ in range(17)]
PHref = [[0 for _ in range(rep)] for _ in range(17)]
PHdisc = [[0 for _ in range(rep)] for _ in range(17)]
# Generate random permutations and degree profiles
for i in range(17):
    delta = deltalist[i]
    # The subsampling probability is denoted by s.
    s = 1 - delta
    # Do rep times of independent experiments
    for j in range(rep):
        # Generate a random permutation
        perm = random.sample(range(num),num)
        Perm[i][j] = perm
        ProfileG, ProfileGref, ProfileGdisc, \
        ProfileH, ProfileHref, ProfileHdisc = subsample(Par, s, perm, radius)
        PG[i][j] = ProfileG; PGref[i][j] = ProfileGref; PGdisc[i][j] = ProfileGdisc
        PH[i][j] = ProfileH; PHref[i][j] = ProfileHref; PHdisc[i][j] = ProfileHdisc


fig, ax = plt.subplots()
ax.set_title("Slashdot network")
ax.set_xlabel("1-s")
ax.set_ylabel("Fraction of correctly matched vertices")
 

# The list of time costs
Time = []   
# Results for CDF distances
discresults = [] 
# Obtain the distances 
for i in range(17):
    # The starting time
    t_start = time.time()
    cnt = 0
    # Do rep times of independent experiments
    for j in range(rep):
        perm = Perm[i][j]
        ProfileGdisc = PGdisc[i][j]; ProfileHdisc = PHdisc[i][j]
        for u in vertices:
            v = matchCDF(vertices, ProfileGdisc, ProfileHdisc, u)
            if v == perm[u]:
                cnt += 1
    # Fraction of vertices being matched
    discresults.append(cnt/(num*rep))
    # The ending time
    t_end = time.time()
    Time.append(t_end - t_start)
    # First coordinate is the intensity of noise, second coordinate is the fraction.
ax.plot(deltalist, discresults, marker="+", label = f"disc({radius})")
print(f"The results for disc({radius}) are", discresults)

# The total time cost is denoted by T.
T = sum(Time)
print(f"The total time cost for disc({radius}) is {T}")
print()


# The list of time costs
Time = []   
# Results for CDF distances
CDFrefresults = []
# Obtain the distances 
for i in range(17):
    # The starting time
    t_start = time.time()
    cnt = 0
    # Do rep times of independent experiments
    for j in range(rep):
        perm = Perm[i][j]
        ProfileGref = PGref[i][j]; ProfileHref = PHref[i][j]
        for u in vertices:
            v = matchCDF(vertices, ProfileGref, ProfileHref, u)
            if v == perm[u]:
                cnt += 1
    # Fraction of vertices being matched
    CDFrefresults.append(cnt/(num*rep))
    # The ending time
    t_end = time.time()
    Time.append(t_end - t_start)
    # First coordinate is the intensity of noise, second coordinate is the fraction.
ax.plot(deltalist, CDFrefresults, marker="d", label = "ref(=DP)")
print("The results for ref(=DP) are", CDFrefresults)

# The total time cost is denoted by T.
T = sum(Time)
print(f"The total time cost for ref(=DP) is {T}")
print()



# The list of time costs
Time = []   
# Results for CDF distances
CDFresults = []
# Obtain the distances 
for i in range(17):
    # The starting time
    t_start = time.time()
    cnt = 0
    # Do rep times of independent experiments
    for j in range(rep):
        perm = Perm[i][j]
        ProfileG = PG[i][j]; ProfileH = PH[i][j]
        for u in vertices:
            v = matchCDF(vertices, ProfileG, ProfileH, u)
            if v == perm[u]:
                cnt += 1
    # Fraction of vertices being matched
    CDFresults.append(cnt/(num*rep))
    # The ending time
    t_end = time.time()
    Time.append(t_end - t_start)
    # First coordinate is the intensity of noise, second coordinate is the fraction.
ax.plot(deltalist, CDFresults, marker="x", label = "cdf")
print("The results for cdf are", CDFresults)

# The total time cost is denoted by T.
T = sum(Time)
print(f"The total time cost for cdf is {T}")
print()


# The list of all the widths of bins.
width = [0.5, 1, 2]


for r in width:
    # The list of time costs
    Time = []      
    # Results for BIN distances with width being r
    BINresults = []
    # Obtain the distances 
    for i in range(17):
        # The starting time
        t_start = time.time()
        cnt = 0
        # Do rep times of independent experiments
        for j in range(rep):
            perm = Perm[i][j]
            ProfileG = PG[i][j]; ProfileH = PH[i][j]
            for u in vertices:
                # Width = r
                v = matchBIN(vertices, ProfileG, ProfileH, u, r)
                if v == perm[u]:
                    cnt += 1
        # Fraction of vertices being matched
        BINresults.append(cnt/(num*rep))
        # The ending time
        t_end = time.time()
        Time.append(t_end - t_start)
        # First coordinate is the intensity of noise, second coordinate is the fraction.
    ax.plot(deltalist, BINresults, marker="o", label = f"r={r}")
    print(f"The results for width={r} are", BINresults)

    # The total time cost is denoted by T.
    T = sum(Time)
    print(f"The total time cost for width={r} is {T}")
    print()

ax.legend()
plt.show()



    
        



    
    
    