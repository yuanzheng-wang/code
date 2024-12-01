# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:55:51 2023

@author: dell
"""

import time
import igraph as ig
from igraph import Graph
import random 
import matplotlib.pyplot as plt
import numpy as np

t1=time.time()

def Generatepower(num, pw, const, sub_prob):
    ParGraph = Graph()
    ParGraph.add_vertices(num)
    for i in range(num):
        for j in range(i+1, num):
            if random.random() < const*(j-i)**(pw):
                ParGraph.add_edge(i, j)
    SubGraph1 = Graph.Erdos_Renyi(n=num, p=sub_prob, directed=False, loops=False)
    SubGraph2 = Graph.Erdos_Renyi(n=num, p=sub_prob, directed=False, loops=False)
    G = ig.intersection([ParGraph, SubGraph1])
    H = ig.intersection([ParGraph, SubGraph2])
    rootdG = [d**0.5 for d in G.degree()]
    rootdH = [d**0.5 for d in H.degree()]
    ProfileG = [[rootdG[j] for j in G.neighbors(i)] for i in range(num)]
    ProfileH = [[rootdH[j] for j in H.neighbors(k)] for k in range(num)] 
    return ProfileG, ProfileH

def Generatepowerref(num, pw, const, sub_prob):
    ParGraph = Graph()
    ParGraph.add_vertices(num)
    for i in range(num):
        for j in range(i+1, num):
            if random.random() < const*(j-i)**(pw):
                ParGraph.add_edge(i, j)
    SubGraph1 = Graph.Erdos_Renyi(n=num, p=sub_prob, directed=False, loops=False)
    SubGraph2 = Graph.Erdos_Renyi(n=num, p=sub_prob, directed=False, loops=False)
    G = ig.intersection([ParGraph, SubGraph1])
    H = ig.intersection([ParGraph, SubGraph2])
    rootdG = [d for d in G.degree()]
    rootdH = [d for d in H.degree()]
    ProfileG = [[rootdG[j] for j in G.neighbors(i)] for i in range(num)]
    ProfileH = [[rootdH[j] for j in H.neighbors(k)] for k in range(num)] 
    return ProfileG, ProfileH

def CDFdist(A, B):
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

def BINdist(A, B, r):
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
    
def matchCDF(num, Profile1, Profile2, i):
    cur_min = num*5
    arg_min = 0
    for k in range(num):
        new = CDFdist(Profile1[i], Profile2[k])
        if new < cur_min:
            cur_min = new
            arg_min = k
    return arg_min

def matchBIN(num, Profile1, Profile2, i, r):
    cur_min = num*5
    arg_min = 0
    for k in range(num):
        new = BINdist(Profile1[i], Profile2[k], r)
        if new < cur_min:
            cur_min = new
            arg_min = k
    return arg_min

def runCDF(num,pw,const,sub_prob):
    ProfileG, ProfileH = Generatepower(num,pw,const,sub_prob)
    cnt = 0
    for i in range(num):
        if matchCDF(num, ProfileG, ProfileH, i) == i:
            cnt += 1
    return cnt

def runCDFref(num,pw,const,sub_prob):
    ProfileG, ProfileH = Generatepowerref(num,pw,const,sub_prob)
    cnt = 0
    for i in range(num):
        if matchCDF(num, ProfileG, ProfileH, i) == i:
            cnt += 1
    return cnt
    
def runBIN(num,pw,const,sub_prob,r):
    ProfileG, ProfileH = Generatepower(num,pw,const,sub_prob)
    cnt = 0
    for i in range(num):
        if matchBIN(num, ProfileG, ProfileH, i, r) == i:
            cnt += 1
    return cnt

num = 1000
pw = -0.5
const = 0.5
fig, ax = plt.subplots()
SIGMA = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

CDFresults = []
for sigma in SIGMA:
    cnt = 0
    for _ in range(10):
        cnt += runCDF(num,pw,const,1-sigma*sigma)
    CDFresults.append(cnt/10000)
ax.plot(SIGMA, CDFresults, marker="x", label = "CDF")
t2=time.time()
print(t2-t1)

BINresults_1 = []
for sigma in SIGMA:
    cnt = 0
    delta=sigma*sigma
    sub_prob=1-delta
    for _ in range(10):
        cnt += runBIN(num,pw,const,sub_prob,0.5)
    BINresults_1.append(cnt/10000)
ax.plot(SIGMA, BINresults_1, marker="o", label = "r=0.5")
t3=time.time()
print(t3-t2)

BINresults_2 = []
for sigma in SIGMA:
    cnt = 0
    delta=sigma*sigma
    sub_prob=1-delta
    for _ in range(10):
        cnt += runBIN(num,pw,const,sub_prob,1)
    BINresults_2.append(cnt/10000)
ax.plot(SIGMA, BINresults_2, marker="o", label = "r=1")
t4=time.time()
print(t4-t3)

CDFrefresults = []
for sigma in SIGMA:
    cnt = 0
    for _ in range(10):
        cnt += runCDFref(num,pw,const,1-sigma*sigma)
    CDFrefresults.append(cnt/10000)
ax.plot(SIGMA, CDFrefresults, marker="x", label = "ref")
t5=time.time()
print(t5-t4)

ax.legend()
plt.show()

