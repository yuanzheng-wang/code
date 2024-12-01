# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:42:51 2023

@author: dell
"""

import time
import networkx as nx
import random 
import matplotlib.pyplot as plt

def GeneratePLD(num, sub_prob, gamma):
    seq = [random.uniform(0,0.01)**(-gamma)/sub_prob for i in range(num)]
    ParGraph = nx.expected_degree_graph(seq, selfloops = False)
    SubGraph1 = nx.gnp_random_graph(num, sub_prob)
    SubGraph2 = nx.gnp_random_graph(num, sub_prob)
    G = nx.intersection(ParGraph, SubGraph1)
    H = nx.intersection(ParGraph, SubGraph2)
    dG = nx.degree(G)
    dH = nx.degree(H)
    rootdG = [dG[i]**0.5 for i in range(num)]
    rootdH = [dH[i]**0.5 for i in range(num)]
    ProfileG = [[rootdG[j] for j in G.neighbors(i)] for i in range(num)]
    ProfileH = [[rootdH[j] for j in H.neighbors(k)] for k in range(num)] 
    return ProfileG, ProfileH

def GeneratePLDref(num, sub_prob, gamma):
    seq = [random.uniform(0,0.01)**(-gamma)/sub_prob for i in range(num)]
    ParGraph = nx.expected_degree_graph(seq, selfloops = False)
    SubGraph1 = nx.gnp_random_graph(num, sub_prob)
    SubGraph2 = nx.gnp_random_graph(num, sub_prob)
    G = nx.intersection(ParGraph, SubGraph1)
    H = nx.intersection(ParGraph, SubGraph2)
    dG = nx.degree(G)
    dH = nx.degree(H)
    rootdG = [dG[i] for i in range(num)]
    rootdH = [dH[i] for i in range(num)]
    ProfileG = [[rootdG[j] for j in G.neighbors(i)] for i in range(num)]
    ProfileH = [[rootdH[j] for j in H.neighbors(k)] for k in range(num)] 
    return ProfileG, ProfileH  

def GeneratePLDdisc(num, sub_prob, gamma, r):
    seq = [random.uniform(0,0.01)**(-gamma)/sub_prob for i in range(num)]
    ParGraph = nx.expected_degree_graph(seq, selfloops = False)
    SubGraph1 = nx.gnp_random_graph(num, sub_prob)
    SubGraph2 = nx.gnp_random_graph(num, sub_prob)
    G = nx.intersection(ParGraph, SubGraph1)
    H = nx.intersection(ParGraph, SubGraph2)
    dG = nx.degree(G)
    dH = nx.degree(H)
    rootdG = [(dG[i]**0.5)//(2*r) for i in range(num)]
    rootdH = [(dH[i]**0.5)//(2*r) for i in range(num)]
    ProfileG = [[rootdG[j] for j in G.neighbors(i)] for i in range(num)]
    ProfileH = [[rootdH[j] for j in H.neighbors(k)] for k in range(num)] 
    return ProfileG, ProfileH  

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
    
def matchCDF(num, Profile1, Profile2, i):
    cur_min = num*5
    arg_min = -1
    for k in range(num):
        new = CDFdist(Profile1[i], Profile2[k])
        if new < cur_min:
            cur_min = new
            arg_min = k
    return arg_min

def matchBIN(num, Profile1, Profile2, i, r):
    cur_min = num*5
    arg_min = -1
    for k in range(num):
        new = BINdist(Profile1[i], Profile2[k], r)
        if new < cur_min:
            cur_min = new
            arg_min = k
    return arg_min

def runCDF(num, sub_prob, gamma):
    ProfileG, ProfileH = GeneratePLD(num, sub_prob, gamma)
    cnt = 0
    for i in range(num):
        if matchCDF(num, ProfileG, ProfileH, i) == i:
            cnt += 1
    return cnt

def runCDFref(num, sub_prob, gamma):
    ProfileG, ProfileH = GeneratePLDref(num, sub_prob, gamma)
    cnt = 0
    for i in range(num):
        if matchCDF(num, ProfileG, ProfileH, i) == i:
            cnt += 1
    return cnt
    
def runBIN(num, sub_prob, gamma, r):
    ProfileG, ProfileH = GeneratePLD(num, sub_prob, gamma)
    cnt = 0
    for i in range(num):
        if matchBIN(num, ProfileG, ProfileH, i, r) == i:
            cnt += 1
    return cnt

def runDISC(num, sub_prob, gamma, r):
    ProfileG, ProfileH = GeneratePLDdisc(num, sub_prob, gamma,r)
    cnt = 0
    for i in range(num):
        if matchBIN(num, ProfileG, ProfileH, i, 0.5) == i:
            cnt += 1
    return cnt

num = 1000
gamma = 2/3
SIGMA = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
rep = 10

fig, ax = plt.subplots()
t1 = time.time()

CDFresults = []
for sigma in SIGMA:
    cnt = 0
    for _ in range(rep):
        cnt += runCDF(num, 1-sigma*sigma, gamma)
    CDFresults.append(cnt/10000)
ax.plot(SIGMA, CDFresults, marker="x", label = "CDF")
t2=time.time()
print(t2-t1)

BINresults_1 = []
for sigma in SIGMA:
    cnt = 0
    for _ in range(rep):
        cnt += runBIN(num, 1-sigma*sigma, gamma, 0.5)
    BINresults_1.append(cnt/10000)
ax.plot(SIGMA, BINresults_1, marker="o", label = "r=0.5")
t3=time.time()
print(t3-t2)


BINresults_2 = []
for sigma in SIGMA:
    cnt = 0
    for _ in range(rep):
        cnt += runBIN(num, 1-sigma*sigma, gamma, 1)
    BINresults_2.append(cnt/10000)
ax.plot(SIGMA, BINresults_2, marker="o", label = "r=1")
t4=time.time()
print(t4-t3)

CDFrefresults = []
for sigma in SIGMA:
    cnt = 0
    for _ in range(rep):
        cnt += runCDFref(num, 1-sigma*sigma, gamma)
    CDFrefresults.append(cnt/10000)
ax.plot(SIGMA, CDFrefresults, marker="x", label = "ref")
t5=time.time()
print(t5-t4)

DISCresults_1 = []
for sigma in SIGMA:
    cnt = 0
    for _ in range(rep):
        cnt += runDISC(num, 1-sigma*sigma, gamma, 0.5)
    DISCresults_1.append(cnt/10000)
ax.plot(SIGMA, DISCresults_1, marker="d", label = "disc")
t6=time.time()
print(t6-t5)

ax.legend()
plt.show()