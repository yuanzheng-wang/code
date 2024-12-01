# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:07:00 2024

@author: dell
"""

from datetime import datetime, timedelta
import time
import numpy as np
from scipy.linalg import eigh
import scipy as sp
import random
import matplotlib.pyplot as plt
import networkx as nx


def shur_complement(V_A, V_B, lambda_, n):
    """
    This is the python version of the shur_complement script in matlab, which computes the Schur complement.

    Parameters:
    n (int): Dimension of the matrices.
    V_A (ndarray): Eigenvector matrix for A (n by n).
    V_B (ndarray): Eigenvector matrix for B (n by n).
    lambda_ (ndarray): Flattened vector of eigenvalues of the form lambda_vec (1 by n^2 row vector).

    Returns:
    ndarray: Schur complement matrix S (2n by 2n).
    """
    S1 = np.zeros((n, n))
    # V_B_one and V_A_one are n*1 column vectors
    V_B_one = V_B.T @ np.ones((n,1))
    V_A_one = V_A.T @ np.ones((n,1))
    
    for i in range(n):
        # temp is n*n matrix
        temp = np.outer(lambda_[i * n:(i + 1) * n], np.ones(n))
        # number times n*n matrix times n*n matirx
        S1 += (V_B_one[i][0] ** 2) * V_A @ (temp * V_A.T)
    
    S2 = np.zeros((n, n))
    for i in range(n):
        # temp is a column vector obtained via entrywise product of tow column vectors
        temp = lambda_[i * n:(i + 1) * n][:,None] * V_A_one
        # number times n*n matrix times n*n matrix (column vector times row vector)
        S2 += V_B_one[i][0] * V_A @ (temp* V_B[:, i])
    
    S3 = S2.T
    
    S4 = np.zeros((n, n))
    for i in range(n):
        # temp is a column vector obtained via entrywise product of tow column vectors
        temp = lambda_[i * n:(i + 1) * n][:,None] * (V_A_one ** 2)
        # number times n by n matrix (column vector times row vector)
        S4 += np.sum(temp) * (V_B[:, i][:,None] * V_B[:, i])
    
    # S1-S4 are n by n matrices. S is 2n by 2n.
    S = -1 * np.block([[S1, S2], [S3, S4]])
    
    return S


def Fourier_basis(n):
    """
    This is the python version of the Fourier_basis script in matlab, which 
    generates an n x n matrix where the columns are the Fourier basis vectors in R^n.

    Parameters:
    n : int
        Dimension of the Fourier basis.

    Returns:
    V : ndarray
        An n by n matrix containing the Fourier basis vectors as columns.
    """
    ell = (n - 1) // 2  # Equivalent to floor((n-1)/2)
    omega = 2 * np.pi / n

    # Generate the cosine and sine parts
    indices = np.arange(n).reshape(-1, 1)  # Column vector [0, 1, ..., n-1]
    harmonics = np.arange(1, ell + 1)      # Row vector [1, 2, ..., ell]

    V1 = np.sqrt(2) * np.cos(omega * indices * harmonics)
    V2 = np.sqrt(2) * np.sin(omega * indices * harmonics)

    # Combine the basis vectors
    V = np.hstack([np.ones((n, 1)), V1, V2])

    # Add the special case for even n
    if n % 2 == 0:
        V = np.hstack([V, np.cos(np.pi * indices)])

    # Normalize by sqrt(n)
    V = V / np.sqrt(n)
    return V



def quadprog_admm(A, B, Aeq, b, lb, ub, rho, alpha, ABSTOL, RELTOL):
    """
    This is the python version of the script quadprog_admm, which solves a quadratic program using ADMM.
    This function uses shur_complement and Fourier_basis.
    
    Below is the function description in the script. Note that here A and B are assumed as symmetric matrices.
    The goal is to find a doubly stochastic matrix X (i.e. X1 = 1, 1X = 1) to minimize ||AX-XB|| (in the Frobenius norm)
_____________________________________________________________________________________________________________________________________
    
    Solve standard form box-constrained QP via ADMM

     [x, history] = quadprog(P, q, r, lb, ub, rho, alpha)

    Solves the following problem via ADMM:

        minimize     (1/2)*x'*P*x + q'*x + r
        subject to   Aeq *x = b;
                     lb <= x <= ub
        where P= kron(speye(n),A'*A)+kron(B*B',speye(n))-2*kron(B',A'); The solution is returned in the vector x.

     history is a structure that contains the objective value, the primal and
     dual residual norms, and the tolerances for the primal and dual residual norms at each iteration.

     rho is the augmented Lagrangian parameter.

     alpha is the over-relaxation parameter (typical values for alpha are
                                             between 1.0 and 1.8).
     
     More information can be found in the paper linked at:
     http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
______________________________________________________________________________________________________________________________________

    Args:
        A, B: Matrices in the objective function.
        Aeq, b: Equality constraint Aeq @ x = b.
        lb, ub: Box constraints lb <= x <= ub.
        rho: Augmented Lagrangian parameter.
        alpha: Relaxation parameter.
        ABSTOL, RELTOL: Tolerances for stopping criteria.

    Returns:
        z: Solution to the QP, which is a 1 by n^2 row vector that is the flattened vector of the doubly stochastic matrix.
        history: Dictionary containing diagnostics during iterations.
    """
    
    t_start_setting = time.time()
    # Constants
    MAX_ITER = 1000
    n = A.shape[0]

    # Initialize variables
    x = np.zeros(n**2)
    z = np.zeros(n**2)
    u = np.zeros(n**2)
    
    history = {"objval": [], "r_norm": [], "s_norm": [], "eps_pri": [], "eps_dual": []}
    

    # Preprocessing d_A is 1*n row vector containing all the eigenvalues, V_A is n*n matrix containing all the corresponding
    # eigenvectors as column vecotrs
    d_A, V_A = eigh(A)
    d_B, V_B = eigh(B)

    # Lambda is n*n matrix
    Lambda = np.outer(d_A**2, np.ones(n)) + np.outer(np.ones(n), d_B**2) - 2 * np.outer(d_A, d_B) + rho
    # Lambda_inv is n*n matrix
    Lambda_inv = 1 / Lambda
    # lambda_vec is 1 by n^2 row vector
    lambda_vec = Lambda_inv.ravel('F')
    
    # S is a 2n by 2n matrix
    S = shur_complement(V_A, V_B, lambda_vec, n)
    
    # v_f is a n by n matrix
    v_f = Fourier_basis(n)
    V_temp1 = np.hstack([v_f[:, [0]] / np.sqrt(2), v_f[:, 1:], np.zeros((n, n - 1))])
    V_temp2 = np.hstack([v_f[:, [0]] / np.sqrt(2), np.zeros((n, n - 1)), v_f[:, 1:]])
    # V_temp is a 2n by 2n-1 matrix
    V_temp = np.vstack([V_temp1, V_temp2])
    
    # S_proj is a 2n-1 by 2n-1 matrix
    S_proj = V_temp.T @ S @ V_temp
    # b_proj is 2n-1 by 1 column vector
    b_proj = V_temp.T @ b
    
    t_end_setting = time.time()
    t = t_end_setting - t_start_setting
    print(f"The time for ADMM setting is {t}")
    
    # ADMM Iterations
    for k in range(MAX_ITER):
        t_start_qp = time.time()
        # x-update
        Z = z.reshape((n, n)).T
        U = u.reshape((n, n)).T

        vec_1 = rho * V_A.T @ (Z - U) @ V_B
        vec_2 = lambda_vec * vec_1.ravel('F')
        mat_1 = V_A @ vec_2.reshape((n, n)).T @ V_B.T
        

        Aeq_mat1 = np.vstack([mat_1 @ np.ones((n,1)), mat_1.T @ np.ones((n,1))])
        mu = np.linalg.solve(S_proj, V_temp.T @ Aeq_mat1 - b_proj)
        V_temp_mu = V_temp @ mu
        vec3 = np.kron(np.ones((n,1)), V_temp_mu[:n]) + np.kron(V_temp_mu[n:2*n], np.ones((n,1)))


        vec_4 = V_A.T @ vec3.reshape((n, n)).T @ V_B
        vec_5 = lambda_vec * vec_4.ravel('F')
        mat_2 = V_A @ vec_5.reshape((n, n)).T @ V_B.T


        x = mat_2 + mat_1
        x = x.ravel('F')


        # z-update with relaxation (box constraint)
        zold = z.copy()
        x_hat = alpha * x + (1 - alpha) * zold
        z = np.minimum(ub, np.maximum(lb, x_hat + u))

        # u-update
        u += x_hat - z
        

        # Diagnostics
        X = x.reshape((n, n)).T
        objval = 0.5 * np.linalg.norm(A @ X - X @ B, 'fro')**2
        r_norm = np.linalg.norm(x - z)
        s_norm = np.linalg.norm(-rho * (z - zold))
        eps_pri = np.sqrt(n) * ABSTOL + RELTOL * max(np.linalg.norm(x), np.linalg.norm(-z))
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * u)

        history["objval"].append(objval)
        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)
        history["eps_pri"].append(eps_pri)
        history["eps_dual"].append(eps_dual)

        t_end_qp = time.time()
        t = t_end_qp - t_start_qp
        print(f"The time for the {k}-th round is {t}")
        if r_norm < eps_pri and s_norm < eps_dual:
            break
        
    return z, history

def matching_full_qp(A,B):
    """
    This is the python version of the script matching_full_qp in matlab, which uses a two-step method to solve
    solve graph matching. First is the convex relaxation step: Find a doubly stochastic matrix X (i.e. X1 = 1, 1X = 1)
    that minimizes ||AX-XB|| (under the Frobenium norm); this step uses quadprog_admm to solve the quadratic programming problem,
    where some parameters are chosen as in the matlab script.
    Second is to project X to the space of permutation matrices, that is, find a permutation matrix P that maximizes 
    the L1 norm of X*P (entrywise product), which is essentially a linear assignment problem.
    
    Parameters
    ----------
    n: dimension of the matrices A and B

    Returns
    -------
    row_ind : a 1 by n array of row indices of the 1's in the permutation matrix P, which is tacitly set as np.array([0,1,...,n-1])
    col_ind : a 1 by n array of column indices of the 1's in the permutation matrix P. 
    
    Note that if P @ A @ P.T = B, then this function returns row_ind and col_ind with P(row_ind, col_ind) = 1.

    """
    n = A.shape[0]
    # x is a 1 by n^2 row vector, which is the flattened vector of the doubly stochastic matrix
    x, history = quadprog_admm(A,B,[],np.ones((2*n,1)),np.zeros(n**2),np.ones(n**2),40,1.5,1e-5,1e-3)
    Xhat = x.reshape(n,n).T
    flag = np.isfinite(Xhat)
    cnt = 0
    for i in range(n):
        for j in range(n):
            if flag[i][j] == False:
                print((i,j), Xhat[i][j])
                cnt += 1
    print(cnt)
    row_ind, col_ind = sp.optimize.linear_sum_assignment(-Xhat.T)
    return row_ind, col_ind


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

t1 = time.time()
# Consider the subgraphs generated by some common vetices. The number of common
# vertices is denoted by threshold.
threshold = 10000
topv = [key for key, value in sorted(dg.items(), key=lambda item: item[1], reverse=True)[:threshold]]
# The vertices in topv are not ordered from 1 to threshold, that's why we need dictv
dictv = dict(zip(topv, list(range(threshold))))
# Generate the subgraphs of these common vertices
for i in range(9):
    graphs[i] = graphs[i].subgraph(topv)

# The list of all the permutations
Perm = []

# Translate a graph into a matrix
Graph_matrix = []
for i in range(9):
    G = graphs[i]
    # Add 0.0001 to avoid the whole-nan bug
    M = np.ones((threshold,threshold))/10000
    # The first graph is not permuted
    if i == 0:
        perm = list(range(threshold))
        perm_inv = list(range(threshold))
    else:
        # Generate a random permutation
        perm = random.sample(range(threshold),threshold)
        # perm_inv is the inverse permutation of perm
        perm_inv = dict(zip(perm, list(range(threshold))))
    # The i-th graph will be randomly permuted, and M will be the adjacency matrix. 
    # We want H = P @ G @ P.T, where P is the permutation matrix for perm.
    for j, k in G.edges():
        M[perm_inv[dictv[j]]][perm_inv[dictv[k]]] = 1
        M[perm_inv[dictv[k]]][perm_inv[dictv[j]]] = 1
    Perm.append(perm)
    Graph_matrix.append(M)


# Consider the subgraphs FURTHER generated by some HIGH-DEGREE VERTICES among 
# the above common vertices(whose number is threshold). The number of high-degree
# vertices is denoted by bar.
bar = 1000

flag = True
for i in range(bar):
    for j in range(bar, threshold):
        if dg[topv[i]] < dg[topv[j]]:
            flag = False
print(flag)
t2 = time.time()
print(t2 - t1) 
    
# Two plots regarding the whole graph and the high-degree subgraph
fig, ax_top = plt.subplots(); fig, ax_high = plt.subplots()
ax_top.set_title("Oregon whole graph"); ax_high.set_title("Oregon high-degree subgraph")
ax_top.set_xlabel("Dates"); ax_top.set_ylabel("Fraction of correctly matched vertices")
ax_high.set_xlabel("Dates"); ax_high.set_ylabel("Fraction of correctly matched vertices")

# The base graph to which every other graph will be compared 
G = Graph_matrix[0]

# The list of time costs
Time = []   
# Results for qp. qpresults_top record the results for the whole graph; 
# qpresults_high record the results for the high-degree graph. 
qpresults_top = []
qpresults_high = [] 
# Obtain the distances 
for i in range(9):
    # The starting time
    t_start = time.time()
    # perm is the i-th permutation
    perm = Perm[i]
    # H is the permutated adjacency matrix of the i-th graph
    H = Graph_matrix[i]
    
    row_ind, col_ind = matching_full_qp(G, H)
    # cnt_top counts the corretly matched vertices in the whole graph
    cnt_top = 0
    for j in range(threshold):
        if col_ind[j] == perm[j]:
            cnt_top +=1
    # Fraction of vertices being matched
    qpresults_top.append(cnt_top/threshold)
    
    # cnt_high counts the corretly matched vertices in the high-degree graph
    cnt_high = 0
    for j in range(bar):
        if col_ind[j] == perm[j]:
            cnt_high +=1
    # Fraction of vertices being matched
    qpresults_high.append(cnt_high/bar)
    
    # The ending time
    t_end = time.time()
    print('time', t_end - t_start)
    print(cnt_top, cnt_high)
    print()
    Time.append(t_end - t_start)
    
# First coordinate is the intensity of noise, second coordinate is the fraction.
ax_top.plot(days, qpresults_top, marker="o", label = 'qp')
print('The results of the whole graph for qp are', qpresults_top)
ax_high.plot(days, qpresults_high, marker="o", label = 'qp')
print('The results of the hd-subgraph for qp are', qpresults_high)

# The total time cost is denoted by T.
T = sum(Time)
print(f"The total time cost for qp is {T}")

ax_top.legend(); ax_high.legend()
plt.show()
