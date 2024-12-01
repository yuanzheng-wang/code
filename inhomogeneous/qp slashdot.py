# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:51:39 2024

@author: dell
"""

import time
import numpy as np
from scipy.linalg import eigh
import scipy as sp
import random
import matplotlib.pyplot as plt


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
    

    # ADMM Iterations
    for k in range(MAX_ITER):
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
    # print(Xhat)
    row_ind, col_ind = sp.optimize.linear_sum_assignment(-Xhat.T)
    return row_ind, col_ind


# Given a graph and a permutation, generate the permuted graph.
def permute(G, number, perm):
    H = np.zeros((number,number))
    for i in range(number):
        for j in range(number):
            # H = P @ A @ P.T where P is the permutation matrix.
            H[i][j] = G[perm[i]][perm[j]]
    return H
    
# Given a parent graph, generate two independently subsampled graph,
# where in each subsampling process, each edge is independently maintained with
# probability sub_prob. The second graph is then permuted via a given permtation. 
def subsample(ParGraph, number, sub_prob, perm):
    G = np.zeros((number, number))
    for i in range(number):
        for j in range(number):
            # Subsampling each edge
            if ParGraph[i][j] == 1:
                p = random.random()
                if p < sub_prob:
                    G[i][j] = 1
                    
    H = np.zeros((number, number))
    for i in range(number):
        for j in range(number):
            # Subsampling each edge
            if ParGraph[i][j] == 1:
                p = random.random()
                if p < sub_prob:
                    H[i][j] = 1
                    
    # Permute H
    H_perm = permute(H, number, perm)
    return G, H_perm

# Count the number of correctly matched vertices
def cnt_matching_qp(G, H_perm, number, perm):
    row_ind, col_ind = matching_full_qp(G, H_perm)
    # cnt is the number of correctly matched vertices
    cnt = 0
    for i in range(number):
        if col_ind[i] == perm[i]:
            cnt += 1
    return cnt
    

# The list of all the delta's. The subsampling probability will equal 1-delta
deltalist = [0.025*i for i in range(17)]

# The number of independent repetitions
rep = 10

# Read the file
file = open('Slashdot0902.txt', 'r')
content = file.readlines()
file.close()

# The number of vertices 
num = 750
# Translate a file into a graph. Par is the parent graph
l = len(content)
Par = np.zeros((num, num))
# Each line in file represents an edge
for i in range(4,l):
    # Add an edge into G
    a,b = map(int, content[i].split('\t'))
    if a < num and b < num and a != b:
        Par[a][b] = 1
        Par[b][a] = 1
 
    
fig, ax = plt.subplots()
ax.set_title("Slashdot network")
ax.set_xlabel("1-s")
ax.set_ylabel("Fraction of correctly matched vertices")
 

# The list of time costs
Time = []   
# Results for CDF distances
qpresults = [] 
# Obtain the distances 
for i in range(17):
    # The starting time
    t_start = time.time()
    # s is the subsampling probability
    s = 1 - deltalist[i]
    # The number of correctly matched vertices
    cnt = 0
    # Do rep times of independent experiments
    for j in range(rep):
        # Generate a random permutation
        perm = random.sample(range(num),num)
        G, H_perm = subsample(Par, num, s, perm)
        t_setting = time.time()
        print(f"The time for setting is {t_setting - t_start}")
        cnt += cnt_matching_qp(G, H_perm, num, perm)
    # Fraction of vertices being matched
    qpresults.append(cnt/(num*rep))
    # The ending time
    t_end = time.time()
    print('time', t_end - t_start)
    print('total cnt',cnt)
    print()
    Time.append(t_end - t_start)
    # First coordinate is the intensity of noise, second coordinate is the fraction.
ax.plot(deltalist, qpresults, marker="+", label = 'qp')
print('The results for qp are', qpresults)
# The total time cost is denoted by T.
T = sum(Time)
print(f"The total time cost for qp is {T}")

ax.legend()
plt.show()
