This folder contains the codes of all the experiments in the paper "Efficiently matching inhomogeneous random graphs via 
degree profiles". The codes are all written in python.

The file "numeric study ER" contains the codes for correlated Erd\H{o}s R\'{e}nyi graphs, which gives Figure 1 in the paper.

The file "numeric study power law" contains the codes for the model in which we connect vertices i and j in the parent graph
with probability 0.5|i-j|^{-1/2}. The results give Figure 2 in the paper.

The file "numeric study PLD_Chung-Lu" contains the codes for correlated Chung-Lu model in which the expected degree sequence 
follows a power law distribution. The results give Figure 3 in the paper.

The file "final_experiment_slashdot" contains the codes of degree-profile-based algorithms in the Slashdot Network experiment.
The file "qp slashdot" contains the codes the codes of QP (quadratic programming) algorithm in the Slashdot Network experiment.
Together, the results give Figure 4 in the paper.

The file "oregon code" contains the codes of degree-profile-based algorithms in the networks of Anonymous System experiment. 
The results give Figure 5 in the paper.

The file "qp oregon" contains the codes the codes of QP (quadratic programming) algorithm in the networks of Anonymous System
experiment. However, QP is too slow on the graphs, and we did not complete running.

