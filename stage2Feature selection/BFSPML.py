"""
Evaluate feature subset by maximize consistency
version of numpy
"""

import torch
import numpy as np
from scipy.spatial.distance import cdist

class BFSPML:
    def __init__(self, X, Y, sigma, k):
        self.sigma = sigma
        self.k = k
        self.X = X
        self.num_samples = X.shape[0]
        # initialize the label similarity matrix using manhattan distance
        self.V = self.__calculate_V(Y=Y)
        self._1_V= 1-self.V
        self.M_1 = np.ones((self.num_samples, self.num_samples))

    def __calculate_V(self,Y):
        # V indicates label similarity among samples calculated by label confidence
        return 1 - cdist(Y,Y,metric='cityblock') / 2

    def __calculate_As_and_Es_and_Z(self, X):
        # As indicates similarity among sample in the feature space
        # Es indicated neighborhood relation among samples in the feature space
        # Z indicate dot product of matrices As and Es
        distance_matrix = cdist(X,X,metric='euclidean')
        # calculate As
        As = np.exp(-distance_matrix**2 / (2 * self.sigma**2))
        # calculate Es
        sorted_indices = np.argsort(distance_matrix, axis=1)
        k_neighbors_indices = sorted_indices[:, 1:self.k + 1]
        Es = np.zeros((self.num_samples, self.num_samples))
        row_indices = np.arange(self.num_samples)[:,np.newaxis]
        Es[row_indices, k_neighbors_indices] = 1
        # calculate Z
        Z = As * Es # * is a dot multiplication operator
        return As, Es, Z

    def __calculate_a_and_b(self, As, V):
        M_1 = self.M_1
        tr_AsV = np.trace(As@V) # @ indicates matrix multiplication
        tr_M1M1 = np.trace(M_1@M_1)

        tr_AsM1 = np.trace(As@M_1)
        tr_VM1 = np.trace(V@M_1)
        tr_AsAs = np.trace(As@As)

        a = (tr_AsV*tr_M1M1-tr_AsM1*tr_VM1)/\
            (tr_AsAs*tr_M1M1-tr_AsM1*tr_AsM1)

        b = (tr_AsAs*tr_VM1-tr_AsM1*tr_AsV)/\
            (tr_AsAs*tr_M1M1-tr_AsM1*tr_AsM1)
        return a, b

    def feature_score(self,feature_index:list):
        As, Es, Z = self.__calculate_As_and_Es_and_Z(self.X[:,feature_index])
        a, b = self.__calculate_a_and_b(As=As, V=self.V)
        term_1 = np.linalg.norm((a*As+b*self.M_1-self.V), 'fro')
        term_2 = np.trace(Z@self._1_V)
        return term_1, term_2



if __name__ == '__main__':
    X = np.array([
        [0.5, 0.7, 0.8],
        [0.1, 0.1, 0.1],
        [1.6, 0.9, 0.6]
    ],dtype=np.float32)

    Y = np.array([
        [0.2,0.5,0.3],
        [0.6,0.1,0.3],
        [1,0,0]
    ], dtype=np.float32)
    fs = BFSPML(X= X, Y=Y, sigma=2,k=1)
    s1, s2 = fs.feature_score(feature_index=[0, 1, 2])
    print(s1)
    print(s2)