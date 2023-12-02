"""
Evaluate feature subset by maximize consistency
version of pytorch
"""

import torch
import numpy as np
from scipy.spatial.distance import cdist

torch.set_default_dtype(torch.float64)

# check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
class BFSPML_torch:
    def __init__(self, X, Y, sigma, k):
        self.sigma = sigma
        self.k = k
        self.X = X
        self.num_samples = X.size(0)
        # initialize the label similarity matrix using manhattan distance
        self.V = self.__calculate_V(Y=Y)
        self._1_V= 1-self.V
        self.M_1 = torch.ones(self.num_samples, self.num_samples,dtype=torch.float32)

    def __calculate_V(self,Y):
        # V indicates label similarity among samples calculated by label confidence
        return 1 - torch.cdist(Y, Y, p=1) / 2  # p=1 indicates manhattan distance

    def __calculate_As_and_Es_and_Z(self, X):
        # As indicates similarity among sample in the feature space
        # Es indicated neighborhood relation among samples in the feature space
        # Z indicate dot product of matrices As and Es
        distance_matrix = torch.cdist(X, X, p=2)
        # calculate As
        As = torch.exp(-distance_matrix.pow(2) / (2 * self.sigma**2))
        # calculate Es
        sorted_indices = torch.argsort(distance_matrix, dim=1)
        k_neighbors = sorted_indices[:, 1:self.k + 1]
        Es = torch.zeros(self.num_samples, self.num_samples, dtype=torch.float32)
        row_indices = torch.arange(self.num_samples).view(-1, 1)
        Es[row_indices, k_neighbors] = 1
        # calculate Z
        Z = As * Es # * is a dot multiplication operator
        return As, Es, Z

    def __calculate_a_and_b(self, As, V):
        M_1 = self.M_1
        tr_AsV = torch.trace(As@V) # @ indicates matrix multiplication
        tr_M1M1 = torch.trace(M_1@M_1)

        tr_AsM1 = torch.trace(As@M_1)
        tr_VM1 = torch.trace(V@M_1)
        tr_AsAs = torch.trace(As@As)

        a = (tr_AsV*tr_M1M1-tr_AsM1*tr_VM1)/\
            (tr_AsAs*tr_M1M1-tr_AsM1*tr_AsM1)

        b = (tr_AsAs*tr_VM1-tr_AsM1*tr_AsV)/\
            (tr_AsAs*tr_M1M1-tr_AsM1*tr_AsM1)
        return a.item(), b.item()

    def feature_score(self,feature_index:list):
        As, Es, Z = self.__calculate_As_and_Es_and_Z(self.X[:,feature_index])
        a, b = self.__calculate_a_and_b(As=As, V=self.V)
        # a, b = 1,1
        term_1 = torch.norm((a*As+b*self.M_1-self.V), p='fro')
        term_2 = torch.trace(Z@self._1_V)
        return term_1.item(), term_2.item()



if __name__ == '__main__':
    X = torch.tensor([
        [0.5, 0.7, 0.8],
        [0.1, 0.1, 0.1],
        [1.6, 0.9, 0.6]
    ],dtype=torch.float32)

    Y = torch.tensor([
        [0.2,0.5,0.3],
        [0.6,0.1,0.3],
        [1,0,0]
    ],dtype=torch.float32)
    fs = BFSPML_torch(X= X, Y=Y, sigma=2,k=1)
    s1,s2 = fs.feature_score(feature_index=[0, 1, 2])
    print(s1)
    print(s2)
