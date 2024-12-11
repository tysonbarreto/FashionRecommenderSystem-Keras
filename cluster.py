from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import Sequence, Union
import numpy as np
import tensorflow as tf

@dataclass
class Cluster:
    '''
    Cluster class uses NearestNeighbors from sklearn
    '''
    feature_list:Union[np.array,tf.Tensor]
    n_neighbors:int=6
    algorithm:str="brute"
    metric:str="euclidean"

    def __post_init__(self):
        self.neighbors = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm, metric=self.metric)
        self.neighbors.fit(self.feature_list)

    def ClassifyNeighbors(self, matrix_like:Sequence[Union[np.array,tf.Tensor]], return_distance:bool=False):
        return self.neighbors.kneighbors(X=matrix_like, return_distance=return_distance)

if __name__=="__main__":
    __all__=["Cluster"]