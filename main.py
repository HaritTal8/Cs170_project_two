#mainpy

#feature selection algorithm
#Harit Talwar, Dani Cruz

from search import FeatureSearcher

import numpy as np
import time
from typing import Tuple, Set
from search import FeatureSearcher

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    #load datasets that are given
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                data.append(values)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file '{filename}' not found.")
    except ValueError:
        raise ValueError("Invalid data format in dataset file.")
    
    data = np.array(data)
    labels = data[:, 0].astype(int)  #class first column
    features = data[:, 1:]  #features
    
    return features, labels

def normalize_features(features: np.ndarray) -> np.ndarray:
    #features have to be normalized
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Avoid division by zero for constant features
    std[std == 0] = 1
    return (features - mean) / std

def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    #euclidean distance
    return np.sqrt(np.sum((point1 - point2) ** 2))

class NearestNeighborClassifier:
    #nearest neighbor classifier
    
    def __init__(self):
        self.training_features = None
        self.training_labels = None
        
    def train(self, features: np.ndarray, labels: np.ndarray, feature_subset: Set[int] = None):
        #train classifier
        
        #features: training feature vectors (normalized)
        #labels: training class labels
        #feature_subset: Set of feature indices to use (1-indexed), if None use all
        if feature_subset is not None:
            #convert 1-indexed to 0-indexed and select subset
            feature_indices = [i-1 for i in feature_subset]
            self.training_features = features[:, feature_indices]
        else:
            self.training_features = features
            
        self.training_labels = labels
    
    def test(self, test_instance: np.ndarray) -> int:
        #classify test
        #predict class label via nearest neighbor and feature vector of test instance
        if self.training_features is None:
            raise ValueError("Classifier must be trained before testing")
        
        min_distance = float('inf')
        nearest_label = None
        
        #nearest training instance
        for i, training_instance in enumerate(self.training_features):
            distance = euclidean_distance(test_instance, training_instance)
            if distance < min_distance:
                min_distance = distance
                nearest_label = self.training_labels[i]
        
        return nearest_label
    
class LeaveOneOutValidator:
    #leave one out cross validation for classifier evaluation
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        #initialize validator with dataset
        
        #features: matrix (normalized)
        #labels: class labels

        self.features = features
        self.labels = labels
        self.num_instances = len(labels)
    
    def validate(self, feature_subset: Set[int], classifier: NearestNeighborClassifier) -> float:
        #leave on out validation
        #feature_subset: set of feature indices to use (1-indexed)
        #classifier: classifier to evaluate
        #returned accuracy between 0 and 1
        if not feature_subset:  #empty
            return 0.0
            
        correct_predictions = 0
        
        #1-indexed to 0-indexed
        feature_indices = [i-1 for i in feature_subset]
        
        for i in range(self.num_instances):
            #training set (all instances except i)
            train_indices = list(range(self.num_instances))
            train_indices.remove(i)
            
            train_features = self.features[train_indices]
            train_labels = self.labels[train_indices]
            
            #test instance
            test_features = self.features[i, feature_indices]
            true_label = self.labels[i]
            
            #classifier makes prediction from train
            classifier.train(train_features, train_labels, feature_subset)
            predicted_label = classifier.test(test_features)
            
            if predicted_label == true_label:
                correct_predictions += 1
        
        return correct_predictions / self.num_instances


def main(): #main entry point for feature selection program
    #number of features from user
    try:
        num_features = int(input("Please enter total number of features: "))
        if num_features <= 0:
            print("Number of features must be positive!")
            return
    except ValueError:
        print("Please enter a valid number!")
        return
    
    #choice of algorithm
    print("Type the number of the algorithm you want to run.")
    print("1 Forward Selection")
    print("2 Backward Elimination")
    
    choice = input().strip()
    
    #searcher with dummy evaluation - part one
    searcher = FeatureSearcher(num_features)
    
    if choice == "1":
        searcher.forward_selection()
    elif choice == "2":
        searcher.backward_elimination()
    else:
        print("Invalid choice! Please enter 1 or 2.")

if __name__ == "__main__":
    main()