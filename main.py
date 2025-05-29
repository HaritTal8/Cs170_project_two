# Feature Selection Algorithm
# Harit Talwar, Dani Cruz

import numpy as np
import time
from typing import Tuple, Set
from search import FeatureSearcher

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    # Load datasets that are given
    # We assume that first column is assigned for class label and all the other ones are features
    dataset = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                dataset.append(values)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file '{filename}' not found.")
    except ValueError:
        raise ValueError("Invalid data format in dataset file.")
    
    dataset = np.array(dataset)
    labels = dataset[:, 0].astype(int)  #class first column
    features = dataset[:, 1:]  #features
    
    return features, labels

def normalize_features(features: np.ndarray) -> np.ndarray:
    # Features have to be normalized (set mean to 0 and standard deviation to 1)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Avoid division by zero for constant features
    std[std == 0] = 1
    return (features - mean) / std

def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    # Euclidean distance between vectors
    return np.sqrt(np.sum((point1 - point2) ** 2))

class NearestNeighborClassifier:
    # Nearest neighbor classifier
    
    def __init__(self):
        self.training_features = None
        self.training_labels = None
        
    def train(self, features: np.ndarray, labels: np.ndarray, feature_subset: Set[int] = None):
        # Train classifier
        
        # Features: training feature vectors (normalized)
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
        # Classify test
        # Using nearest neighbor and feature vector of test instance, predict class label
        if self.training_features is None:
            raise ValueError("Classifier must be trained before testing")
        
        closest_dist = float('inf')
        nearest_label = None
        
        # Nearest training instance
        for i, training_instance in enumerate(self.training_features):
            distance = euclidean_distance(test_instance, training_instance)
            if distance < closest_dist:
                closest_dist = distance
                nearest_label = self.training_labels[i]
        
        return nearest_label
    
class LeaveOneOutValidator:

    # Leave one out cross validation for classifier evaluation
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Initialize validator with dataset
        
        # Features: normalized matrix
        # Labels: class labels

        self.features = features
        self.labels = labels
        self.num_instances = len(labels)
    
    def validate(self, feature_subset: Set[int], classifier: NearestNeighborClassifier) -> float:
        # Leave on out validation
        # Feature_subset: set of feature indices to use (1-indexed)
        # Classifier: classifier to evaluate
        # Returned accuracy between 0 and 1
        if not feature_subset:  #empty
            return 0.0
            
        correct_predictions = 0
        
        # 1-indexed to 0-indexed
        feature_indices = [i-1 for i in feature_subset]
        
        for i in range(self.num_instances):
            # Training set (all instances except i)
            train_indices = list(range(self.num_instances))
            train_indices.remove(i)
            
            train_features = self.features[train_indices]
            train_labels = self.labels[train_indices]
            
            # Test instance
            test_features = self.features[i, feature_indices]
            true_label = self.labels[i]
            
            # Classifier makes prediction from train
            classifier.train(train_features, train_labels, feature_subset)
            predicted_label = classifier.test(test_features)
            
            if predicted_label == true_label:
                correct_predictions += 1
        
        return correct_predictions / self.num_instances


def test_classifier():
   # Compute tests in large and small datasets to test accuracy

   # Test classifier and validator
   print("Testing classifier and validator...")
   
   # Test small dataset
   print("\nTesting small dataset with features {3, 5, 7}:")
   try:
       features, labels = load_dataset('small_dataset.txt')
       features = normalize_features(features)
       
       validator = LeaveOneOutValidator(features, labels)
       classifier = NearestNeighborClassifier()
       
       accuracy = validator.validate({3, 5, 7}, classifier)
       print(f"Accuracy: {accuracy:.3f} (Expected: ~0.89)")
       
       if abs(accuracy - 0.89) < 0.05:
           print("Test PASSED - Accuracy within expected range")
       else:
           print("Test result differs from expected value")
           
   except FileNotFoundError:
       print("Small dataset not found. Please ensure small_dataset.txt exists.")
   except Exception as e:
       print(f"Error testing small dataset: {e}")
   
    # Test large dataset
   print("\nTesting large dataset with features {1, 15, 27}:")
   try:
       features, labels = load_dataset('large_dataset.txt')
       features = normalize_features(features)
       
       validator = LeaveOneOutValidator(features, labels)
       classifier = NearestNeighborClassifier()
       
       accuracy = validator.validate({1, 15, 27}, classifier)
       print(f"Accuracy: {accuracy:.3f} (Expected: ~0.949)")
       
       if abs(accuracy - 0.949) < 0.05:
           print("Test PASSED - Accuracy within expected range")
       else:
           print("Test result differs from expected value")
           
   except FileNotFoundError:
       print("Large dataset not found. Please ensure large_dataset.txt exists.")
   except Exception as e:
       print(f"Error testing large dataset: {e}")

def main():
    # Main entry point for the feature selection application
    print("Welcome to Harit Talwar and Daniela Cruz Feature Selection Algorithm.")
    
    # Option to test classifier first
    test_choice = input("Do you want to test the classifier first? (y/n): ").lower()
    if test_choice == 'y':
        test_classifier()
        return
    
    # Get dataset file or run part one
    dataset_choice = input("Use real dataset? (y/n - n for Part I dummy evaluation): ").lower()
    
    if dataset_choice == 'n':
        # Dummy evaluation
        try:
            num_features = int(input("Please enter total number of features: "))
            if num_features <= 0:
                print("Number of features must be positive!")
                return
        except ValueError:
            print("Please enter a valid number!")
            return
        
        searcher = FeatureSearcher(num_features)
        
    else:
        # Real evaluation
        filename = input("Enter dataset filename: ")
        
        try:
            start_time = time.time()
            print("Loading and normalizing data...")
            features, labels = load_dataset(filename)
            features = normalize_features(features)
            load_time = time.time() - start_time
            print(f"Data loaded in {load_time:.2f} seconds")
            
            num_features = features.shape[1]
            print(f"Dataset has {len(labels)} instances and {num_features} features")
            
            # Validator and classifier
            validator = LeaveOneOutValidator(features, labels)
            classifier = NearestNeighborClassifier()
            
            # Searcher with real evaluation
            searcher = FeatureSearcher(num_features)
            searcher.set_real_evaluation(validator, classifier)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
    
  # User selection of algorithm method
    print("\nType the number of the algorithm you want to run.")
    print("1 Forward Selection")
    print("2 Backward Elimination")
    
    choice = input().strip()
    
    start_time = time.time()
    if choice == "1":
        best_features, best_accuracy = searcher.forward_selection()
    elif choice == "2":
        best_features, best_accuracy = searcher.backward_elimination()
    else:
        print("Invalid choice! Please enter 1 or 2.")
        return
    
    search_time = time.time() - start_time
    print(f"\nSearch completed in {search_time:.2f} seconds")

if __name__ == "__main__":
    main()
