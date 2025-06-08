#mainpy

#feature selection algorithm
#Harit Talwar, Dani Cruz

from search import FeatureSearcher

import numpy as np
import time
from typing import Tuple, Set
from search import FeatureSearcher
import os

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load datasets in the expected format.
    Enhanced to handle both space-separated and other formats.
    """
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    # Handle both space-separated and comma-separated
                    if ',' in line:
                        values = [float(x.strip()) for x in line.split(',') if x.strip()]
                    else:
                        # Split by multiple spaces (as in your current format)
                        values = [float(x) for x in line.split() if x]
                    
                    if values:  # Only add non-empty rows
                        data.append(values)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file '{filename}' not found.")
    except ValueError as e:
        raise ValueError(f"Invalid data format in dataset file: {e}")
    
    if not data:
        raise ValueError("Dataset file is empty.")
    
    data = np.array(data)
    labels = data[:, 0].astype(int)  # Class labels in first column
    features = data[:, 1:]  # Features in remaining columns
    
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
    #nearest neighbor classifier supporting both nn (k=1) and knn (k>1)
    #enhanced version that supports different k values as required in part iii
    
    def __init__(self, k: int = 1):
        #initialize classifier
        
        #k: number of neighbors to consider (default=1 for nn)
        self.k = k
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
        #optimized test method with vectorized distance calculation
        if self.training_features is None:
            raise ValueError("Classifier must be trained before testing")
        
        #vectorized distance calculation for better performance
        distances = np.sqrt(np.sum((self.training_features - test_instance) ** 2, axis=1))
        
        #get k nearest neighbors indices
        k_nearest_indices = np.argpartition(distances, min(self.k-1, len(distances)-1))[:self.k]
        
        if self.k == 1:
            return self.training_labels[k_nearest_indices[0]]
        
        #majority voting for k > 1
        k_nearest_labels = self.training_labels[k_nearest_indices]
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
    
    def set_k(self, k: int):
        #change the k value for the classifier
        
        #k: new k value
        self.k = k


    
class LeaveOneOutValidator:
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        self.num_instances = len(labels)
        self.validation_count = 0  # Add this line
    
    def validate(self, feature_subset: Set[int], classifier) -> float:
        if not feature_subset:  #empty
            return 0.0
        
        self.validation_count += 1  # Add this line
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
    
    def reset_validation_count(self):
        #reset the validation counter for new runs
        self.validation_count = 0
    
    def get_validation_count(self) -> int:
        #get the number of validations performed
        return self.validation_count



def test_classifier():
   #test classifier and validator
   print("Testing classifier and validator...")
   
   #test small dataset
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
   
    #test large dataset
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
    #part three requirements added
    print("Welcome to Harit Talwar and Daniela Cruz Feature Selection Algorithm.")
    print("Part III: Complete Implementation with Real Datasets\n")
    
    #option to test classifier first
    test_choice = input("Do you want to test the classifier first? (y/n): ").lower().strip()
    if test_choice == 'y':
        test_classifier_comprehensive()
        print("\nTesting complete.\n")
    
    #get dataset choice
    dataset_choice = input("Use real dataset? (y/n - n for Part I dummy evaluation): ").lower().strip()
    
    if dataset_choice == 'n':
        # Part I: dummy evaluation
        print("\n=== Part I: Testing Search Algorithm Structure ===")
        try:
            num_features = int(input("Please enter total number of features: "))
            if num_features <= 0:
                print("Number of features must be positive!")
                return
        except ValueError:
            print("Please enter a valid number!")
            return
        
        searcher = FeatureSearcher(num_features)
        
        print("\nType the number of the algorithm you want to run.")
        print("1. Forward Selection")
        print("2. Backward Elimination")
        
        choice = input("Choice: ").strip()
        if choice not in ["1", "2"]:
            print("Invalid choice! Please enter 1 or 2.")
            return
        
        run_feature_search_with_timing(searcher, choice, f"Dummy Dataset ({num_features} features)")
        
    else:
        # Part III: real evaluation
        print("\n=== Part III: Real Dataset Analysis ===")
        filename = input("Enter dataset filename: ")
        
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            return
        
        try:
            #load and preprocess data
            print("Loading and normalizing data...")
            start_time = time.time()
            features, labels = load_dataset(filename)
            features = normalize_features(features)
            load_time = time.time() - start_time
            
            print(f"Data loaded in {load_time:.2f} seconds")
            print(f"This dataset has {features.shape[1]} features (not including the class attribute), "
                  f"with {len(labels)} instances.")
            print("Please wait while I normalize the data... Done!")
            
                #set up validator
            validator = LeaveOneOutValidator(features, labels)
            
            #run both algorithms and compare results
            print("\n=== Running Both Algorithms for Comparison ===")
            
            results = {}
            k_values = [1, 3, 5, 7]  # Part III requirement
            
            for k in k_values:
                print(f"\n--- Testing with k={k} ---")
                classifier = NearestNeighborClassifier(k=k)
                searcher = FeatureSearcher(features.shape[1])
                searcher.set_real_evaluation(validator, classifier)
                
                #forward selection
                print(f"\nForward Selection (k={k}):")
                validator.reset_validation_count()
                start_time = time.time()
                forward_features, forward_acc = searcher.forward_selection(verbose=True)
                forward_time = time.time() - start_time
                forward_validations = validator.get_validation_count()
                
                #backward elimination
                print(f"\nBackward Elimination (k={k}):")
                validator.reset_validation_count()
                start_time = time.time()
                backward_features, backward_acc = searcher.backward_elimination(verbose=True)
                backward_time = time.time() - start_time
                backward_validations = validator.get_validation_count()
                
                #store results
                results[k] = {
                    'forward': {
                        'features': forward_features,
                        'accuracy': forward_acc,
                        'time': forward_time,
                        'validations': forward_validations
                    },
                    'backward': {
                        'features': backward_features,
                        'accuracy': backward_acc,
                        'time': backward_time,
                        'validations': backward_validations
                    }
                }
            
            #print results summary
            print_comprehensive_results(results, filename)
            
        except Exception as e:
            print(f"Error processing dataset: {e}")
            return

def print_comprehensive_results(results: dict, filename: str):

    #print comprehensive results
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE RESULTS SUMMARY FOR {filename.upper()}")
    print(f"{'='*60}")
    
    print("\n--- ALGORITHM COMPARISON ---")
    for k in sorted(results.keys()):
        print(f"\nk = {k}:")
        forward = results[k]['forward']
        backward = results[k]['backward']
        
        print(f"  Forward Selection:")
        features_str = ','.join(map(str, sorted(forward['features']))) if forward['features'] else 'None'
        print(f"    Features: {{{features_str}}}")
        print(f"    Accuracy: {forward['accuracy']:.1%}")
        print(f"    Time: {forward['time']:.2f}s")
        print(f"    Validations: {forward['validations']}")
        
        print(f"  Backward Elimination:")
        features_str = ','.join(map(str, sorted(backward['features']))) if backward['features'] else 'None'
        print(f"    Features: {{{features_str}}}")
        print(f"    Accuracy: {backward['accuracy']:.1%}")
        print(f"    Time: {backward['time']:.2f}s")
        print(f"    Validations: {backward['validations']}")
    
    #find thebest overall results
    print(f"\n--- BEST RESULTS ---")
    best_forward = max(results.values(), key=lambda x: x['forward']['accuracy'])['forward']
    best_backward = max(results.values(), key=lambda x: x['backward']['accuracy'])['backward']
    
    print(f"Best Forward Selection:")
    features_str = ','.join(map(str, sorted(best_forward['features']))) if best_forward['features'] else 'None'
    print(f"  Features: {{{features_str}}}, Accuracy: {best_forward['accuracy']:.1%}")
    
    print(f"Best Backward Elimination:")
    features_str = ','.join(map(str, sorted(best_backward['features']))) if best_backward['features'] else 'None'
    print(f"  Features: {{{features_str}}}, Accuracy: {best_backward['accuracy']:.1%}")
    
    #knn comparison
    print(f"\n--- KNN PERFORMANCE COMPARISON ---")
    print("Algorithm\t\tk=1\t\tk=3\t\tk=5\t\tk=7")
    print("-" * 50)
    
    forward_accs = [results[k]['forward']['accuracy'] for k in [1,3,5,7]]
    backward_accs = [results[k]['backward']['accuracy'] for k in [1,3,5,7]]
    
    print(f"Forward\t\t\t{forward_accs[0]:.1%}\t\t{forward_accs[1]:.1%}\t\t{forward_accs[2]:.1%}\t\t{forward_accs[3]:.1%}")
    print(f"Backward\t\t{backward_accs[0]:.1%}\t\t{backward_accs[1]:.1%}\t\t{backward_accs[2]:.1%}\t\t{backward_accs[3]:.1%}")

    def generate_trace_output(results: dict, dataset_name: str):

        #geneate trace output
        print(f"\n{'='*60}")
        print(f"TRACE OUTPUT FOR REPORT - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        for k in sorted(results.keys()):
            print(f"\n{dataset_name} Results, k = {k}:")
            
            for algorithm in ['forward', 'backward']:
                alg_name = "Forward Selection" if algorithm == 'forward' else "Backward Elimination"
                result = results[k][algorithm]
                
                print(f"\n{alg_name}:")
                features_str = ','.join(map(str, sorted(result['features']))) if result['features'] else 'None'
                print(f"Feature Subset: {{{features_str}}}, Acc: {result['accuracy']:.1%}")


if __name__ == "__main__":
    main()
