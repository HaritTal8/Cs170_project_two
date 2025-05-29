# Search.py

# Feature search algorithm
# Forward selection
# Backwards elimination

# Main ideas:

# Feature search = find optimal subset of features that gives best performance for model
# Forward selection = star with empty SET of featuires, iteratively add one feature at a time, for each iteration - try adding each
# remaining feature and evaluate performance. keep the one that gives the best improvement. continue until no further improvement is possible.

# Backwards elimination= opposite of forward selection, start with all features, iteratively remove instead of add, least decrease in performance
# continue until removing more features hurts performance

# Handle feature search algorithm here

import random
from typing import Set, Tuple

class FeatureSearcher:

    # Take number of features + optimal evaluation function
    # Dummy evaluation function = random accuracy (set up for testing)
    
    def __init__(self, num_features: int, evaluation_function=None):
        self.num_features = num_features
        self.evaluation_function = evaluation_function or self._dummy_evaluation
    
    def _dummy_evaluation(self, feature_subset: Set[int]) -> float:
        #random accuracy
        return random.uniform(0.0, 1.0)
    
    # Forward Selection
    # Starts with no features; tries adding available features one by one, keeps track of best, returns best track and its accuracy
    def forward_selection(self) -> Tuple[Set[int], float]:
        #forward selection
        print("Using no features and \"random\" evaluation, I get an accuracy of")
        current_features = set()
        baseline_accuracy = self.evaluation_function(current_features)
        print(f"{baseline_accuracy:.1%} Beginning search.\n")
        
        # Empty features set
        best_overall_features = set()
        best_overall_accuracy = baseline_accuracy

        # Create set for all available features
        available_features = set(range(1, self.num_features + 1))
        
        # While we still have features that can be tested
        while available_features:
            best_feature_to_add = None
            best_accuracy = -1
            
            # Attempt adding each remaining feature
            for feature in available_features:
                test_features = current_features | {feature}
                accuracy = self.evaluation_function(test_features)
                print(f"Using feature(s) {{{','.join(map(str, sorted(test_features)))}}} accuracy is {accuracy:.1%}")
                
                # Check which feature gives the best accuracy and overall accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature_to_add = feature
            
            # Check if adding the best feature improves OVERALL accuracy
            if best_accuracy > best_overall_accuracy:
                current_features.add(best_feature_to_add)
                available_features.remove(best_feature_to_add)
                best_overall_features = current_features.copy()
                best_overall_accuracy = best_accuracy
                print(f"Feature set {{{','.join(map(str, sorted(current_features)))}}} was best, accuracy is {best_accuracy:.1%}\n")
            else:
                # Stop searching if no improvement in overall performance found 
                print("(Warning: Decreased accuracy!)")
                break
        
        print(f"Search finished! The best subset of features is {{{','.join(map(str, sorted(best_overall_features)))}}}, which has an accuracy of {best_overall_accuracy:.1%}")
        return best_overall_features, best_overall_accuracy
        
    # Backwards Elimination:
    # Start with all features; iteratively remove the features that have the least performance improvement
    
    def backward_elimination(self) -> Tuple[Set[int], float]:
        print("Starting with all features and \"random\" evaluation, I get an accuracy of")
        current_features = set(range(1, self.num_features + 1))
        baseline_accuracy = self.evaluation_function(current_features)
        print(f"{baseline_accuracy:.1%} Beginning search.\n")
        
        # Track variables
        best_overall_features = current_features.copy()
        best_overall_accuracy = baseline_accuracy
        
         # While we have at least 1 feature, continue looking for best accuracy in that iteration
        while len(current_features) > 1:
            best_feature_to_remove = None
            best_accuracy = -1
            
            # Try removing each feature, one at a time
            for feature in current_features:
                test_features = current_features - {feature}
                accuracy = self.evaluation_function(test_features)
                print(f"Using feature(s) {{{','.join(map(str, sorted(test_features)))}}} accuracy is {accuracy:.1%}")
                
                # Keep track of best performance
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature_to_remove = feature
            
            # Check if removing the feature improves OVERALL accuracy
            if best_accuracy > best_overall_accuracy:
                current_features.remove(best_feature_to_remove)  # UPDATE current_features
                best_overall_features = current_features.copy()
                best_overall_accuracy = best_accuracy
                print(f"Feature set {{{','.join(map(str, sorted(current_features)))}}} was best, accuracy is {best_accuracy:.1%}\n")
            else:
                # Stop searching if no improvement in overall performance found 
                print("(Warning: Decreased accuracy!)")
                break
        
        print(f"Search finished! The best subset of features is {{{','.join(map(str, sorted(best_overall_features)))}}}, which has an accuracy of {best_overall_accuracy:.1%}")
        return best_overall_features, best_overall_accuracy


    def set_real_evaluation(self, validator, classifier):
        #set up real evaluation function using leave-one-out validation
        def real_eval(feature_subset):
            if not feature_subset:  #empty set
                return 0.0
            return validator.validate(feature_subset, classifier)
        
        # Swap evaluation function with real one
        self.evaluation_function = real_eval