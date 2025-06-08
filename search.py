#search.py

#featurre search algorithm
#forward selection
#backwards elimination

#main ideas:

# feature search = find optimal subset of features that gives best performance for model
# forward selection = star with empty SET of featuires, iteratively add one feature at a time, for each iteration - try adding each
# remaining feature and evaluate performance. keep the one that gives the best improvement. continue until no further improvement is possible.

#backwards elimination= opposite of forward selection, start with all features, iteratively remove instead of add, least decrease in performance
#continue until removing more features hurts performance

#handle feature search algorithm here

import random
from typing import Set, Tuple

class FeatureSearcher:

    #take number of features + optimal evaluation function
    #dummy evaluation function = random accuracy (set up for testing)
    #start wuth no features
    #try adding available feature one by one, keep track of best, return best + accuracy
    
    def __init__(self, num_features: int, evaluation_function=None):
        self.num_features = num_features
        self.evaluation_function = evaluation_function or self._dummy_evaluation
    
    def _dummy_evaluation(self, feature_subset: Set[int]) -> float:
        #random accuracy
        return random.uniform(0.0, 1.0)
    
    def forward_selection(self, verbose: bool = True) -> Tuple[Set[int], float]:
        #forward selection
        #start with empty set, iteratively add best feature

        if verbose:
            print("Using no features and evaluation function, I get an accuracy of", end=" ")
        
        current_features = set()
        baseline_accuracy = self.evaluation_function(current_features)
        
        if verbose:
            print(f"{baseline_accuracy:.1%}")
            print("Beginning search.\n")
        
        best_overall_features = set()
        best_overall_accuracy = baseline_accuracy
        available_features = set(range(1, self.num_features + 1))
        
        while available_features:
            best_feature_to_add = None
            best_accuracy = -1
            
            # Try adding each remaining feature
            for feature in available_features:
                test_features = current_features | {feature}
                accuracy = self.evaluation_function(test_features)
                
                if verbose:
                    features_str = ','.join(map(str, sorted(test_features)))
                    print(f"Using feature(s) {{{features_str}}} accuracy is {accuracy:.1%}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature_to_add = feature
            
            #check if adding the best feature improves overall accuracy
            if best_accuracy > best_overall_accuracy:
                current_features.add(best_feature_to_add)
                available_features.remove(best_feature_to_add)
                best_overall_features = current_features.copy()
                best_overall_accuracy = best_accuracy
                
                if verbose:
                    features_str = ','.join(map(str, sorted(current_features)))
                    print(f"Feature set {{{features_str}}} was best, accuracy is {best_accuracy:.1%}\n")
            else:
                if verbose:
                    print("(Warning: Accuracy has decreased! Stopping search.)")
                break
        
        if verbose:
            features_str = ','.join(map(str, sorted(best_overall_features)))
            print(f"Search finished! The best feature subset is {{{features_str}}}, "
                f"which has an accuracy of {best_overall_accuracy:.1%}")
        
        return best_overall_features, best_overall_accuracy

    def backward_elimination(self, verbose: bool = True) -> Tuple[Set[int], float]:
        #backeards elimination
        #start with all features, iteratively remove worst feature
        #continue until removing more features hurts performance

        if verbose:
            print("Starting with all features and evaluation function, I get an accuracy of", end=" ")
        
        current_features = set(range(1, self.num_features + 1))
        baseline_accuracy = self.evaluation_function(current_features)
        
        if verbose:
            print(f"{baseline_accuracy:.1%}")
            print("Beginning search.\n")
        
        best_overall_features = current_features.copy()
        best_overall_accuracy = baseline_accuracy
        
        while len(current_features) > 1:
            best_feature_to_remove = None
            best_accuracy = -1
            
            # Try removing each feature
            for feature in current_features.copy():
                test_features = current_features - {feature}
                accuracy = self.evaluation_function(test_features)
                
                if verbose:
                    if test_features:
                        features_str = ','.join(map(str, sorted(test_features)))
                        print(f"Using feature(s) {{{features_str}}} accuracy is {accuracy:.1%}")
                    else:
                        print(f"Using no features, accuracy is {accuracy:.1%}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature_to_remove = feature
            
            #check ifremoving the feature improves overall accuracy
            if best_accuracy > best_overall_accuracy:
                current_features.remove(best_feature_to_remove)
                best_overall_features = current_features.copy()
                best_overall_accuracy = best_accuracy
                
                if verbose:
                    if current_features:
                        features_str = ','.join(map(str, sorted(current_features)))
                        print(f"Feature set {{{features_str}}} was best, accuracy is {best_accuracy:.1%}\n")
                    else:
                        print(f"Empty feature set was best, accuracy is {best_accuracy:.1%}\n")
            else:
                if verbose:
                    print("(Warning: Accuracy has decreased! Stopping search.)")
                break
        
        if verbose:
            if best_overall_features:
                features_str = ','.join(map(str, sorted(best_overall_features)))
                print(f"Search finished! The best feature subset is {{{features_str}}}, "
                      f"which has an accuracy of {best_overall_accuracy:.1%}")
            else:
                print(f"Search finished! The best feature subset is empty, "
                      f"which has an accuracy of {best_overall_accuracy:.1%}")
        
        return best_overall_features, best_overall_accuracy

    def set_real_evaluation(self, validator, classifier):
        #set up real evaluation function using leave-one-out validation
        def real_eval(feature_subset):
            if not feature_subset:  #empty set
                return 0.0
            return validator.validate(feature_subset, classifier)
        
        self.evaluation_function = real_eval

    def get_final_accuracy(self) -> float:
        """Get the final accuracy from the last search."""
        if hasattr(self, 'search_history') and self.search_history:
            return self.search_history[-1]['accuracy']
        return 0.0

    def get_total_evaluations(self) -> int:
        """Get total number of feature subset evaluations performed."""
        if hasattr(self, 'search_history'):
            return sum(len(step.get('all_tested', [])) for step in self.search_history)
        return 0
