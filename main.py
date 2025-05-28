#mainpy

#feature selection algorithm
#Harit Talwar, Dani Cruz

from search import FeatureSearcher

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