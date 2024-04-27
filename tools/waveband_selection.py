import numpy as np
import pandas as pd
from analysis import cross_val_montecarlo, find_number_components
from tqdm.auto import tqdm


class VISSA:
    """
     Variable Iterative Space Shrinkage Approach (VISSA) algorith for variable
     selection inspired by Model Population Analysis.
     Original paper: Analyst, 2014,139, 4836-4845
    """
    
    def fit(self, X, y, ncomp=5, n_splits=50, num_iter=1, num_samples=2000):
        n_samples, n_features = X.shape
        
        self.weights = np.ones(n_features) * 0.5
        self.num_samples = num_samples

        for i in range(num_iter):
            self.sampled_features = sample_features(self.weights, self.num_samples)
            progress_bar = tqdm(total=len(self.sampled_features), desc=f"Iteration #{i + 1}")
            progress_bar.colour = '#000000'
    
            r2s = []
            rmses = []
    
            for features in self.sampled_features:
                selected_features = X.iloc[:, features.astype(bool)]
    
                rmse, r2 = cross_val_montecarlo(selected_features, y, ncomp=ncomp, n_splits=n_splits)
                #rmse, r2 = find_number_components(selected_features, y, (1, 10))
                    
                r2s.append(r2)
                rmses.append(rmse)
                progress_bar.update(1)

                num_select = int(self.num_samples * 0.05)
                indices = np.argsort(rmses)[:num_select]

            print("R2: ", np.mean(r2s))
            print("RMSE: ", np.mean(rmses))
            
            freq = np.mean(self.sampled_features[indices], axis=0)
            self.weights = freq
            progress_bar.close()
            
        return self.weights
    

def sample_features(prob1d: np.array, num_samplings: int) -> np.array:
    """
    Creates a bool matrix (K x P) where K is the number of samplings and P is the number of 
    features. Each row (1, 2, ... , K) is a different sample where for each feature (1, 2, ... , P)
    we can have 0 (don't chose) or 1 (choose).

    Arguments:
        prob1d - vectors of weights for each feature (P)
        num_samplings - number of random samplings (K)

    Returns:
        (K x P) np.array
    """

    prob2d = np.tile(prob1d, (num_samplings, 1))
    sample = np.random.binomial(n=1, p=prob2d, size=(num_samplings, len(prob1d)))
    permutation = np.random.permutation(sample)
    
    return permutation