import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def isolation_forest(features: pd.DataFrame, 
                     index: bool = False, 
                     **kwargs) -> pd.DataFrame:
    """ 
    Isolation Forest, unsupervised learning algorithm for outlier detection
    Arguments:
        An array with features (pd.Series or pd.DataFrame),
        index - returns bool array if true
        **kwargs for IsolationForest from sklearn.ensemble
    Returns:
        bool array if index is True and DataFrame with scores otherwise
    """
    
    model = IsolationForest(**kwargs)

    model.fit(features)

    anomaly_scores = model.predict(features)
    scores = model.decision_function(features)
    select = anomaly_scores == 1
    
    df = pd.DataFrame({
        "Anomaly Scores": anomaly_scores,
        "Scores": scores
    }, index = features.index)

    if index:
        return select
    
    return df


class MonteCarloOutliers:
    """
    Outlier detection algorithm based on monte carlo cross validation.
    Original paper: 
            Liu, Z., Cai, W. & Shao, X. Outlier detection in near-infrared spectroscopic analysis 
            by using Monte Carlo cross-validation. Sci. China Ser. B-Chem. 51, 751–759 (2008). 
            https://doi.org/10.1007/s11426-008-0080-x
    """

    def fit(self, X, y, *, ncomp=5, n_models=1000, test_size=0.2):
        """
        Fits the model and stores the accumulated means (f_ac).
        """
        self.train_size = 1 - test_size
        
        model = PLSRegression(n_components=ncomp)
        press = []
    
        x_train_index = np.zeros((n_models, X.shape[0]), dtype=int)
    
        for i in range(n_models):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
            train_indices = X_train.index.to_numpy()
            x_train_index[i, train_indices] = 1
            
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            
            mse = mean_squared_error(y_test, predicted)
            r2 = r2_score(y_test, predicted)
    
            press.append(mse) # prediction residual error  sum of squares
            
        self.sorted_indecies = np.argsort(press)
        self.sorted_models = np.sort(press)
    
        x_train_index = x_train_index[self.sorted_indecies]
    
        rows, cols = x_train_index.shape
        self.accumulative_means = np.zeros((rows, cols))
        
        for i in range(rows):
            self.accumulative_means[i, :] = np.mean(x_train_index[:i+1, :], axis=0) * 100
    
        return None

    def plot_accumulative_means(self):
        plt.plot(self.accumulative_means, alpha=0.5, linewidth=0.8)
        plt.xlabel("Model rank")
        plt.ylabel("$f_{ac}$")
        plt.axhline(y=self.train_size * 100, color='k', linestyle='--')
        plt.xlim([0, 1000])

    def plot_samples(self, model_num, std_num=2):
        f_ac = self.accumulative_means[model_num]
        
        plt.scatter(np.arange(len(f_ac)), f_ac, edgecolor='k', facecolor='w')

        mean = f_ac.mean()
        std = f_ac.std()
        
        plt.axhline(y=mean, linestyle='--', color='k')
        plt.fill_between(np.arange(len(f_ac) + 1), mean + std_num * std, mean - std_num * std, alpha=0.1)
        plt.xlim([0, len(f_ac)])
        plt.xlabel("Sample Number")
        plt.ylabel("$f_{ac}$")
        
        indices_beyond_std = np.where(f_ac > mean + std_num * std)[0]
        plt.scatter(indices_beyond_std, f_ac[indices_beyond_std], edgecolor='r', facecolor='w')

        return indices_beyond_std