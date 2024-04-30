import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def crossval_plsr(X, y, ncomp, cv=5, message=False):
    """
    Fit PLS Regression model.
        Arguments: 
            X - independent values data (absorbance)
            y - dependent values data
            n_comp - number of components
        Returns:
            y_cv - predicted values
            r2 - R squared
            r - pearson R
            rmse - root mean squared error

    Maybe redo with stratified later
    """

    pls = PLSRegression(n_components=ncomp)

    y_cv = cross_val_predict(pls, X, y, cv=cv)

    r2 = r2_score(y, y_cv)
    r = pearsonr(np.ravel(y), np.ravel(y_cv))[0]
    rmse = np.sqrt(mean_squared_error(y, y_cv))

    if message:
        message = f"RMSE = {rmse:.4f}\nR2 = {r2:.4f}\nR = {r:.4f}"
        header = f"Crossval PLSR"
        print_boxed_output(message, header)
    
    return y_cv, r2, rmse


def plsr_r2_plot(y_true, y_predicted):
    """
    Plot y true against y predicted
    """
    fig, axs = plt.subplots(figsize=(6, 4))
    
    r2 = r2_score(y_true, y_predicted)
    r = pearsonr(y_true, y_predicted)[0]
    mse = mean_squared_error(y_true, y_predicted)
    rmse = np.sqrt(mse)

    axs.scatter(y_true, y_predicted, facecolor='w', edgecolor='k')
    axs.plot(y_true, y_true, color='k')
    axs.set_xlabel("y true")
    axs.set_ylabel("y predicted")
    plt.text(min(y_true), max(y_predicted), f"$R^2 = {r2:.3f}$\n$R = {r:.3f}$\nRMSE$ = {rmse:.3f}$", verticalalignment='top')

    return fig, axs


def residual_plot(y_true: np.array, y_predicted: np.array, num_std: int = None):
    """
    Plots residuals with std if defined
    """

    residuals = (y_true - y_predicted).to_numpy()
    mean = residuals.mean()
    std = residuals.std()

    plt.scatter(np.arange(len(residuals)), residuals, facecolor='w', edgecolor='k')
    plt.axhline(y=mean, color='k', linestyle='--')

    plt.xlabel("Sample")
    plt.ylabel("Residuals")

    plt.xlim([-5, len(y_true) + 5])
    
    if num_std is not None:
        plt.fill_between((-5, len(y_true) + 5), mean + num_std * std, mean - num_std * std, alpha=0.1)
        indices_beyond_std = np.where((residuals > mean + num_std * std) | (residuals < mean - num_std * std))[0]
        plt.scatter(indices_beyond_std, residuals[indices_beyond_std], edgecolor='r', facecolor='w')

        return indices_beyond_std
    
    return 


def calculate_total_iterations(start, end, step_size=1):
    return np.floor((end - start) / step_size)


def print_boxed_output(string, header):
    """ Printed output """

    length_string = max(len(line) for line in string.split('\n'))
    length_header = max(len(line) for line in header.split('\n'))

    length = max(length_string, length_header)
    
    header_padding = (length - len(header)) // 2

    print("┌" + "─" * (length + 2) + "┐")
    print(f"│ {' ' * header_padding}{header}{' ' * (length - len(header) - header_padding)} │")
    print("├" + "─" * (length + 2) + "┤")

    for line in string.split('\n'):
        print(f"│ {line.ljust(length)} │")

    print("└" + "─" * (length + 2) + "┘")


def find_number_components(X, y, number_range, cv, message=True, returns='list'):
    """ 
    Find an optimal number of components for PLS regression. 
    Returns:
        1) list
        2) values
    """

    r2s = []
    rmses = []
    y_cvs = []
    rs = []

    for num_components in range(*number_range):
        y_cv, r2, rmse = crossval_plsr(X, y, ncomp=num_components, cv=cv)

        r2s.append(r2)
        rs.append(pearsonr( np.ravel(y), np.ravel(y_cv) )[0])
        rmses.append(rmse)
        y_cvs.append(y_cv)

    number = np.argmax(r2s)
    r2 = r2s[number]
    r = rs[number]
    rmse = rmses[number]

    if message:
        text = f"RMSE = {rmse:.4f}\nR2 = {r2:.4f}\nR = {r:.4f}\nNumber of components = {number + 1}"
        header = "FIT RESULTS"
        print_boxed_output(text, header)

    if returns == 'list':
        return y_cvs, r2s, rmses
    
    if returns == 'values':
        return number + 1, r2, rmse



def plot_number_components(X, y, number_range=(1, 20), cv=5):
    """
    Produces plots based on find_number_components
    """
    y_cvs, r2s, rmses = find_number_components(X, y, number_range, cv=cv, returns='list')

    ncomp = np.argmax(r2s)
    y_cv = np.array(y_cvs)[ncomp]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(np.arange(*number_range), r2s, marker='o', color='k')
    axs[0, 1].plot(np.arange(*number_range), rmses, marker='o', color='k')
    axs[0, 0].set_xlabel("Num of components")
    axs[0, 1].set_xlabel("Num of components")
    axs[0, 0].set_ylabel("$R^2$")
    axs[0, 1].set_ylabel("RMSE")

    axs[1, 0].scatter(y, y_cv, facecolor='w', edgecolor='k')
    axs[1, 0].plot(y, y, color='k', linewidth=1.2)
    axs[1, 0].set_xlabel("y true")
    axs[1, 0].set_ylabel("y predicted")

    residuals = y - y_cv
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    axs[1, 1].scatter(np.arange(len(residuals)), residuals, facecolor='w', edgecolor='k')
    axs[1, 1].axhline(y=mean_residual, color='k', linestyle='--')
    axs[1, 1].set_xlabel("Sample")
    axs[1, 1].set_ylabel("Residuals")
    axs[1, 1].set_xlim([-5, len(y) + 5])

    return fig, axs


def cross_val_montecarlo(X, y, ncomp, *, n_splits=200, test_size=0.5, message=False):
    """
    Monte Carlo cross validation.
    Detailed explanation:
        Xu, Qing-Song, and Yi-Zeng Liang. "Monte Carlo cross validation." 
        Chemometrics and Intelligent Laboratory Systems 56.1 (2001): 1-11.
    """

    model = PLSRegression(n_components=ncomp)
    rmses = []
    r2s = []
    rs = []

    for _ in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predicted)
        r2 = r2_score(y_test, predicted)
        r = pearsonr(np.ravel(y_test), np.ravel(predicted))[0]

        rmses.append(np.sqrt(mse))
        r2s.append(r2)
        rs.append(r)

    mean_rmse = np.mean(rmses)
    mean_r2 = np.mean(r2s)
    mean_r = np.mean(rs)

    if message:
        message = f"RMSE = {np.sqrt(mse):.4f}\nR2 = {r2:.4f}\nR = {r:.4f}"
        header = f"MCCV"
        print_boxed_output(message, header)

        return None
    
    return mean_r2, mean_r, mean_rmse


def read_spa(filepath):
    '''
    Input
    Read a file (string) *.spa
    ----------
    Output
    Return spectra, wavelength
    '''
    with open(filepath, 'rb') as f:
        f.seek(564)
        Spectrum_Pts = np.fromfile(f, np.int32,1)[0]
        f.seek(30)
        SpectraTitles = np.fromfile(f, np.uint8,255)
        SpectraTitles = ''.join([chr(x) for x in SpectraTitles if x!=0])

        f.seek(576)
        Max_Wavenum=np.fromfile(f, np.single, 1)[0]
        Min_Wavenum=np.fromfile(f, np.single, 1)[0]
        # print(Min_Wavenum, Max_Wavenum, Spectrum_Pts)
        Wavenumbers = np.flip(np.linspace(Min_Wavenum, Max_Wavenum, Spectrum_Pts))

        f.seek(288);

        Flag=0
        while Flag != 3:
            Flag = np.fromfile(f, np.uint16, 1)

        DataPosition=np.fromfile(f,np.uint16, 1)
        f.seek(DataPosition[0])

        Spectra = np.fromfile(f, np.single, Spectrum_Pts)
        
    return np.stack((Wavenumbers, Spectra), axis=1)


def minmax(array: np.array):
    """ minmax sample scaling """
    return (array - array.min()) / (array.max() - array.min())