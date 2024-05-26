from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jaxfit import CurveFit

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import r2_score

def combined_pseudo_voigt(x, *params):
    """ 
    Voigt profile approximation by linear combination of
    Lorenzian and Gaussian distributinos.
    mu, A, A0, gamma gauss, gamma lorentz, eta
    """
    N = len(params) // 5
    result = jnp.zeros_like(x)
    for i in range(N):
        mu, gamma_gaussian, gamma_lorentzian, amplitude, eta = params[i*5:(i+1)*5]
        
        a_G = (2 / gamma_gaussian) * jnp.sqrt(jnp.log(2) / jnp.pi)
        b_G = (4 * jnp.log(2)) / (gamma_gaussian**2)
        gaussian_term = a_G * jnp.exp(-b_G * (x - mu)**2)
        
        lorentzian_term = (1 / jnp.pi) * ( (gamma_lorentzian / 2) / ( (x - mu)**2 + (gamma_lorentzian / 2)**2) ) 
        
        result += amplitude * (eta * gaussian_term + (1 - eta) * lorentzian_term)

    return result

def create_params_pseudo_voigt(x_values, y_values, peaks):
    """ 
    Creates initial parameters and bounds for the fit given
    a list of peaks 
    """
    params = []
    lower_bound = []
    upper_bound = []

    for i, peak in enumerate(peaks):
        wavenumber = x_values[peak]
        absorbance = y_values[peak]
        
        params.extend([
            wavenumber,      # center
            5,               # FWHM Gaussian
            5,               # FWHM Lorentzian
            absorbance,      # A
            0.5              # eta (initial guess, can be modified)
        ])

        lower_bound.extend([
            wavenumber - 5,      # Lower bound for center
            0,                   # Lower bound for Gaussian FWHM (sigma)
            0,                   # Lower bound for Lorentzian FWHM (gamma)
            0,                   # Lower bound for A
            0                    # Lower bound for eta
        ])

        upper_bound.extend([
            wavenumber + 5,      # Upper bound for center
            np.inf,              # Upper bound for Gaussian FWHM (sigma)
            np.inf,              # Upper bound for Lorentzian FWHM (gamma)
            np.inf,              # Upper bound for A
            1                    # Upper bound for eta
        ])
 
    return (lower_bound, upper_bound), params

def fit_pseudo_voigt(x_values, y_values, peaks, fig_show=False, wavenumber=False, residuals=False):
    """
    Curve fitting using jaxfit.
    """
    if wavenumber:
        peaks = [np.where(x_values == value)[0][0] for value in peaks]
    
    bounds, initial_guess = create_params_pseudo_voigt(x_values, y_values, peaks)
    
    jcf = CurveFit()
    params, pcov = jcf.curve_fit(combined_pseudo_voigt, x_values, y_values, p0=initial_guess, bounds=bounds, ftol=1e-12)

    predicted = combined_pseudo_voigt(x_values, *params)

    sum_of_squares = np.sum((predicted - y_values)**2)
    r2 = r2_score(y_values, predicted)
    
    n = len(y_values)
    k = len(params)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    chi_squared = np.sum((predicted - y_values)**2 / y_values)

    print(f"SS: {sum_of_squares:.3e}", 
          f"R2: {r2:.5f}", 
          f"adj R2: {adjusted_r2:.5f}",
         f"chi squared: {chi_squared:.5f}")

    fit_results_text = (
    f"<b>SS:</b> {sum_of_squares:.3e}<br>"
    f"<b>R2:</b> {r2:.5f}<br>"
    f"<b>adj R2:</b> {adjusted_r2:.5f}<br>"
    f"<b>chi squared:</b> {chi_squared:.5f}"
    )

    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Absorbance'))
    fig.add_trace(go.Scatter(x=x_values, y=predicted, mode='lines', name='Best fit', line=dict(color='black')))
    fig.add_trace(go.Scatter(
        x=x_values[peaks],
        y=y_values[peaks],
        mode='markers',
        name='Peaks',
        marker=dict(size=8, color='red'),
        visible='legendonly'
    ))

    # fig.add_annotation(
    #     x=1100,
    #     y=0.9,
    #     xref="x",
    #     yref="y",
    #     text=fit_results_text,
    #     showarrow=False,
    #     align="left",
    #     font=dict(
    #         family="Arial",
    #         size=14,
    #         color="black"
    #     ),
    #     bgcolor="white",
    #     bordercolor="black",
    #     borderwidth=1.2,
    #     borderpad=4,
    #     opacity=0.8
    # )
    
    fig.update_layout(title='Pseudo Voigt fit', 
                      xaxis_title='Wavenumber', 
                      yaxis_title='Absorbance', 
                      height=650, 
                      width=950
                     )


    fwhms = []
    areas = []

    y_combined = combined_pseudo_voigt(x_values, *params)
    total_area = np.trapz(y_combined, x=x_values)
    
    for i in range(0, len(params), 5):
        y_pred = combined_pseudo_voigt(x_values, *params[i:i+5])
        fwhms.append(calculate_FWHM_pseudo_voigt(x_values, y_pred))
        areas.append(np.trapz(y_pred / total_area, x=x_values))
        
        fig.add_trace(go.Scatter(x=x_values, 
                                 y=combined_pseudo_voigt(x_values, *params[i:i+5]), 
                                 mode='lines', 
                                 line=dict(width=1, dash='dash'),
                                name=str(round(params[i], 2))))

    

    if fig_show:
        fig.show()

    residual = y_values - predicted
    
    fig_residuals = go.Figure()
    fig_residuals.add_trace(go.Scatter(x=x_values, y=residual, mode='lines', name='Residual', line=dict(color='green')))
    fig_residuals.update_layout(title='Pseudo Voigt fit Residuals', 
              xaxis_title='Wavenumber', 
              yaxis_title='Residual', 
              height=650, 
              width=950
             )

    if residuals:
        fig_residuals.show()

    params_df = pd.DataFrame({
        "peak_found": x_values[peaks],
        "wavenumber": params[::5],
        "gamma_gauss": params[1::5],
        "gamma_lorentz": params[2::5],
        "amplitude": params[3::5],
        "eta": params[4::5],
        "FWHM": fwhms,
        "area": areas,
    })

    fit_result = {
        "fit_figure": fig,
        "residuals_figure": fig_residuals,
        "SS": sum_of_squares,
        "R2": r2,
        "adj R2": adjusted_r2,
        "chi squared": chi_squared
    }
    
    return params_df, fit_result


def calculate_FWHM_pseudo_voigt(x, y):
    max_index = np.argmax(y)
    max_y = y[max_index]

    half_max = max_y / 2

    left_index = np.argmin(np.abs(y[:max_index] - half_max))
    right_index = max_index + np.argmin(np.abs(y[max_index:] - half_max))

    x_at_half_max_left = np.interp(half_max, y[left_index:left_index+2], x[left_index:left_index+2])
    x_at_half_max_right = np.interp(half_max, y[right_index:right_index+2], x[right_index:right_index+2])

    FWHM = x_at_half_max_right - x_at_half_max_left

    return FWHM