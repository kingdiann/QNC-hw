"""
GAMLSS (Generalized Additive Models for Location, Scale and Shape) Implementation

This module provides a Python implementation of GAMLSS models for growth chart analysis
and developmental trajectory modeling, based on:
- Rigby & Stasinopoulos (2005) - GAMLSS framework
- Bethlehem et al. (2022) - Brain charts for the human lifespan

Key Features:
- Multiple distributions (Normal, BCCG, BCT)
- P-spline smoothing for age effects
- Random effects for study/site variability
- Automated model selection via AIC/BIC
- Comprehensive diagnostics (worm plots, Q-Q plots)
- Centile curve prediction

Author: Adapted for Python from R GAMLSS package
Date: 2025
"""

# %%
import numpy as np
import pandas as pd
from scipy import stats, optimize, interpolate
from scipy.interpolate import interp1d
from scipy.special import gamma as gamma_func
from scipy.special import gammaln, digamma
import warnings
from typing import Optional, Union, List, Dict, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from dataclasses import dataclass


# %%
# ============================================================================
# Distribution Classes
# ============================================================================

class Distribution:
    """Base class for GAMLSS distributions"""
    
    def __init__(self):
        self.n_params = 0
        self.param_names = []
        
    def pdf(self, y, **params):
        """Probability density function"""
        raise NotImplementedError
        
    def cdf(self, y, **params):
        """Cumulative distribution function"""
        raise NotImplementedError
        
    def quantile(self, p, **params):
        """Quantile function (inverse CDF)"""
        raise NotImplementedError
        
    def loglik(self, y, **params):
        """Log-likelihood"""
        return np.sum(np.log(self.pdf(y, **params) + 1e-10))
    
    def initialize_params(self, y):
        """Initialize parameters from data"""
        raise NotImplementedError


class NormalDistribution(Distribution):
    """
    Normal (Gaussian) distribution
    Parameters: mu (mean), sigma (standard deviation)
    """
    
    def __init__(self):
        super().__init__()
        self.n_params = 2
        self.param_names = ['mu', 'sigma']
        self.name = 'NO'
        
    def pdf(self, y, mu, sigma):
        """PDF of normal distribution"""
        return stats.norm.pdf(y, loc=mu, scale=sigma)
    
    def cdf(self, y, mu, sigma):
        """CDF of normal distribution"""
        return stats.norm.cdf(y, loc=mu, scale=sigma)
    
    def quantile(self, p, mu, sigma):
        """Quantile function"""
        return stats.norm.ppf(p, loc=mu, scale=sigma)
    
    def loglik(self, y, mu, sigma):
        """Log-likelihood"""
        return np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    
    def initialize_params(self, y):
        """Initialize from data"""
        mu = np.mean(y)
        sigma = np.std(y)
        return {'mu': mu, 'sigma': sigma}


class BCCGDistribution(Distribution):
    """
    Box-Cox Cole and Green distribution
    Parameters: mu (median), sigma (coefficient of variation), nu (skewness/Box-Cox power)
    
    If nu != 0: Y^nu = mu * exp(sigma * Z) where Z ~ N(0,1)
    If nu == 0: log(Y) = log(mu) + sigma * Z
    """
    
    def __init__(self):
        super().__init__()
        self.n_params = 3
        self.param_names = ['mu', 'sigma', 'nu']
        self.name = 'BCCG'
        
    def _box_cox_transform(self, y, nu):
        """Box-Cox transformation"""
        nu = np.asarray(nu)
        y = np.asarray(y)
        
        # Handle scalar and array nu
        if np.isscalar(nu) or nu.size == 1:
            nu_val = float(nu)
            if np.abs(nu_val) < 1e-10:
                return np.log(y)
            else:
                return (y**nu_val - 1) / nu_val
        else:
            # Array case
            result = np.zeros_like(y)
            near_zero = np.abs(nu) < 1e-10
            result[near_zero] = np.log(y[near_zero])
            result[~near_zero] = (y[~near_zero]**nu[~near_zero] - 1) / nu[~near_zero]
            return result
    
    def _inverse_box_cox(self, z, nu):
        """Inverse Box-Cox transformation"""
        nu = np.asarray(nu)
        z = np.asarray(z)
        
        if np.isscalar(nu) or nu.size == 1:
            nu_val = float(nu)
            if np.abs(nu_val) < 1e-10:
                return np.exp(z)
            else:
                # Ensure the base is positive before power operation
                base = nu_val * z + 1
                base = np.maximum(base, 1e-10)  # Prevent negative or zero values
                return base**(1/nu_val)
        else:
            result = np.zeros_like(z)
            near_zero = np.abs(nu) < 1e-10
            result[near_zero] = np.exp(z[near_zero])
            # Ensure the base is positive before power operation
            base = nu[~near_zero] * z[~near_zero] + 1
            base = np.maximum(base, 1e-10)  # Prevent negative or zero values
            result[~near_zero] = base**(1/nu[~near_zero])
            return result
    
    def pdf(self, y, mu, sigma, nu):
        """PDF of BCCG distribution"""
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        nu = np.asarray(nu)
        y = np.asarray(y)
        
        # Transform to standard normal
        z = (self._box_cox_transform(y, nu) - self._box_cox_transform(mu, nu)) / sigma
        
        # Jacobian
        jacobian = y**(nu - 1) if not np.isscalar(nu) or np.abs(nu) >= 1e-10 else 1/y
        
        # PDF
        pdf_val = stats.norm.pdf(z) * jacobian / sigma
        return pdf_val
    
    def cdf(self, y, mu, sigma, nu):
        """CDF of BCCG distribution"""
        z = (self._box_cox_transform(y, nu) - self._box_cox_transform(mu, nu)) / sigma
        return stats.norm.cdf(z)
    
    def quantile(self, p, mu, sigma, nu):
        """Quantile function"""
        z = stats.norm.ppf(p)
        y_transform = self._box_cox_transform(mu, nu) + sigma * z
        return self._inverse_box_cox(y_transform, nu)
    
    def loglik(self, y, mu, sigma, nu):
        """Log-likelihood"""
        try:
            pdf_val = self.pdf(y, mu, sigma, nu)
            return np.sum(np.log(pdf_val + 1e-10))
        except:
            return -np.inf
    
    def initialize_params(self, y):
        """Initialize from data"""
        mu = np.median(y)
        sigma = np.std(np.log(y + 1))
        nu = 0.5  # Start with moderate Box-Cox power
        return {'mu': mu, 'sigma': sigma, 'nu': nu}


class BCTDistribution(Distribution):
    """
    Box-Cox t distribution
    Parameters: mu, sigma, nu (Box-Cox power), tau (degrees of freedom for t-distribution)
    
    More flexible than BCCG - can handle heavy tails via tau parameter
    """
    
    def __init__(self):
        super().__init__()
        self.n_params = 4
        self.param_names = ['mu', 'sigma', 'nu', 'tau']
        self.name = 'BCT'
        
    def _box_cox_transform(self, y, nu):
        """Box-Cox transformation"""
        nu = np.asarray(nu)
        y = np.asarray(y)
        
        if np.isscalar(nu) or nu.size == 1:
            nu_val = float(nu)
            if np.abs(nu_val) < 1e-10:
                return np.log(y)
            else:
                return (y**nu_val - 1) / nu_val
        else:
            result = np.zeros_like(y)
            near_zero = np.abs(nu) < 1e-10
            result[near_zero] = np.log(y[near_zero])
            result[~near_zero] = (y[~near_zero]**nu[~near_zero] - 1) / nu[~near_zero]
            return result
    
    def _inverse_box_cox(self, z, nu):
        """Inverse Box-Cox transformation"""
        nu = np.asarray(nu)
        z = np.asarray(z)
        
        if np.isscalar(nu) or nu.size == 1:
            nu_val = float(nu)
            if np.abs(nu_val) < 1e-10:
                return np.exp(z)
            else:
                # Ensure the base is positive before power operation
                base = nu_val * z + 1
                base = np.maximum(base, 1e-10)  # Prevent negative or zero values
                return base**(1/nu_val)
        else:
            result = np.zeros_like(z)
            near_zero = np.abs(nu) < 1e-10
            result[near_zero] = np.exp(z[near_zero])
            # Ensure the base is positive before power operation
            base = nu[~near_zero] * z[~near_zero] + 1
            base = np.maximum(base, 1e-10)  # Prevent negative or zero values
            result[~near_zero] = base**(1/nu[~near_zero])
            return result
    
    def pdf(self, y, mu, sigma, nu, tau):
        """PDF of BCT distribution"""
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        nu = np.asarray(nu)
        tau = np.asarray(tau)
        y = np.asarray(y)
        
        # Transform to t-distribution
        z = (self._box_cox_transform(y, nu) - self._box_cox_transform(mu, nu)) / sigma
        
        # Jacobian
        jacobian = y**(nu - 1) if not np.isscalar(nu) or np.abs(nu) >= 1e-10 else 1/y
        
        # PDF using t-distribution
        pdf_val = stats.t.pdf(z, df=tau) * jacobian / sigma
        return pdf_val
    
    def cdf(self, y, mu, sigma, nu, tau):
        """CDF of BCT distribution"""
        z = (self._box_cox_transform(y, nu) - self._box_cox_transform(mu, nu)) / sigma
        return stats.t.cdf(z, df=tau)
    
    def quantile(self, p, mu, sigma, nu, tau):
        """Quantile function"""
        z = stats.t.ppf(p, df=tau)
        y_transform = self._box_cox_transform(mu, nu) + sigma * z
        return self._inverse_box_cox(y_transform, nu)
    
    def loglik(self, y, mu, sigma, nu, tau):
        """Log-likelihood"""
        try:
            pdf_val = self.pdf(y, mu, sigma, nu, tau)
            return np.sum(np.log(pdf_val + 1e-10))
        except:
            return -np.inf
    
    def initialize_params(self, y):
        """Initialize from data"""
        mu = np.median(y)
        sigma = np.std(np.log(y + 1))
        nu = 0.5
        tau = 10.0  # Start with moderate degrees of freedom
        return {'mu': mu, 'sigma': sigma, 'nu': nu, 'tau': tau}

# %%
# ============================================================================
# Smoothing Functions
# ============================================================================

def create_pspline_basis(x, df=5, degree=3):
    """
    Create P-spline basis functions
    
    Parameters:
    -----------
    x : array-like
        Predictor variable
    df : int
        Degrees of freedom (number of basis functions)
    degree : int
        Degree of B-splines (typically 3 for cubic)
    periodic : bool
        Whether to use periodic splines
        
    Returns:
    --------
    B : array
        Basis matrix (n x df)
    knots : array
        Knot positions
    """
    x = np.asarray(x)
    n = len(x)
    
    # Determine knot positions
    x_min, x_max = np.min(x), np.max(x)
    
    # Ensure df is valid for the given degree
    min_df = degree + 1
    if df < min_df:
        df = min_df
        warnings.warn(f"df too small for degree {degree}, using df={df}")
    
    # Number of interior knots
    n_interior_knots = df - degree - 1
    
    # Create knots
    if n_interior_knots > 0:
        interior_knots = np.linspace(x_min, x_max, n_interior_knots + 2)[1:-1]
    else:
        interior_knots = np.array([])
    
    # Add boundary knots (repeated for B-spline basis)
    knots = np.concatenate([
        [x_min] * (degree + 1),
        interior_knots,
        [x_max] * (degree + 1)
    ])
    
    # Calculate correct number of coefficients
    n_coef = len(knots) - degree - 1
    
    # Create B-spline basis using scipy
    B = np.zeros((n, n_coef))
    for i in range(n_coef):
        # Create a B-spline for each basis function
        coef = np.zeros(n_coef)
        coef[i] = 1.0
        spl = interpolate.BSpline(knots, coef, degree, extrapolate=False)
        B[:, i] = spl(x)
    
    # Handle any NaN values from extrapolation
    B = np.nan_to_num(B, nan=0.0)
    
    return B, knots


def pspline_smooth(x, y, df=5, degree=3, weights=None):
    """
    Fit P-spline smoother to data
    
    Parameters:
    -----------
    x : array-like
        Predictor variable
    y : array-like
        Response variable
    df : int
        Degrees of freedom
    degree : int
        Polynomial degree of B-splines (typically 3 for cubic)
    weights : array-like or None
        Observation weights
        
    Returns:
    --------
    fitted : array
        Fitted values
    coef : array
        Spline coefficients
    basis : array
        Basis matrix
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Create basis
    B, knots = create_pspline_basis(x, df=df, degree=degree)
    
    # Fit using weighted least squares
    if weights is None:
        weights = np.ones(len(y))
    else:
        weights = np.asarray(weights)
    
    # Weighted design matrix
    W = np.diag(np.sqrt(weights))
    B_w = W @ B
    y_w = W @ y
    
    # Solve with small ridge penalty for stability
    ridge_penalty = 1e-6
    actual_df = B.shape[1]  # Use actual number of basis functions
    coef = np.linalg.solve(B_w.T @ B_w + ridge_penalty * np.eye(actual_df), B_w.T @ y_w)
    
    # Fitted values
    fitted = B @ coef
    
    return fitted, coef, B

# %%
# ============================================================================
# Main GAMLSS Fitter Class
# ============================================================================

@dataclass
class FormulaSpec:
    """Specification for a parameter formula"""
    fixed_df: int = 5
    degree: int = 3
    random: bool = False
    
    
class GAMLSSFitter:
    """
    Main GAMLSS fitter class
    
    Fits Generalized Additive Models for Location, Scale and Shape
    """
    
    def __init__(self, 
                 distribution: Union[str, Distribution] = 'BCCG',
                 age_col: str = 'age',
                 y_col: str = 'y',
                 group_col: Optional[str] = None,
                 mu_formula: Optional[Dict] = None,
                 sigma_formula: Optional[Dict] = None,
                 nu_formula: Optional[Dict] = None,
                 tau_formula: Optional[Dict] = None,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 verbose: bool = False):
        """
        Parameters:
        -----------
        distribution : str or Distribution
            'NO' (Normal), 'BCCG', or 'BCT'
        age_col : str
            Column name for age/time variable
        y_col : str
            Column name for outcome variable
        group_col : str or None
            Column name for grouping (random effects)
        mu_formula : dict or None
            Formula for mu: {'fixed_df': 5, 'degree': 3, 'random': True}
        sigma_formula : dict or None
            Formula for sigma: {'fixed_df': 3, 'degree': 3, 'random': False}
        nu_formula : dict or None
            Formula for nu (BCCG/BCT only): {'fixed_df': 1, 'degree': 3, 'random': False}
        tau_formula : dict or None
            Formula for tau (BCT only): {'fixed_df': 1, 'degree': 3, 'random': False}
        max_iter : int
            Maximum iterations for fitting
        tol : float
            Convergence tolerance
        verbose : bool
            Print fitting progress
        """
        # Set up distribution
        if isinstance(distribution, str):
            if distribution == 'NO':
                self.distribution = NormalDistribution()
            elif distribution == 'BCCG':
                self.distribution = BCCGDistribution()
            elif distribution == 'BCT':
                self.distribution = BCTDistribution()
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
        else:
            self.distribution = distribution
        
        self.age_col = age_col
        self.y_col = y_col
        self.group_col = group_col
        
        # Set up formulas with defaults
        default_formula = {'fixed_df': 5, 'degree': 3, 'random': False}
        self.mu_formula = mu_formula or default_formula.copy()
        self.sigma_formula = sigma_formula or {'fixed_df': 3, 'degree': 3, 'random': False}
        self.nu_formula = nu_formula or {'fixed_df': 1, 'degree': 3, 'random': False}  # Often constant
        self.tau_formula = tau_formula or {'fixed_df': 1, 'degree': 3, 'random': False}
        
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # Storage for fitted values
        self.fitted_params_ = {}
        self.coef_ = {}
        self.basis_ = {}
        self.converged_ = False
        self.aic_ = np.inf
        self.bic_ = np.inf
        self.loglik_ = -np.inf
        self.n_params_ = 0
        
    def _prepare_data(self, data):
        """Prepare data for fitting"""
        self.data = data.copy()
        self.age = data[self.age_col].values
        self.y = data[self.y_col].values
        self.n = len(self.y)
        
        # Handle groups
        if self.group_col and self.group_col in data.columns:
            self.groups = data[self.group_col].values
            self.unique_groups = np.unique(self.groups)
            self.n_groups = len(self.unique_groups)
            # Create group indices
            self.group_indices = {g: np.where(self.groups == g)[0] 
                                 for g in self.unique_groups}
        else:
            self.groups = None
            self.unique_groups = None
            self.n_groups = 0
            self.group_indices = {}
    
    def _fit_parameter(self, param_name, y_working, weights, formula_spec):
        """
        Fit a single parameter using P-splines
        
        Parameters:
        -----------
        param_name : str
            Parameter name ('mu', 'sigma', etc.)
        y_working : array
            Working response
        weights : array
            Working weights
        formula_spec : dict
            Formula specification
            
        Returns:
        --------
        fitted : array
            Fitted parameter values
        """
        df = formula_spec.get('fixed_df', 5)
        degree = formula_spec.get('degree', 3)
        has_random = formula_spec.get('random', False)
        
        if not has_random or self.groups is None:
            # Fixed effects only
            fitted, coef, basis = pspline_smooth(self.age, y_working, df=df, degree=degree, weights=weights)
            actual_df = basis.shape[1]  # Store actual df (might be adjusted)
            self.coef_[param_name] = {'fixed': coef, 'df': actual_df}
            self.basis_[param_name] = basis
            return fitted
        else:
            # Fixed + random effects
            # First fit fixed effects
            fitted_fixed, coef_fixed, basis = pspline_smooth(
                self.age, y_working, df=df, degree=degree, weights=weights
            )
            actual_df = basis.shape[1]  # Store actual df (might be adjusted)
            
            # Compute residuals
            resid = y_working - fitted_fixed
            
            # Estimate random effects per group (simple group means)
            random_effects = np.zeros(self.n)
            group_re = {}
            for g in self.unique_groups:
                idx = self.group_indices[g]
                group_mean = np.average(resid[idx], weights=weights[idx])
                random_effects[idx] = group_mean
                group_re[g] = group_mean
            
            fitted = fitted_fixed + random_effects
            
            self.coef_[param_name] = {
                'fixed': coef_fixed,
                'random': group_re,
                'df': actual_df
            }
            self.basis_[param_name] = basis
            
            return fitted
    
    def fit(self, data):
        """
        Fit GAMLSS model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with columns specified in __init__
            
        Returns:
        --------
        self : GAMLSSFitter
            Fitted model
        """
        self._prepare_data(data)
        
        # Initialize parameters
        init_params = self.distribution.initialize_params(self.y)
        
        # Initialize fitted parameters (start with constants)
        for param_name in self.distribution.param_names:
            self.fitted_params_[param_name] = np.full(self.n, init_params[param_name])
        
        # Iterative fitting using backfitting algorithm
        prev_loglik = -np.inf
        
        for iteration in range(self.max_iter):
            # Update each parameter in turn
            for param_name in self.distribution.param_names:
                # Get current parameter values
                current_params = {p: self.fitted_params_[p] for p in self.distribution.param_names}
                
                # Compute working response and weights (simplified)
                if param_name == 'mu':
                    y_working = self.y
                    weights = np.ones(self.n)
                    formula = self.mu_formula
                elif param_name == 'sigma':
                    # Work on log scale for sigma
                    residuals = self.y - self.fitted_params_['mu']
                    y_working = np.log(np.abs(residuals) + 0.1)
                    weights = np.ones(self.n)
                    formula = self.sigma_formula
                elif param_name == 'nu':
                    # Keep nu relatively constant or slowly varying
                    y_working = np.full(self.n, init_params['nu'])
                    weights = np.ones(self.n)
                    formula = self.nu_formula
                elif param_name == 'tau':
                    # Keep tau relatively constant
                    y_working = np.full(self.n, init_params['tau'])
                    weights = np.ones(self.n)
                    formula = self.tau_formula
                else:
                    continue
                
                # Fit this parameter
                fitted_param = self._fit_parameter(param_name, y_working, weights, formula)
                
                # Apply constraints
                if param_name == 'sigma':
                    fitted_param = np.exp(fitted_param)  # Sigma must be positive
                    fitted_param = np.maximum(fitted_param, 0.01)  # Lower bound
                elif param_name == 'nu':
                    fitted_param = np.clip(fitted_param, -2, 2)  # Reasonable range
                elif param_name == 'tau':
                    fitted_param = np.maximum(fitted_param, 2.1)  # tau > 2
                
                self.fitted_params_[param_name] = fitted_param
            
            # Check convergence
            loglik = self.distribution.loglik(self.y, **self.fitted_params_)
            
            if self.verbose:
                print(f"Iteration {iteration + 1}: Log-likelihood = {loglik:.4f}")
            
            if np.isfinite(loglik):
                if abs(loglik - prev_loglik) < self.tol:
                    self.converged_ = True
                    if self.verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    break
                prev_loglik = loglik
            else:
                warnings.warn("Non-finite log-likelihood encountered")
                break
        
        if not self.converged_:
            warnings.warn(f"Did not converge after {self.max_iter} iterations")
        
        # Compute model statistics
        self.loglik_ = prev_loglik
        self._compute_model_stats()
        
        return self
    
    def _compute_model_stats(self):
        """Compute AIC, BIC, etc."""
        # Count effective parameters
        n_params = 0
        for param_name in self.distribution.param_names:
            if param_name in self.coef_:
                n_params += len(self.coef_[param_name]['fixed'])
                if 'random' in self.coef_[param_name]:
                    n_params += len(self.coef_[param_name]['random'])
        
        self.n_params_ = n_params
        self.aic_ = -2 * self.loglik_ + 2 * n_params
        self.bic_ = -2 * self.loglik_ + np.log(self.n) * n_params
    
    def predict_centiles(self, age_values, centiles=[5, 25, 50, 75, 95], 
                        group=None):
        """
        Predict centile curves at specified age values
        
        Parameters:
        -----------
        age_values : array-like
            Age values at which to predict
        centiles : list
            Percentiles to compute
        group : str or None
            Specific group for group-specific predictions
            
        Returns:
        --------
        predictions : dict
            Dictionary mapping centile to predicted values
        """
        age_values = np.asarray(age_values)
        n_pred = len(age_values)
        
        # Predict each parameter at the age values
        pred_params = {}
        for param_name in self.distribution.param_names:
            # Get basis and coefficients
            if param_name not in self.basis_ or param_name not in self.coef_:
                # Use mean value if not fitted
                pred_params[param_name] = np.full(n_pred, 
                                                   np.mean(self.fitted_params_[param_name]))
                continue
            
            coef = self.coef_[param_name]
            
            # Use interpolation for prediction (more robust than basis reconstruction)
            # Compute fitted values on training data
            fitted_values = self.basis_[param_name] @ coef['fixed']
            
            # Create smooth interpolator
            # Use cubic for smooth curves, but clip to data range for stability
            kind = 'cubic' if len(self.age) > 10 else 'linear'
            
            try:
                interp_func = interp1d(
                    self.age, fitted_values, 
                    kind=kind, 
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                pred_fixed = interp_func(age_values)
            except:
                # Fallback to linear if cubic fails
                interp_func = interp1d(
                    self.age, fitted_values, 
                    kind='linear', 
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                pred_fixed = interp_func(age_values)
            
            # Add random effects if requested and available
            if group and 'random' in coef and group in coef['random']:
                pred = pred_fixed + coef['random'][group]
            else:
                pred = pred_fixed
            
            # Apply transformations
            if param_name == 'sigma':
                pred = np.exp(pred)
                pred = np.maximum(pred, 0.01)
            elif param_name == 'nu':
                pred = np.clip(pred, -2, 2)
            elif param_name == 'tau':
                pred = np.maximum(pred, 2.1)
            
            pred_params[param_name] = pred
        
        # Compute centiles
        centile_curves = {}
        for c in centiles:
            p = c / 100.0
            centile_curves[c] = self.distribution.quantile(p, **pred_params)
        
        return centile_curves
    
    def compute_z_scores(self, data=None):
        """
        Compute normalized z-scores for observations
        
        Parameters:
        -----------
        data : pd.DataFrame or None
            Data to compute z-scores for (uses training data if None)
            
        Returns:
        --------
        z_scores : array
            Normalized quantile residuals
        """
        if data is None:
            y = self.y
            params = self.fitted_params_
        else:
            # Would need to predict parameters for new data
            raise NotImplementedError("Z-scores for new data not yet implemented")
        
        # Compute CDF of observations
        cdf_vals = self.distribution.cdf(y, **params)
        
        # Convert to standard normal quantiles
        z_scores = stats.norm.ppf(np.clip(cdf_vals, 1e-6, 1-1e-6))
        
        return z_scores

# %%
# ============================================================================
# Model Selection
# ============================================================================

class GAMLSSModelSelector:
    """
    Comprehensive model selection for GAMLSS
    """
    
    def __init__(self, age_col='age', y_col='y', group_col=None):
        """
        Parameters:
        -----------
        age_col : str
            Age variable column name
        y_col : str
            Outcome variable column name
        group_col : str or None
            Grouping variable for random effects
        """
        self.age_col = age_col
        self.y_col = y_col
        self.group_col = group_col
        self.fitted_models = {}
        self.comparison_table = None
        
    def _build_model(self, distribution, df, random_effects='none'):
        """Build a model configuration"""
        mu_formula = {'fixed_df': df, 'degree': 3, 'random': 'mu' in random_effects}
        sigma_formula = {'fixed_df': max(3, df-2), 'degree': 3, 'random': 'sigma' in random_effects}
        
        fitter = GAMLSSFitter(
            distribution=distribution,
            age_col=self.age_col,
            y_col=self.y_col,
            group_col=self.group_col if random_effects != 'none' else None,
            mu_formula=mu_formula,
            sigma_formula=sigma_formula,
            verbose=False
        )
        return fitter
    
    def fit_candidate_models(self, data, 
                            distributions=['NO', 'BCCG', 'BCT'],
                            df_range=[3, 5, 7, 9],
                            random_effects_options=['none', 'mu', 'mu+sigma'],
                            n_jobs=1):
        """
        Fit all candidate models
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        distributions : list
            Distributions to try
        df_range : list
            Degrees of freedom for splines to try
        random_effects_options : list
            Which parameters get random effects
        n_jobs : int
            Number of parallel jobs (1 = sequential, -1 = all cores)
            
        Returns:
        --------
        comparison : pd.DataFrame
            Table of all models with fit statistics
        """
        results = []
        model_id = 0
        
        # Build list of all model configurations
        configs = []
        for dist in distributions:
            for df in df_range:
                for re_option in random_effects_options:
                    configs.append((model_id, dist, df, re_option))
                    model_id += 1
        
        # Fit models (sequential or parallel)
        if n_jobs == 1:
            # Sequential fitting
            for model_id, dist, df, re_option in configs:
                result = self._fit_single_model(data, model_id, dist, df, re_option)
                if result:
                    results.append(result)
        else:
            # Parallel fitting
            results_parallel = Parallel(n_jobs=n_jobs)(
                delayed(self._fit_single_model)(data, mid, d, df, re)
                for mid, d, df, re in configs
            )
            results = [r for r in results_parallel if r is not None]
        
        # Create comparison table
        self.comparison_table = pd.DataFrame(results).sort_values('BIC').reset_index(drop=True)
        return self.comparison_table
    
    def _fit_single_model(self, data, model_id, dist, df, re_option):
        """Fit a single model configuration"""
        try:
            fitter = self._build_model(dist, df, re_option)
            fitter.fit(data)
            
            # Store fitted model
            self.fitted_models[model_id] = fitter
            
            return {
                'model_id': model_id,
                'distribution': dist,
                'df': df,
                'random_effects': re_option,
                'AIC': fitter.aic_,
                'BIC': fitter.bic_,
                'log_likelihood': fitter.loglik_,
                'n_params': fitter.n_params_,
                'converged': fitter.converged_
            }
        except Exception as e:
            print(f"Failed to fit model {model_id} ({dist}, df={df}, re={re_option}): {e}")
            return None
    
    def get_best_model(self, criterion='BIC'):
        """
        Get the best model by specified criterion
        
        Parameters:
        -----------
        criterion : str
            'AIC' or 'BIC'
            
        Returns:
        --------
        best_fitter : GAMLSSFitter
            Best fitted model
        """
        if self.comparison_table is None:
            raise ValueError("Must run fit_candidate_models first")
        
        best_row = self.comparison_table.sort_values(criterion).iloc[0]
        best_id = int(best_row['model_id'])
        return self.fitted_models[best_id]
    
    def compare_models(self, top_n=10):
        """
        Display top N models
        
        Parameters:
        -----------
        top_n : int
            Number of top models to show
            
        Returns:
        --------
        top_models : pd.DataFrame
            Top N models ranked by BIC
        """
        if self.comparison_table is None:
            raise ValueError("Must run fit_candidate_models first")
        
        return self.comparison_table.head(top_n)

# %%
# ============================================================================
# Diagnostics
# ============================================================================

class GAMLSSDiagnostics:
    """
    Diagnostic plots and checks for GAMLSS models
    """
    
    def __init__(self, fitter):
        """
        Parameters:
        -----------
        fitter : GAMLSSFitter
            Fitted GAMLSS model
        """
        self.fitter = fitter
        self.z_scores = fitter.compute_z_scores()
        
    def plot_worm(self, n_groups=9, ax=None):
        """
        Worm plot (detrended Q-Q plot by age groups)
        
        Parameters:
        -----------
        n_groups : int
            Number of age groups to create
        ax : matplotlib axis
            Axis to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create age groups
        age_groups = pd.qcut(self.fitter.age, q=n_groups, duplicates='drop')
        
        # Plot worm plot for each age group
        for i, (name, group_idx) in enumerate(pd.Series(age_groups).groupby(age_groups).groups.items()):
            z_group = self.z_scores[group_idx]
            z_group = z_group[np.isfinite(z_group)]
            
            if len(z_group) < 3:
                continue
            
            # Compute theoretical and sample quantiles
            z_group_sorted = np.sort(z_group)
            n = len(z_group_sorted)
            theoretical_q = stats.norm.ppf(np.arange(1, n+1) / (n+1))
            
            # Detrend (subtract theoretical quantiles)
            detrended = z_group_sorted - theoretical_q
            
            # Plot
            ax.plot(theoretical_q, detrended, 'o-', alpha=0.6, markersize=3, 
                   label=f'Group {i+1}')
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Unit Normal Quantiles')
        ax.set_ylabel('Deviations')
        ax.set_title('Worm Plot')
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
        sns.despine()
        
        return ax
    
    def plot_qq(self, ax=None):
        """
        Q-Q plot of normalized quantile residuals
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        z_finite = self.z_scores[np.isfinite(self.z_scores)]
        stats.probplot(z_finite, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Normalized Residuals')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_residuals(self, ax=None):
        """
        Residual plots vs age and fitted values
        """
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        else:
            axes = [ax]
        
        # Plot vs age
        if len(axes) >= 2:
            axes[0].scatter(self.fitter.age, self.z_scores, alpha=0.5, s=20)
            axes[0].axhline(y=0, color='red', linestyle='--')
            axes[0].set_xlabel(f'Age ({self.fitter.age_col})')
            axes[0].set_ylabel('Normalized Residuals')
            axes[0].set_title('Residuals vs Age')
            axes[0].grid(True, alpha=0.3)
            sns.despine(ax=axes[0])
        
        # Plot vs fitted
        if len(axes) >= 2:
            fitted_mu = self.fitter.fitted_params_['mu']
            axes[1].scatter(fitted_mu, self.z_scores, alpha=0.5, s=20)
            axes[1].axhline(y=0, color='red', linestyle='--')
            axes[1].set_xlabel('Fitted Values')
            axes[1].set_ylabel('Normalized Residuals')
            axes[1].set_title('Residuals vs Fitted')
            axes[1].grid(True, alpha=0.3)
            sns.despine(ax=axes[1])
        
        return axes
    
    def plot_centile_check(self, centiles=[5, 25, 50, 75, 95], ax=None):
        """
        Overlay actual data quantiles with fitted centiles
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw data
        ax.scatter(self.fitter.age, self.fitter.y, alpha=0.3, s=10, 
                  color='gray', label='Data')
        
        # Plot fitted centiles
        age_pred = np.linspace(self.fitter.age.min(), self.fitter.age.max(), 100)
        centile_curves = self.fitter.predict_centiles(age_pred, centiles=centiles)
        
        colors = plt.cm.berlin(np.linspace(0.1, 0.9, len(centiles)))
        for c, color in zip(centiles, colors):
            ax.plot(age_pred, centile_curves[c], linewidth=2, color=color,
                   label=f'{c}th percentile')
        
        ax.set_xlabel(f'Age ({self.fitter.age_col})')
        ax.set_ylabel(f'{self.fitter.y_col}')
        ax.set_title('Fitted Centile Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine()
        
        return ax
    
    def plot_all_diagnostics(self, figsize=(15, 10)):
        """
        Create comprehensive diagnostic plot panel
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_centile_check(ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_qq(ax=ax2)
        
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_worm(ax=ax3)
        
        axes_resid = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
        self.plot_residuals(ax=axes_resid[0])
        
        # Histogram of z-scores
        axes_resid[1].hist(self.z_scores[np.isfinite(self.z_scores)], 
                          bins=30, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(-3, 3, 100)
        axes_resid[1].plot(x, stats.norm.pdf(x), 'r-', linewidth=2, 
                          label='Standard Normal')
        axes_resid[1].set_xlabel('Normalized Residuals')
        axes_resid[1].set_ylabel('Density')
        axes_resid[1].set_title('Residual Distribution')
        axes_resid[1].legend()
        axes_resid[1].grid(True, alpha=0.3)
        sns.despine(ax=axes_resid[1])
        
        plt.suptitle(f'GAMLSS Diagnostics: {self.fitter.distribution.name}', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def diagnostic_summary(self):
        """
        Print comprehensive diagnostic summary
        """
        z_finite = self.z_scores[np.isfinite(self.z_scores)]
        
        # Shapiro-Wilk test
        if len(z_finite) > 3:
            shapiro_stat, shapiro_p = stats.shapiro(z_finite[:5000])  # Limit for large n
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Summary statistics
        summary = {
            'mean_residuals': np.mean(z_finite),
            'std_residuals': np.std(z_finite),
            'skewness': stats.skew(z_finite),
            'kurtosis': stats.kurtosis(z_finite),
            'shapiro_statistic': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'n_observations': len(z_finite)
        }
        
        print("\n" + "="*60)
        print("GAMLSS Diagnostic Summary")
        print("="*60)
        print(f"Model: {self.fitter.distribution.name}")
        print(f"N observations: {summary['n_observations']}")
        print(f"\nLog-likelihood: {self.fitter.loglik_:.2f}")
        print(f"AIC: {self.fitter.aic_:.2f}")
        print(f"BIC: {self.fitter.bic_:.2f}")
        print(f"\nNormalized Residual Statistics:")
        print(f"  Mean: {summary['mean_residuals']:.4f} (should be ~0)")
        print(f"  Std Dev: {summary['std_residuals']:.4f} (should be ~1)")
        print(f"  Skewness: {summary['skewness']:.4f} (should be ~0)")
        print(f"  Kurtosis: {summary['kurtosis']:.4f} (should be ~0)")
        print(f"\nShapiro-Wilk test for normality:")
        print(f"  Statistic: {summary['shapiro_statistic']:.4f}")
        print(f"  P-value: {summary['shapiro_pvalue']:.4f}")
        if summary['shapiro_pvalue'] > 0.05:
            print(f"  -> Residuals appear normally distributed (p > 0.05)")
        else:
            print(f"  -> Residuals deviate from normality (p < 0.05)")
        print("="*60 + "\n")
        
        return summary

# %%
# ============================================================================
# Convenience Functions
# ============================================================================

def auto_select_gamlss(data, age_col, y_col, group_col=None,
                       quick=False, verbose=True):
    """
    Automatic model selection with sensible defaults
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    age_col : str
        Age variable column name
    y_col : str
        Outcome variable column name
    group_col : str or None
        Grouping variable for random effects
    quick : bool
        If True, only try a few models (faster)
        If False, comprehensive search
    verbose : bool
        Print progress
    
    Returns:
    --------
    best_model : GAMLSSFitter
        Best fitted model
    comparison : pd.DataFrame
        Model comparison table
    diagnostics : GAMLSSDiagnostics
        Diagnostic object for the best model
    
    Example:
    --------
    >>> model, comparison, diag = auto_select_gamlss(
    ...     data=data_roii,
    ...     age_col='Age_in_months_bin',
    ...     y_col='volume',
    ...     group_col='dataset'
    ... )
    >>> print(comparison.head())
    >>> diag.plot_all_diagnostics()
    """
    selector = GAMLSSModelSelector(age_col, y_col, group_col)
    
    if quick:
        distributions = ['NO', 'BCCG']
        df_range = [5, 7]
        re_options = ['none', 'mu'] if group_col else ['none']
    else:
        distributions = ['NO', 'BCCG', 'BCT']
        df_range = [3, 5, 7, 9]
        re_options = ['none', 'mu', 'mu+sigma'] if group_col else ['none']
    
    # Fit all candidate models
    if verbose:
        print("Fitting candidate models...")
        print(f"  Distributions: {distributions}")
        print(f"  Degrees of freedom: {df_range}")
        print(f"  Random effects: {re_options}")
        print(f"  Total models: {len(distributions) * len(df_range) * len(re_options)}")
    
    comparison = selector.fit_candidate_models(
        data, 
        distributions=distributions,
        df_range=df_range,
        random_effects_options=re_options
    )
    
    # Get best model
    best_model = selector.get_best_model(criterion='BIC')
    
    # Run diagnostics
    diagnostics = GAMLSSDiagnostics(best_model)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Best model selected:")
        print(f"  Distribution: {best_model.distribution.name}")
        print(f"  Degrees of freedom: {best_model.mu_formula['fixed_df']}")
        print(f"  Random effects: {'Yes' if best_model.group_col else 'No'}")
        print(f"  BIC: {best_model.bic_:.2f}")
        print(f"  AIC: {best_model.aic_:.2f}")
        print(f"{'='*60}\n")
        print("\nTop 5 models by BIC:")
        print(comparison.head())
    
    return best_model, comparison, diagnostics


def plot_gamlss_comparison(data, fitter, age_col, y_col, 
                           centiles=[5, 50, 95],
                           show_data=True, ax=None, **kwargs):
    """
    Convenient plotting function for GAMLSS results
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to plot
    fitter : GAMLSSFitter
        Fitted GAMLSS model
    age_col : str
        Age column name
    y_col : str
        Outcome column name
    centiles : list
        Which centiles to plot
    show_data : bool
        Whether to show raw data points
    ax : matplotlib axis
        Axis to plot on
    **kwargs : dict
        Additional plotting arguments
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    age = data[age_col].values
    y = data[y_col].values
    
    # Plot data
    if show_data:
        ax.scatter(age, y, alpha=0.3, s=20, color='gray', label='Data')
    
    # Plot centiles
    age_pred = np.linspace(age.min(), age.max(), 100)
    centile_curves = fitter.predict_centiles(age_pred, centiles=centiles)
    
    colors = kwargs.get('colors', plt.cm.berlin(np.linspace(0.1, 0.9, len(centiles))))
    
    for c, color in zip(centiles, colors):
        label = f'{c}th' if c != 50 else 'Median'
        linewidth = 3 if c == 50 else 2
        ax.plot(age_pred, centile_curves[c], linewidth=linewidth, 
               color=color, label=label)
    
    ax.set_xlabel(age_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'GAMLSS Growth Chart ({fitter.distribution.name})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine()
    
    return ax

# %% # CORTICAL SURFACE

# if __name__ == "__main__":
#     # Create DataFrame
#     measure = "volume"

#     # --- Load data ---
#     GMmeasure_df = pd.read_csv(
#         f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/cortical/17schaefer_{measure}_stats.csv"
#     )
#     age_df = pd.read_csv(
#         "/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/subject_ages.csv"
#     )

#     # --- Merge age and GM measure ---
#     data = GMmeasure_df.merge(age_df, on="subject_id", how="inner")
#     print(f"Merged dataset has {len(data)} subjects")
#     print(data.head())
    
#     print("="*60)
#     print("GAMLSS Example - Brain Volume Growth Chart")
#     print("="*60)
    
#     # Automatic model selection
#     best_model, comparison, diagnostics = auto_select_gamlss(
#         data=data,
#         age_col='age',
#         y_col='volume',
#         group_col='dataset',
#         quick=True,
#         verbose=True
#     )
    
#     # Show diagnostics
#     diagnostics.diagnostic_summary()
    
#     # Create plots
#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
#     # Plot growth chart
#     plot_gamlss_comparison(data, best_model, 'age', 'volume', ax=axes[0])
    
#     # Plot diagnostics
#     diagnostics.plot_qq(ax=axes[1])
    
#     plt.tight_layout()
#     # plt.savefig('/tmp/gamlss_example.png', dpi=150, bbox_inches='tight')
#     # print("\nExample plot saved to /tmp/gamlss_example.png")
    
#     # Full diagnostic panel
#     fig_diag = diagnostics.plot_all_diagnostics()
#     # plt.savefig('/tmp/gamlss_diagnostics.png', dpi=150, bbox_inches='tight')
#     # print("Diagnostic plot saved to /tmp/gamlss_diagnostics.png")
#%% # THOMAS NUCLEI VOLUMES
import os
if __name__ == "__main__":
    # -----------------------------
    # Load and prepare data
    # -----------------------------
    data = pd.read_csv(
        "/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_raw.csv"
    )

    # Automatically find the age column
    age_col = next((c for c in data.columns if "age" in c.lower()), None)
    if age_col is None:
        raise ValueError("No column containing 'age' found in your CSV!")
    print(f"Using age column: {age_col}")

    # -----------------------------
    # Identify nucleus labels
    # -----------------------------
    volume_cols = [c for c in data.columns if c.startswith(("L_", "R_"))]
    labels = sorted(set(int(c.split("_")[1]) for c in volume_cols))

    # Compute bilateral volumes
    for label in labels:
        left = f"L_{label}"
        right = f"R_{label}"
        if left in data.columns and right in data.columns:
            data[f"whole_{label}"] = data[left] + data[right]
            
            # Print the first few combined values for verification
            print(f"\n--- {label} ---")
            print(f"Left ({left}): {data[left].head().tolist()}")
            print(f"Right ({right}): {data[right].head().tolist()}")
            print(f"Sum (whole_{label}): {data[f'whole_{label}'].head().tolist()}")  
    print(f"Computed bilateral volumes for {len(labels)} nuclei")
    # Add after computing bilateral volumes:
    print("\n=== DATA VERIFICATION ===")
    for label in labels[:3]:  # Check first 3
        y_col = f"Volume (mm3; {label})"
        if y_col in data.columns:
            print(f"\n{y_col}:")
            print(f"  Range: [{data[y_col].min():.2f}, {data[y_col].max():.2f}]")
            print(f"  NaN count: {data[y_col].isna().sum()}")
            print(f"  Zero count: {(data[y_col] == 0).sum()}")
            print(f"  Negative count: {(data[y_col] < 0).sum()}")
    # -----------------------------
    # Run GAMLSS per nucleus
    # -----------------------------
    output_dir = (
        "/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/figures"
    )
    os.makedirs(output_dir, exist_ok=True)

    for label in labels:
        y_col = f"Volume (mm3; {label})"
        if y_col not in data.columns:
            continue

        print("=" * 60)
        print(f"GAMLSS: Thalamus nucleus {label}")
        print("=" * 60)

        try:
            # Check data validity first
            valid_data = data[[age_col, y_col]].dropna()
            if len(valid_data) < 50:
                print(f"WARNING: Only {len(valid_data)} valid observations. Skipping.")
                continue
                
            print(f"Valid observations: {len(valid_data)}")
            print(f"Age range: [{valid_data[age_col].min():.1f}, {valid_data[age_col].max():.1f}]")
            print(f"Volume range: [{valid_data[y_col].min():.2f}, {valid_data[y_col].max():.2f}]")

            # Fit GAMLSS
            best_model, comparison, diagnostics = auto_select_gamlss(
                data=valid_data,  # Use cleaned data
                age_col=age_col,
                y_col=y_col,
                quick=False,
                verbose=True,
            )

            # Check if model actually converged
            if not best_model.converged_:
                print("WARNING: Model did not converge!")
                continue

            # Show diagnostics summary
            diagnostics.diagnostic_summary()

            # Plot results
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            plot_gamlss_comparison(valid_data, best_model, age_col, y_col, ax=axes[0])
            diagnostics.plot_qq(ax=axes[1])
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"gamlss_thalamus_{label}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            # Full diagnostic panel
            fig_diag = diagnostics.plot_all_diagnostics()
            fig_diag.savefig(
                os.path.join(output_dir, f"gamlss_thalamus_{label}_diagnostics.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            
            print(f"✓ Successfully processed nucleus {label}\n")
            
        except Exception as e:
            print(f"✗ ERROR processing nucleus {label}:")
            print(f"  {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        

# %%
# %% # THOMAS NUCLEI VOLUMES
import os
if __name__ == "__main__":
    # -----------------------------
    # Load and prepare data
    # -----------------------------
    data = pd.read_csv(
        "/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_raw.csv"
    )

    # Automatically find the age column
    age_col = next((c for c in data.columns if "age" in c.lower()), None)
    if age_col is None:
        raise ValueError("No column containing 'age' found in your CSV!")
    print(f"Using age column: {age_col}")

    # -----------------------------
    # Identify nucleus labels
    # -----------------------------
    volume_cols = [c for c in data.columns if c.startswith(("L_", "R_"))]
    labels = sorted(set(int(c.split("_")[1]) for c in volume_cols))

    # Compute bilateral volumes
    for label in labels:
        left = f"L_{label}"
        right = f"R_{label}"
        if left in data.columns and right in data.columns:
            data[f"whole_{label}"] = data[left] + data[right]
            
            # Print the first few combined values for verification
            print(f"\n--- {label} ---")
            print(f"Left ({left}): {data[left].head().tolist()}")
            print(f"Right ({right}): {data[right].head().tolist()}")
            print(f"Sum (whole_{label}): {data[f'whole_{label}'].head().tolist()}")  
    print(f"Computed bilateral volumes for {len(labels)} nuclei")

    # -----------------------------
    # Run GAMLSS per nucleus
    # -----------------------------
    output_dir = (
        "/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/figures"
    )
    os.makedirs(output_dir, exist_ok=True)

    for label in labels:
        y_col = f"whole_{label}"
        if y_col not in data.columns:
            continue

        print("=" * 60)
        print(f"GAMLSS: Thalamus nucleus {label}")
        print("=" * 60)

        # Clean data first
        valid_data = data[[age_col, y_col]].dropna()
        if len(valid_data) < 50:
            print(f"WARNING: Only {len(valid_data)} valid observations. Skipping.")
            continue

        # =================================================================
        # REPLACE YOUR auto_select_gamlss CALL WITH THIS:
        # =================================================================
        results = {}
        
        for dist_name in ['NO', 'BCCG', 'BCT']:
            print(f"\nFitting: {dist_name}")
            
            try:
                if dist_name == 'NO':
                    fitter = GAMLSSFitter(
                        distribution=dist_name,
                        age_col=age_col,
                        y_col=y_col,
                        mu_formula={'fixed_df': 5, 'degree': 3, 'random': False},
                        sigma_formula={'fixed_df': 3, 'degree': 3, 'random': False},
                        verbose=False
                    )
                elif dist_name == 'BCCG':
                    fitter = GAMLSSFitter(
                        distribution=dist_name,
                        age_col=age_col,
                        y_col=y_col,
                        mu_formula={'fixed_df': 5, 'degree': 3, 'random': False},
                        sigma_formula={'fixed_df': 3, 'degree': 3, 'random': False},
                        nu_formula={'fixed_df': 1, 'degree': 3, 'random': False},
                        verbose=False
                    )
                else:  # BCT
                    fitter = GAMLSSFitter(
                        distribution=dist_name,
                        age_col=age_col,
                        y_col=y_col,
                        mu_formula={'fixed_df': 5, 'degree': 3, 'random': False},
                        sigma_formula={'fixed_df': 3, 'degree': 3, 'random': False},
                        nu_formula={'fixed_df': 1, 'degree': 3, 'random': False},
                        tau_formula={'fixed_df': 1, 'degree': 3, 'random': False},
                        verbose=False
                    )
                
                # FIT THE MODEL
                fitter.fit(valid_data)
                
                # Store results
                results[dist_name] = {
                    'fitter': fitter,
                    'diagnostics': GAMLSSDiagnostics(fitter)
                }
                
                print(f"  {dist_name}: BIC={fitter.bic_:.2f}, Converged={fitter.converged_}")
                
            except Exception as e:
                print(f"  {dist_name} failed: {e}")

        if not results:
            print(f"  All models failed for nucleus {label}")
            continue

        # Choose best model
        best_dist = min(results.keys(), key=lambda d: results[d]['fitter'].bic_)
        best_model = results[best_dist]['fitter']
        diagnostics = results[best_dist]['diagnostics']

        print(f"\nBest model: {best_dist} (BIC={best_model.bic_:.1f})")

        # Show diagnostics summary
        diagnostics.diagnostic_summary()

        # Plot worm comparison for all three distributions
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (dist_name, result) in enumerate(results.items()):
            result['diagnostics'].plot_worm(ax=axes[i])
            axes[i].set_title(f"{dist_name}\nBIC={result['fitter'].bic_:.1f}")
        plt.suptitle(f"Nucleus {label}: Worm Plot Comparison")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"worm_comparison_{label}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

        # Plot results using best model
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        plot_gamlss_comparison(valid_data, best_model, age_col, y_col, ax=axes[0])
        diagnostics.plot_qq(ax=axes[1])

        plt.tight_layout()
        # plt.savefig(
        #     os.path.join(output_dir, f"gamlss_thalamus_{label}.png"),
        #     dpi=150,
        #     bbox_inches="tight",
        # )
        plt.show()

        # Full diagnostic panel
        fig_diag = diagnostics.plot_all_diagnostics()
        # fig_diag.savefig(
        #     os.path.join(output_dir, f"gamlss_thalamus_{label}_diagnostics.png"),
        #     dpi=150,
        #     bbox_inches="tight",
        # )
        plt.show()
        
        print(f"Saved plots for nucleus {label}\n")
# %%
