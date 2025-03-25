# statistics for scatter plots
import numpy as np
from scipy.signal import correlate

# Collection of statistics used to evaluate agreement between
# two sets of points (e.g. scatterplots).
# No attempt to optimize or vectorize these has been performed.

# Might want to pass flattened arrays to these

# S = simulated (modeled) data points
# O = observed (measured, real) data points

# Statistics from MBCM
# Menaschi et al. (2013) Problems in RMSE-based wave model validations" Ocean Modeling 72:53-58.

def calc_HH( S, O ):
    """Equation MBCM 22
    The authors argue that this statistic is better than RMSE or NRMSE especially
    when null or negative biase and lots of scatter exist.
    Dimensionless.
    """
    return np.sqrt( np.nansum( ( S - O )**2 ) / np.nansum( S*O ) )

def calc_bias( S, O, verbose=False ):
    """Equation MBCM 6
    Units are same as input.
    """
    ok = np.isfinite(S+O) # eliminate if NaNs in either dataset
    bias = np.mean( S[ok] ) - np.mean( O[ok] )
    if( verbose ):
        print("Bias: {:.4f} Positive means model is higher.".format(bias))
    return np.mean( S[ok] ) - np.mean( O[ok] )

def calc_MAD ( S, O ):
    """Mean absolute difference
    """
    ok = np.isfinite(S+O) # eliminate if NaNs in either dataset
    return np.mean( np.abs (S - O ) )


def calc_nbias( S, O ):
    """Normalized bias
    Dimensionless
    Is this a thing?
    """
           
def calc_RMSE( S, O ):
    """Equation MBCM  2
    Units are same as input.
    """
    ok = np.isfinite(S+O) # eliminate if NaNs in either dataset
    return np.sqrt( np.mean( (S[ok]-O[ok])**2 ) )

def calc_NRMSE( S, O ):
    """Equation MBCM 1
    Dimensionless.
    """
    ok = np.isfinite(S+O) # eliminate if NaNs in either dataset
    return np.sqrt( np.sum( (S[ok]-O[ok])**2 ) / np.sum( O[ok]**2 ) )

# Other stats
def calc_rho( S, O ):
    """Correlaton coefficient AKA Pearsons r
    Ranges from -1 (perfect negative correlation) to 0 (no correlation) to +1 (perfect correlation)
    Dimensionless.
    """
    ok = np.isfinite(S+O) # eliminate if NaNs in either dataset
    # longhand to check...answers match
    # Sbar = np.mean(S[ok])
    # Obar = np.mean(O[ok])
    # Sd = (S[ok]-Sbar)
    # Od = (O[ok]-Obar)
    # rho2 = ( np.sum( Sd*Od )) / np.sqrt( np.sum( Sd**2 ) * np.sum( Od**2 ) )
    rho =  np.corrcoef( S[ok], O[ok] )[0,1]
    return rho

def calc_lag_corr( mod, obs, delta_t, verbose=False):
    # Assumes time series are equal length, eaual time steps
    # Assumes delta_t is in seconds
    if verbose:
        print('Pos. lag indicates model is earlier.')
    # Remove mean
    obs = obs - np.mean(obs)
    mod = mod - np.mean(mod)
    
    # Compute standard deviations
    std_obs = np.std(obs)
    std_mod = np.std(mod)
    
    # Compute cross-correlation
    n = len(obs)
    cross_corr = correlate(obs, mod, mode='full', method='auto')
    
    # Generate lag values
    lags = np.arange(-n + 1, n )
    
    # Normalize by the product of standard deviations and length
    cross_corr /= (n * std_obs * std_mod)
    
    # Find correlation at zero lag
    zero_lag_idx = np.where(lags == 0)[0][0]
    zero_lag_corr = cross_corr[zero_lag_idx]
    
    # Find max correlation and corresponding lag
    max_corr_idx = np.argmax(cross_corr)
    max_corr = cross_corr[max_corr_idx]
    max_lag = lags[max_corr_idx]  # Lag at which max correlation occurs
    max_lags = (max_lag * delta_t).astype(int)
    
    # Print results
    print(f"Zero-lag Correlation: {zero_lag_corr:.3f}")
    print(f"Max Correlation: {max_corr:.3f} at Lag = {max_lag} time steps")

    ts = "Corr at lag( 0 ): {:.3f}\nMax corr: {:.3f} at lag( {:d} s )".format( zero_lag_corr, max_corr, max_lags )
    return zero_lag_corr, max_corr, max_lags, ts

def calc_WSS( S, O ):
    """
    Willmott skill score
    WSS = 1: Perfect model (no error).
    WSS = 0: Model has the same performance as just predicting the mean of the observed data.
    WSS < 0: The model performs worse than predicting the mean.
    """
    ok = np.isfinite(S+O) # eliminate if NaNs in either dataset
    # Calculate the mean of the observed data
    observed_mean = np.mean( O )
    # Calculate the numerator and denominator of the WSS formula
    numerator = np.sum((O[ok] - S[ok]) ** 2)
    denominator = np.sum((np.abs( O[ok] - observed_mean)) ** 2)
    # Calculate the two terms in the denominator
    denominator = np.sum( (S[ok] - observed_mean) + (O[ok] - observed_mean) ** 2)
    # Calculate Willmott's Skill Score
    wss = 1 - (numerator / denominator)
    return wss


def scat_stats_array( S, O ):
    S = S.flatten()
    O = O.flatten()
    assert S.shape == O.shape
    N = np.prod(O.shape)
    Nnan = np.sum(np.isnan(O))
    RMSE = calc_RMSE( S, O )
    rho = calc_rho( S, O )
    bias = calc_bias( S, O )
    NRMSE = calc_NRMSE( S, O )
    HH = calc_HH( S, O )
    return np.array([N, Nnan, RMSE, rho, bias, NRMSE, HH])

def scat_stats_string( S, O, sep_lines=True ):
    a = scat_stats_array( S, O )
    s = 'N: {0:.0f}\nNnan: {1:.0f}\nRMSE: {2:.3f}\nrho: {3:.3f}\nBias: {4:.3f}\nNRMSE: {5:.3f}\nHH: {6:.3f}'.\
    format( a[0],a[1],a[2],a[3],a[4], a[5], a[6] ) 
    return a, s
    
    