import pandas as pd
import numpy as np
from scipy.signal import correlate

class wl_stats:
    def __init__(self, sim_name, df_sim, obs_name, df_obs, start_time, end_time):
        self.sim_name = sim_name  # Store the simulation name
        self.obs_name = obs_name  # Store the observation name
        self.df_sim = self.preprocess(df_sim, start_time, end_time)
        self.df_obs = self.preprocess(df_obs, start_time, end_time)
        self.time = self.df_obs['time']
        self.S = self.df_sim['water level'].values
        self.O = self.df_obs['water level'].values
        self.calculate_stats()
        self.find_peak_levels()

    def preprocess(self, df, start_time, end_time):
        df = df.copy()  # Ensure a new copy to prevent warnings
        df.loc[:, "time"] = pd.to_datetime(df["time"]) # ensure 'time'
        df = df.set_index("time").resample("h").nearest().reset_index()
        df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
        return df

    def calculate_stats(self):
        ok = np.isfinite(self.S) & np.isfinite(self.O)  # ensure valid pairs
        S_valid = self.S[ok]
        O_valid = self.O[ok]
        self.N = len(S_valid)  # Number of valid pairs
        self.bias = np.mean(S_valid) - np.mean(O_valid)
        self.MAD = np.mean(np.abs(S_valid - O_valid))
        self.RMSE = np.sqrt(np.mean((S_valid - O_valid) ** 2))
        self.NRMSE = np.sqrt(np.sum((S_valid - O_valid) ** 2) / np.sum(O_valid ** 2))
        # TODO add "skill" calc
        self.rho = np.corrcoef(S_valid, O_valid)[0, 1]
        self.WSS = 1 - (np.sum((O_valid - S_valid) ** 2) / np.sum((np.abs(O_valid - np.mean(O_valid))) ** 2))
        self.lag_corr_analysis()

    def lag_corr_analysis(self):
        S_interp = pd.Series(self.S).interpolate().values    # fills any missing values
        O_interp = pd.Series(self.O).interpolate().values
        delta_t = (self.time.diff().dropna().dt.total_seconds()).iloc[0]
        obs = O_interp - np.mean(O_interp)
        mod = S_interp - np.mean(S_interp)
        n = len(obs)
        cross_corr = correlate(obs, mod, mode="full", method="auto")
        lags = np.arange(-n + 1, n)
        cross_corr /= (n * np.std(obs) * np.std(mod))
        zero_lag_idx = np.where(lags == 0)[0][0]
        self.zero_lag_corr = cross_corr[zero_lag_idx]
        max_corr_idx = np.argmax(cross_corr)
        self.max_corr = cross_corr[max_corr_idx]
        self.max_lag = (lags[max_corr_idx] * delta_t).astype(int)

    def find_peak_levels(self):
        obs_peak_idx = self.df_obs["water level"].idxmax()
        self.obs_peak_time = self.df_obs.loc[obs_peak_idx, "time"]
        self.obs_peak_value = self.df_obs.loc[obs_peak_idx, "water level"]
        pred_peak_idx = self.df_sim["water level"].idxmax()
        self.pred_peak_time = self.df_sim.loc[pred_peak_idx, "time"]
        self.pred_peak_value = self.df_sim.loc[pred_peak_idx, "water level"]

    def generate_summary(self):
        return (f"{self.sim_name} v. {self.obs_name}\n"
                f"N = {self.N}\n"
                f"Bias: {self.bias:.4f}\nMAD: {self.MAD:.4f}\nRMSE: {self.RMSE:.4f}\n"
                f"NRMSE: {self.NRMSE:.4f}\nWSS: {self.WSS:.4f}\n"
                f"Zero-lag Correlation: {self.zero_lag_corr:.4f}\nMax Correlation: {self.max_corr:.4f} at Lag {self.max_lag} s\n"
                f"Observed Peak: {self.obs_peak_value:.4f} at {self.obs_peak_time}\n"
                f"Modeled Peak: {self.pred_peak_value:.4f} at {self.pred_peak_time}\n"
                f"Number of valid pairs: {self.N}")

    def write_csv(self, filename):
        stats = {
            "Model":self.sim_name,
            "Obs.":self.obs_name,
            "N": self.N,
            "Bias": self.bias,
            "MAD": self.MAD,
            "RMSE": self.RMSE,
            "NRMSE": self.NRMSE,
            "Correlation": self.rho,
            "WSS": self.WSS,
            "Zero-lag Correlation": self.zero_lag_corr,
            "Max Correlation": self.max_corr,
            "Max Lag (s)": self.max_lag,
            "Obs Peak Value": self.obs_peak_value,
            "Obs Peak Time": self.obs_peak_time,
            "Pred Peak Value": self.pred_peak_value,
            "Pred Peak Time": self.pred_peak_time
        }
        if os.path.exists(filename):
            pd.DataFrame([stats]).to_csv(filename, mode='a', header=False, index=False)
        else:
            pd.DataFrame([stats]).to_csv(filename, index=False)

# Usage Example:
# df_sim and df_obs should be DataFrames with 'time' and 'water level' columns
# start_time and end_time should be pandas timestamps
#
# stats = wl_stats(df_sim, df_obs, start_time, end_time)
# print(stats.generate_summary())
# stats.write_csv("output_stats.csv")
