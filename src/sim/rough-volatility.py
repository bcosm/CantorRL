import numpy as np
import pandas as pd
import math
import sys

# Statistical helper functions
def mean(v):
    return np.mean(v) if len(v) > 0 else 0.0

def variance(v):
    if len(v) < 2:
        return 0.0
    return np.var(v, ddof=1)

def covariance(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return np.cov(x, y, ddof=1)[0, 1]

def detrend_segment(segment):
    n = len(segment)
    if n < 2:
        return segment
    t = np.arange(1, n+1, dtype=np.float64)
    tm = mean(t)
    ym = mean(segment)
    num = np.sum((t - tm) * (segment - ym))
    den = np.sum((t - tm) ** 2)
    if abs(den) < 1e-14:
        return segment
    slope = num / den
    intercept = ym - slope * tm
    # Remove the linear trend from the segment.
    return segment - (slope * t + intercept)

def hurst_exponent_DFA(data_in):
    data = np.array(data_in, dtype=np.float64)
    if len(data) < 2:
        return 0.5

    # Remove mean and create integrated process
    data = data - mean(data)
    data = np.cumsum(data)

    log_window_size = []
    log_fluctuation = []

    min_window_size = 4
    max_window_size = len(data) // 4
    w = min_window_size

    while w <= max_window_size:
        fluctuations = []
        # Process non-overlapping segments of size w
        for start in range(0, len(data) - w + 1, w):
            segment = data[start:start+w]
            detrended = detrend_segment(segment)
            rms = np.sqrt(np.mean(detrended**2))
            fluctuations.append(rms)
        mf = mean(fluctuations)
        if mf > 0.0:
            log_window_size.append(np.log(w))
            log_fluctuation.append(np.log(mf))
        w *= 2

    n = len(log_window_size)
    if n < 2:
        return 0.5

    sumX = np.sum(log_window_size)
    sumY = np.sum(log_fluctuation)
    sumXX = np.sum(np.array(log_window_size)**2)
    sumXY = np.sum(np.array(log_window_size)*np.array(log_fluctuation))
    slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX**2)
    return slope

# Rough Volatility model class
class RoughVolatility:
    def __init__(self):
        pass

    def log_returns(self, prices):
        prices = np.array(prices, dtype=np.float64)
        if len(prices) < 2:
            return np.array([])
        return np.log(prices[1:] / prices[:-1])

    def estimate_R(self, logrets, dt_yr):
        mu = mean(logrets)
        annual_mu = mu / dt_yr
        return annual_mu

    def estimate_xi(self, logrets, dt_yr):
        var_r = variance(logrets)
        annual_var = var_r / dt_yr
        return annual_var

    def estimate_H(self, logrets):
        return hurst_exponent_DFA(logrets)

    def estimate_eta(self, logrets, H, window=20):
        import numpy as np
        import math

        if len(logrets) < window:
            stdev = math.sqrt(np.var(logrets, ddof=1))
            return stdev * 2.0

        realized_var = []
        for i in range(window - 1, len(logrets)):
            window_returns = logrets[i - window + 1:i + 1]
            rv = np.mean(np.square(window_returns))
            realized_var.append(rv)
        realized_var = np.array(realized_var)
        
        log_rv = np.log(realized_var)
        log_diff = np.diff(log_rv)
        
        daily_eta = np.std(log_diff, ddof=1)
        
        annual_eta = daily_eta * np.sqrt(252)
        return annual_eta


    def estimate_rho(self, logrets):
        logrets = np.array(logrets, dtype=np.float64)
        sq = logrets**2
        c = covariance(logrets, sq)
        denom = math.sqrt(variance(logrets) * variance(sq))
        rho = c / denom if denom != 0.0 else 0.0
        if rho > 0.0:
            rho = -0.3
        return rho

    def next_power_of_two(self, n):
        p = 1
        while p < n:
            p <<= 1
        return p

    def rbergomi_lambda(self, time_grid, H):
        time_grid = np.array(time_grid, dtype=np.float64)
        return 0.5 * (time_grid ** (2 * H))

    def rbergomi_phi(self, lam, H):
        N = len(lam)
        M = self.next_power_of_two(N)
        # Pad lam with zeros up to M.
        lam_padded = np.zeros(M, dtype=np.float64)
        lam_padded[:N] = lam
        # Use numpy's FFT.
        phi = np.fft.fft(lam_padded)
        return phi

    def gen_complex_gaussians(self, N):
        # Generate N complex Gaussian random numbers.
        re = np.random.normal(0.0, 1.0, N)
        im = np.random.normal(0.0, 1.0, N)
        return re + 1j * im

    def gaussians(self, N):
        return np.random.normal(0.0, 1.0, N)

    def fractional_gaussian(self, phi, Z, H, eta):
        N = len(Z)
        M = self.next_power_of_two(N)
        A = phi.copy()  # phi is already length M
        # Multiply element-wise: note that only first N elements of Z are non-padded.
        A[:N] = A[:N] * Z
        # Inverse FFT using numpy (which scales by 1/M automatically)
        A_ifft = np.fft.ifft(A)
        res = A_ifft.real
        scale = math.sqrt(2 * H) * eta
        X = scale * res
        # Return only the first N elements
        return X[:N]

    def forward_variance(self, X, t_grid, xi, H, eta):
        N = len(X)
        v = np.zeros(N, dtype=np.float64)
        t_grid = np.array(t_grid, dtype=np.float64)
        for i in range(N):
            t = t_grid[i]
            ma = -0.5 * eta * eta * (t ** (2 * H))
            v[i] = xi * np.exp(X[i] + ma)
        return v

    def generate_stock_price_paths(self, historical_prices, forward_steps, path_num):
        historical_prices = np.array(historical_prices, dtype=np.float64)
        if len(historical_prices) < 2:
            raise ValueError("Historical prices vector too small.")

        dt_yr = 1.0 / 252.0  # one trading day in years
        dt = dt_yr

        rets = self.log_returns(historical_prices)
        # In the C++ code, r is hard-coded to 0.04.
        r = 0.04
        xi = self.estimate_xi(rets, dt_yr)
        H = self.estimate_H(rets)
        eta = self.estimate_eta(rets, H)
        rho = self.estimate_rho(rets)
        S0 = historical_prices[-1]

        num_paths = path_num
        num_steps = forward_steps
        T = num_steps * dt

        time_grid = np.linspace(0, num_steps*dt, num_steps + 1)
        lam = self.rbergomi_lambda(time_grid, H)
        phi = self.rbergomi_phi(lam, H)

        # Pre-allocate array for price paths.
        paths = np.zeros((num_paths, num_steps + 1), dtype=np.float64)

        for i in range(num_paths):
            Z = self.gen_complex_gaussians(num_steps)
            X = self.fractional_gaussian(phi, Z, H, eta)
            v = self.forward_variance(X, time_grid, xi, H, eta)
            W1 = self.gaussians(num_steps)
            W2 = self.gaussians(num_steps)

            paths[i, 0] = S0
            for j in range(1, num_steps + 1):
                dw1 = math.sqrt(dt) * W1[j - 1]
                dw2 = math.sqrt(dt) * W2[j - 1]
                dW = rho * dw1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * dw2
                vt = v[j - 1]
                drift = (r - 0.5 * vt) * dt
                diff = math.sqrt(max(0.0, vt)) * dW
                paths[i, j] = paths[i, j - 1] * math.exp(drift + diff)
        return paths

def main():
    # Input CSV file name (should be in the same folder).
    input_filename = "scripts/simulation/historical_prices.csv"
    output_filename = "scripts/simulation/price_paths.csv"

    try:
        # Try reading the CSV assuming a single column (with or without header).
        try:
            df = pd.read_csv(input_filename)
            # If there is more than one column, assume the first column contains the prices.
            prices = df.iloc[:, 0].values
        except Exception as e:
            print("Error reading CSV:", e)
            sys.exit(1)
    except FileNotFoundError:
        print(f"File {input_filename} not found.")
        sys.exit(1)

    # Set simulation parameters.
    forward_steps = 252   # number of time steps forward
    path_num = 500         # number of simulated paths

    rv = RoughVolatility()
    paths = rv.generate_stock_price_paths(prices, forward_steps, path_num)

    # Create a DataFrame to save paths. Each row represents one simulation path.
    df_paths = pd.DataFrame(paths)
    df_paths.to_csv(output_filename, index=False)
    print(f"Simulated price paths saved to {output_filename}")

if __name__ == "__main__":
    main()
