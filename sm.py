"""
Multitask Spectral Mixture GP for solar + wind at a single bus.
------------------------------------------------------------------
Fits a joint GP over [solar(t), wind(t)] using:
    k((t,task),(t',task')) = k_SM(t,t') * B[task,task']
- k_SM: Spectral Mixture kernel (captures daily/weekly periodicity)
- B:    2x2 task-covariance matrix (the solar-wind "interaction")

REPLACE the load_data() function with your actual RE-Europe loading logic.
Everything else should run as-is.
"""

import numpy as np
import torch
import gpytorch
from scipy.signal import lombscargle
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove if running in a notebook
import matplotlib.pyplot as plt


# ----------------------------------------------------------------
# 1. DATA LOADING  -- replace this with your real RE-Europe extraction
# ----------------------------------------------------------------
def load_data(n_hours=24 * 21, bus_id=1, seed=0):
    """
    Placeholder generator standing in for real RE-Europe solar/wind series.
    Replace with something like:
        df = pd.read_csv("RE-Europe/solar_signal.csv")
        solar = df[str(bus_id)].values
        df2 = pd.read_csv("RE-Europe/wind_signal.csv")
        wind = df2[str(bus_id)].values
        t = np.arange(len(solar))
    Must return: t (n,), solar (n,), wind (n,)  as 1D numpy arrays, same length.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)

    daily = np.sin(2 * np.pi * t / 24)
    weekly = np.sin(2 * np.pi * t / (24 * 7))

    # solar: strong daily cycle, ~0 at night (clip), some weekly weather modulation
    solar = np.clip(daily, 0, None) * (1 + 0.3 * weekly) + 0.05 * rng.standard_normal(n_hours)
    solar = np.clip(solar, 0, None)

    # wind: weaker daily pattern, stronger synoptic (weekly) swings,
    # anti-correlated with solar via shared weather driver
    weather = 0.6 * weekly + 0.4 * rng.standard_normal(n_hours) * 0  # placeholder for real weather signal
    wind = 0.5 + 0.3 * weekly - 0.4 * np.clip(daily, 0, None) + 0.1 * rng.standard_normal(n_hours)
    wind = np.clip(wind, 0, None)

    return t.astype(float), solar, wind


# ----------------------------------------------------------------
# 2. MODEL
# ----------------------------------------------------------------
class MultitaskSMModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures=4, num_tasks=2, rank=1):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        data_covar = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            data_covar, num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def init_sm_frequencies(covar_module, t_raw, y_pooled, num_mixtures, t_scale):
    """
    Manual frequency init via Lomb-Scargle, as a more robust alternative
    to .initialize_from_data() which can behave inconsistently inside
    a MultitaskKernel wrapper.

    IMPORTANT UNITS NOTE: Lomb-Scargle is run on the RAW time axis (t_raw,
    e.g. in hours) since that's the most interpretable/numerically stable
    space to search for periods in. But the GP itself is trained on a
    RESCALED time axis (mean 0, std 1) for optimization stability. A
    frequency found in raw-hour units is NOT the same number as the
    corresponding frequency in rescaled units -- you must multiply by
    t_scale (the std used to rescale t) to convert.

    IMPORTANT PEAK-SELECTION NOTE: naively taking the top-N periodogram
    bins by power tends to select N adjacent bins from the SAME dominant
    peak (e.g. 4 bins all describing the weekly cycle) rather than N
    genuinely distinct periodicities (e.g. daily AND weekly). This starves
    the model of a real daily-cycle component whenever the weekly/synoptic
    signal has much higher raw power than the daily one -- which is common
    for wind, and even for solar once you pool with wind. We use
    scipy.signal.find_peaks with a minimum bin separation to enforce that
    selected frequencies are genuinely distinct before ranking by power.
    """
    from scipy.signal import find_peaks

    freqs_raw = np.linspace(1 / len(t_raw), 0.5, 4000)  # cycles per raw-time-unit (e.g. per hour)
    power = lombscargle(t_raw, y_pooled - y_pooled.mean(), freqs_raw * 2 * np.pi, normalize=True)

    # minimum separation between selected peaks: ~5% of the frequency range,
    # enough to keep daily/weekly/etc. bands from colliding
    min_dist = max(int(len(freqs_raw) * 0.02), 1)
    peak_idx, properties = find_peaks(power, distance=min_dist)

    if len(peak_idx) >= num_mixtures:
        # take the num_mixtures strongest DISTINCT peaks
        top = peak_idx[np.argsort(power[peak_idx])[-num_mixtures:]]
    else:
        # fallback: not enough distinct peaks found, pad with top raw bins
        top = np.argsort(power)[-num_mixtures:]

    mu_raw = np.sort(freqs_raw[top])
    mu_scaled = mu_raw * t_scale  # convert cycles/raw-unit -> cycles/rescaled-unit

    sm_kernel = covar_module.data_covar_module if hasattr(covar_module, "data_covar_module") else covar_module
    sm_kernel.mixture_means = torch.tensor(mu_scaled, dtype=torch.float32).reshape(num_mixtures, 1, 1)
    # give each component a distinct starting bandwidth proportional to its own
    # frequency (a shared bandwidth for both a 24h and 168h component is a poor
    # starting point for either one)
    sm_kernel.mixture_scales = torch.tensor(mu_scaled / 10, dtype=torch.float32).reshape(num_mixtures, 1, 1)
    sm_kernel.mixture_weights = torch.full((num_mixtures,), 1.0 / num_mixtures, dtype=torch.float32)
    return mu_raw  # return raw-unit periods since those are human-interpretable (e.g. hours)


# ----------------------------------------------------------------
# 3. MAIN PIPELINE
# ----------------------------------------------------------------
def main():
    torch.manual_seed(0)

    # --- load + split ---
    t, solar, wind = load_data()
    n = len(t)
    n_train = int(n * 0.8)

    # --- standardize (fit scaler on train only) ---
    solar_mean, solar_std = solar[:n_train].mean(), solar[:n_train].std()
    wind_mean, wind_std = wind[:n_train].mean(), wind[:n_train].std()
    solar_z = (solar - solar_mean) / solar_std
    wind_z = (wind - wind_mean) / wind_std

    t_scaled = (t - t[:n_train].mean()) / t[:n_train].std()  # scale time to ~unit range

    train_x = torch.tensor(t_scaled[:n_train], dtype=torch.float32)
    test_x = torch.tensor(t_scaled[n_train:], dtype=torch.float32)

    train_y = torch.tensor(
        np.stack([solar_z[:n_train], wind_z[:n_train]], axis=1), dtype=torch.float32
    )
    test_y = torch.tensor(
        np.stack([solar_z[n_train:], wind_z[n_train:]], axis=1), dtype=torch.float32
    )

    # --- model + likelihood ---
    num_mixtures = 4
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskSMModel(train_x, train_y, likelihood, num_mixtures=num_mixtures)

    # robust frequency init (pooled solar+wind signal for peak-finding)
    t_scale = t[:n_train].std()
    pooled = solar_z[:n_train] + wind_z[:n_train]
    mu_init_raw = init_sm_frequencies(model.covar_module, t[:n_train], pooled, num_mixtures, t_scale)
    print(f"Initial mixture periods (in raw time units, e.g. hours): {sorted(1/mu_init_raw)}")

    # empirical-correlation init for the task-covariance (rank-1) vector.
    # Without this, the rank-1 vector starts near-random and commonly
    # collapses to ~0 implied correlation during optimization (see below).
    emp_corr = np.corrcoef(solar_z[:n_train], wind_z[:n_train])[0, 1]
    print(f"Empirical solar-wind correlation (raw data): {emp_corr:.3f}")
    with torch.no_grad():
        # task_covar_factor has shape (num_tasks, rank); set task 0 to unit
        # weight and task 1 proportional to the empirical correlation sign/magnitude
        model.covar_module.task_covar_module.covar_factor.data = torch.tensor(
            [[1.0], [float(np.sign(emp_corr)) * max(abs(emp_corr), 0.3)]], dtype=torch.float32
        )

    # --- train ---
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_iter = 150
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print(f"iter {i+1}/{n_iter} - loss: {loss.item():.4f}")

    # --- diagnostics: learned frequencies + task covariance ---
    sm = model.covar_module.data_covar_module
    learned_freqs_scaled = sm.mixture_means.detach().numpy().flatten()
    learned_periods_raw = t_scale / learned_freqs_scaled  # convert back to raw time units (e.g. hours)
    print(f"\nLearned mixture periods (in raw time units, e.g. hours): {sorted(learned_periods_raw)}")
    print("  (check whether these land near ~24h and ~168h if using hourly data)")

    covar_matrix = model.covar_module.task_covar_module.covar_matrix
    if hasattr(covar_matrix, "to_dense"):
        B = covar_matrix.to_dense().detach().numpy()
    else:
        B = covar_matrix.evaluate().detach().numpy()  # older gpytorch versions
    implied_corr = B[0, 1] / np.sqrt(B[0, 0] * B[1, 1])
    print(f"\nLearned task-covariance matrix B:\n{B}")
    print(f"Implied solar-wind correlation: {implied_corr:.3f}")

    # --- predict on held-out test set ---
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
        mean = preds.mean.numpy()
        lower, upper = preds.confidence_region()
        lower, upper = lower.numpy(), upper.numpy()

    # unstandardize for reporting
    solar_pred = mean[:, 0] * solar_std + solar_mean
    wind_pred = mean[:, 1] * wind_std + wind_mean
    solar_true = test_y[:, 0].numpy() * solar_std + solar_mean
    wind_true = test_y[:, 1].numpy() * wind_std + wind_mean

    rmse_solar = np.sqrt(np.mean((solar_pred - solar_true) ** 2))
    rmse_wind = np.sqrt(np.mean((wind_pred - wind_true) ** 2))
    print(f"\nTest RMSE - solar: {rmse_solar:.4f}, wind: {rmse_wind:.4f}")

    # --- plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, true, pred, name in zip(
        axes, [solar_true, wind_true], [solar_pred, wind_pred], ["Solar", "Wind"]
    ):
        ax.plot(true, label="true", lw=1)
        ax.plot(pred, label="predicted", lw=1)
        ax.set_title(f"{name} (test set)")
        ax.legend()
    plt.tight_layout()
    plt.savefig("/home/claude/sm_kernel_fit.png", dpi=120)
    print("\nSaved diagnostic plot to sm_kernel_fit.png")


if __name__ == "__main__":
    main()