"""
POD-ML-ROM Joint Training — St and Cf together
================================================
Trains a single surrogate model that maps shared physics inputs to the
concatenated POD coefficients of BOTH quantities of interest:

    [log10(Kn), Ma, τ]  →  [α_St (L_st modes) | α_Cf (L_cf modes)]

Three surrogates benchmarked:
  1. Joint Neural Network   — shared hidden layers learn St/Cf correlations
  2. Joint RBF Interpolation — single thin-plate spline for all outputs
  3. Joint GPR               — one GP per output mode (parallelised)

St and Cf coefficients are scaled independently before concatenation so
neither field dominates the training loss.

Log:   logs/training_log_joint.txt
"""

import sys
import time as time_module
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from scipy.interpolate import RBFInterpolator

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
POD_DIR   = Path("pod_results")
MODEL_DIR = Path("trained_models")
DATA_DIR  = Path("data_extraction")
LOG_DIR   = Path("logs")
for d in [MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical constants (Argon VHS)
# ---------------------------------------------------------------------------
R_ARGON   = 208.14
GAMMA     = 5.0 / 3.0
MU_REF    = 50.7e-6
T_REF     = 1000.0
OMEGA_VHS = 0.72
L_PLATE   = 0.055

# ---------------------------------------------------------------------------
# Case parameters
# ---------------------------------------------------------------------------
CASES = {
    "case1":   {"T": 80,   "p": 2.020, "U": 1600},
    "case2":   {"T": 120,  "p": 2.200, "U": 1800},
    "case3":   {"T": 160,  "p": 2.260, "U": 2000},
    "case4":   {"T": 220,  "p": 2.370, "U": 2200},
    "case5":   {"T": 300,  "p": 2.710, "U": 2500},
    "case6":   {"T": 400,  "p": 3.140, "U": 2800},
    "case7":   {"T": 500,  "p": 3.480, "U": 3200},
    "case8":   {"T": 600,  "p": 3.770, "U": 3600},
    "case9":   {"T": 700,  "p": 4.010, "U": 4000},
    "case10":  {"T": 800,  "p": 4.240, "U": 4500},
    "case11":  {"T": 80,   "p": 0.202, "U": 1500},
    "case12":  {"T": 150,  "p": 0.289, "U": 1900},
    "case13":  {"T": 200,  "p": 0.296, "U": 2300},
    "case14":  {"T": 280,  "p": 0.318, "U": 2700},
    "case15":  {"T": 380,  "p": 0.360, "U": 3100},
    "case16":  {"T": 480,  "p": 0.392, "U": 3500},
    "case17":  {"T": 580,  "p": 0.418, "U": 3900},
    "valcase": {"T": 64.5, "p": 3.73,  "U": 1893.7},
    # T-S slip test case (Kn=0.050)
    "test1":   {"T": 350,  "p": 2.929, "U": 2650},
}

# ---------------------------------------------------------------------------
# NN hyperparameters
# ---------------------------------------------------------------------------
N_RESTARTS = 10
N_EPOCHS   = 15000
BATCH_SIZE = None    # None → full-batch (better for small N=120)
L2_REG     = 1e-4    # stronger regularisation — network was overfitting from epoch 1
HIDDEN     = [64, 64, 32]   # smaller: 120 training points can't support [128,128,64,64,32]
PATIENCE   = 800     # give NN more room before stopping

# File-index → physical time mapping
_KNOWN_TIMES = {
    1: 5e-05, 2: 1e-04, 3: 1.5e-04, 4: 2e-04,  5: 2.5e-04,
    6: 3e-04, 7: 3.5e-04, 8: 4e-04, 9: 4.5e-04, 10: 5e-04,
}

FIELDS = ["stanton", "cf"]   # order must stay consistent throughout


# ===========================================================================
# PHYSICS FEATURES
# ===========================================================================

def physics_features(P):
    """[t, T, p, U] → [log10(Kn), Ma, tau=t·U/L]"""
    t, T, p, U = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    mu  = MU_REF * (T / T_REF) ** OMEGA_VHS
    lam = (mu / p) * np.sqrt(np.pi * R_ARGON * T / 2.0)
    Kn  = lam / L_PLATE
    Ma  = U / np.sqrt(GAMMA * R_ARGON * T)
    tau = t * U / L_PLATE
    return np.column_stack([np.log10(Kn), Ma, tau])


# ===========================================================================
# SCALER
# ===========================================================================

class MeanNormalizer:
    """x̃ = (x − x̄) / (x_max − x_min)"""
    def fit(self, X):
        self.mean_  = X.mean(axis=0)
        rng         = X.max(axis=0) - X.min(axis=0)
        rng[rng < 1e-12] = 1.0
        self.scale_ = rng
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X_sc):
        return X_sc * self.scale_ + self.mean_


# ===========================================================================
# DATA BUILDERS
# ===========================================================================

def build_param_matrix(case_list, time_values):
    """Raw P = [t, T, p, U] in time-major order."""
    rows = []
    for t in time_values:
        for cname in case_list:
            c = CASES[cname]
            rows.append([t, c["T"], c["p"], c["U"]])
    return np.array(rows, dtype=float)


def load_fom_fields(case_list, time_values, field_name):
    """
    Load actual FOM spatial fields from .dat files in time-major order.
    Returns (N_snapshots, 399).
    """
    prefix = "stanton" if field_name == "stanton" else "cf"
    sub    = "St"      if field_name == "stanton" else "Cf"
    rows   = []
    for t_val in time_values:
        t_idx = next(
            (k for k, v in _KNOWN_TIMES.items() if np.isclose(v, t_val, rtol=1e-6)),
            None
        )
        if t_idx is None:
            raise ValueError(f"Time {t_val:.2e} s not in known mapping")
        fpath = DATA_DIR / sub / f"{prefix}_t{t_idx}.dat"
        data  = np.loadtxt(fpath)
        with open(fpath) as f:
            header_cases = f.readline().strip("# \n").split()
        for cname in case_list:
            rows.append(data[:, header_cases.index(cname)])
    return np.array(rows)   # (N_snapshots, 399)


# ===========================================================================
# RECONSTRUCTION & ERRORS
# ===========================================================================

def reconstruct_fields(alpha_pred, fom_fields, Phi, S_mean):
    """
    Reconstruct ROM prediction and compute relative L2 errors vs actual FOM.

    Args:
        alpha_pred : (N, L)    predicted POD coefficients
        fom_fields : (N, N_x)  actual FOM fields from disk
        Phi        : (N_x, L)  POD basis
        S_mean     : (N_x, 1)  training mean

    Returns:
        fields_pred, fields_true, rel_errs
    """
    fields_pred = (Phi @ alpha_pred.T + S_mean).T
    fields_true = fom_fields
    norms       = np.linalg.norm(fields_true, axis=1)
    rel_errs    = (np.linalg.norm(fields_pred - fields_true, axis=1)
                   / (norms + 1e-20))
    return fields_pred, fields_true, rel_errs


def error_summary(label, rel_errs):
    print(f"    {label:15s}: mean = {rel_errs.mean()*100:.3f}%,  "
          f"max = {rel_errs.max()*100:.3f}%")


# ===========================================================================
# JOINT PREDICTION SPLITTER
# ===========================================================================

def split_and_decode(Y_joint, L_st, out_sc_st, out_sc_cf):
    """
    Split joint prediction [alpha_St_sc | alpha_Cf_sc] and
    inverse-transform each field independently.

    Returns:
        alpha_st : (N, L_st)
        alpha_cf : (N, L_cf)
    """
    alpha_st = out_sc_st.inverse_transform(Y_joint[:, :L_st])
    alpha_cf = out_sc_cf.inverse_transform(Y_joint[:, L_st:])
    return alpha_st, alpha_cf


# ===========================================================================
# MODEL 1 — JOINT NEURAL NETWORK
# ===========================================================================

def build_nn(n_in, n_out):
    reg   = keras.regularizers.l2(L2_REG)
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_in,)))
    for units in HIDDEN:
        model.add(layers.Dense(units, activation='elu', kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(n_out))
    return model


def train_nn_joint(X_train, Y_train, X_val, Y_val, n_out, plot_dir):
    print(f"\n  [NN-Joint] Architecture : {X_train.shape[1]} → {HIDDEN} → {n_out} "
          f"(ELU + BN)")
    print(f"             Restarts × Epochs : {N_RESTARTS} × {N_EPOCHS}, "
          f"batch = {BATCH_SIZE}")

    cbs = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=150, min_lr=1e-6, verbose=0),
        callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                restore_best_weights=True, verbose=0),
    ]

    best_val_loss = np.inf
    best_model    = None
    histories     = []
    t0 = time_module.time()

    for r in range(N_RESTARTS):
        tf.random.set_seed(r * 42)
        np.random.seed(r * 42)
        model = build_nn(X_train.shape[1], n_out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
        bs = len(X_train) if BATCH_SIZE is None else min(BATCH_SIZE, len(X_train))
        h = model.fit(X_train, Y_train,
                      validation_data=(X_val, Y_val),
                      epochs=N_EPOCHS,
                      batch_size=bs,
                      callbacks=cbs, verbose=0, shuffle=True)
        val_loss = min(h.history['val_loss'])
        histories.append(h)
        print(f"             Restart {r+1:2d}/{N_RESTARTS}: val_loss = {val_loss:.4e} "
              f"({len(h.history['loss'])} epochs)")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model    = model

    elapsed = time_module.time() - t0
    print(f"             Best val MSE : {best_val_loss:.4e}  ({elapsed:.0f} s)")

    # Training history plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    for i, h in enumerate(histories):
        ax.semilogy(h.history['loss'],     color=colors[i], alpha=0.6, lw=1.0)
        ax.semilogy(h.history['val_loss'], color=colors[i], alpha=0.6, lw=1.0, ls='--')
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("MSE", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(plot_dir / "joint_nn_history.png", dpi=300)
    plt.close(fig)

    return best_model, best_val_loss


# ===========================================================================
# MODEL 2 — JOINT RBF
# ===========================================================================

def train_rbf_joint(X_train, Y_train, smoothing=1e-3):
    print(f"\n  [RBF-Joint] Thin-plate spline,  smoothing = {smoothing:.0e},  "
          f"{Y_train.shape[1]} outputs")
    t0  = time_module.time()
    rbf = RBFInterpolator(X_train, Y_train, kernel='thin_plate_spline',
                          smoothing=smoothing)
    print(f"              Fit time : {time_module.time() - t0:.2f} s")
    return rbf


# ===========================================================================
# MODEL 3 — JOINT GPR
# ===========================================================================

def train_gpr_joint(X_train, Y_train):
    print(f"\n  [GPR-Joint] Matern-5/2,  {Y_train.shape[1]} independent GPs "
          f"(parallelised)")
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)))
    gpr   = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                     normalize_y=True)
    model = MultiOutputRegressor(gpr, n_jobs=-1)
    t0    = time_module.time()
    model.fit(X_train, Y_train)
    print(f"              Fit time : {time_module.time() - t0:.1f} s")
    return model


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_final_time(fields_pred, fields_true, rel_errs,
                    case_list, n_times, x_mm,
                    field_name, split_label, model_name, plot_dir):
    """One figure per case — HF vs ROM at the final time snapshot."""
    n_cases = len(case_list)
    last_ti = n_times - 1
    for ci, cname in enumerate(case_list):
        snap_idx = last_ti * n_cases + ci
        ylabel = ("Stanton Number $St$ [-]" if field_name == "stanton"
                  else r"Skin Friction $C_f$ [-]")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_mm, fields_true[snap_idx], 'r-',  lw=2.0, label="High-fidelity (FOM)")
        ax.plot(x_mm, fields_pred[snap_idx], 'b--', lw=1.8, label=f"POD-{model_name}")
        ax.set_xlabel("x [mm]", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.4)
        fig.tight_layout()
        fname = plot_dir / f"{field_name}_{model_name}_{cname}_{split_label.lower()}_final.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"    Saved: {fname.name}")


def plot_model_comparison(results, case_list, split_label, n_times,
                          x_mm, field_name, plot_dir):
    """Overlay all model predictions for each case at the final time."""
    n_cases = len(case_list)
    last_ti = n_times - 1
    colors  = {'NN': 'steelblue', 'RBF': 'darkorange', 'GPR': 'forestgreen'}
    styles  = {'NN': '--',        'RBF': '-.',           'GPR': ':'}

    for ci, cname in enumerate(case_list):
        snap_idx = last_ti * n_cases + ci
        ylabel = ("Stanton Number $St$ [-]" if field_name == "stanton"
                  else r"Skin Friction $C_f$ [-]")
        fig, ax = plt.subplots(figsize=(9, 5))

        _, fields_true, _ = next(iter(results.values()))
        ax.plot(x_mm, fields_true[snap_idx], 'r-', lw=2.2, label="FOM")

        for mname, (fields_pred, _, rel_errs) in results.items():
            ax.plot(x_mm, fields_pred[snap_idx],
                    color=colors.get(mname, 'k'),
                    linestyle=styles.get(mname, '--'), lw=1.8,
                    label=f"POD-{mname}  ({rel_errs[snap_idx]*100:.2f}%)")

        ax.set_xlabel("x [mm]", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.4)
        fig.tight_layout()
        fname = plot_dir / f"{field_name}_all_models_{cname}_{split_label.lower()}.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"    Saved: {fname.name}")


def plot_error_bar_chart(all_errors, field_name, split_label, plot_dir):
    model_names = list(all_errors.keys())
    means = [all_errors[m] * 100 for m in model_names]
    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, means, color=['steelblue', 'darkorange', 'forestgreen'],
                  alpha=0.85, edgecolor='k', width=0.5)
    ax.bar_label(bars, fmt='%.2f%%', fontsize=14, padding=4)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=16)
    ax.set_ylabel("Mean Relative Error [%]", fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    fname = plot_dir / f"{field_name}_model_comparison_{split_label.lower()}.png"
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"    Saved: {fname.name}")


# ===========================================================================
# LOGGER
# ===========================================================================

class Logger:
    """Tee stdout to log file. stderr is left alone to avoid matplotlib conflicts."""
    def __init__(self, path):
        self.logfile      = open(path, 'w', buffering=1)
        self._stdout_orig = sys.stdout

    def write(self, msg):
        self._stdout_orig.write(msg)
        self.logfile.write(msg)

    def flush(self):
        self._stdout_orig.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


# ===========================================================================
# MAIN
# ===========================================================================

def train_and_evaluate():
    log_path = LOG_DIR / "training_log_joint.txt"
    logger   = Logger(log_path)
    sys.stdout = logger
    try:
        _run()
    finally:
        sys.stdout = logger._stdout_orig
        logger.close()
        print(f"  Log saved: {log_path}")


def _run():
    print(f"\n{'#'*65}")
    print(f"  POD-ML-ROM  |  JOINT training (St + Cf)")
    print(f"  Started: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*65}")

    # -----------------------------------------------------------------------
    # Load POD results for BOTH fields
    # -----------------------------------------------------------------------
    pod = {}
    for field in FIELDS:
        pod[field] = {
            "Phi":         np.load(POD_DIR / f"{field}_basis.npy"),
            "S_mean":      np.load(POD_DIR / f"{field}_mean.npy"),
            "alpha_train": np.load(POD_DIR / f"{field}_alpha_train.npy").T,
            "alpha_val":   np.load(POD_DIR / f"{field}_alpha_val.npy").T,
            "alpha_test":  np.load(POD_DIR / f"{field}_alpha_test.npy").T,
        }

    TRAIN_CASES = list(np.load(POD_DIR / "train_cases.npy"))
    VAL_CASES   = list(np.load(POD_DIR / "val_cases.npy"))
    TEST_CASES  = list(np.load(POD_DIR / "test_cases.npy"))
    TIME_VALUES = list(np.load(POD_DIR / "time_values.npy"))
    n_times     = len(TIME_VALUES)

    L_st = pod["stanton"]["alpha_train"].shape[1]
    L_cf = pod["cf"]["alpha_train"].shape[1]
    L_joint = L_st + L_cf

    print(f"\n  POD modes  : St = {L_st},  Cf = {L_cf},  joint output = {L_joint}")
    print(f"  Snapshots  : train={pod['stanton']['alpha_train'].shape[0]}, "
          f"val={pod['stanton']['alpha_val'].shape[0]}, "
          f"test={pod['stanton']['alpha_test'].shape[0]}")
    print(f"  Time steps : {n_times}  ({TIME_VALUES[0]:.2e} … {TIME_VALUES[-1]:.2e} s)")

    # -----------------------------------------------------------------------
    # Build shared physics-feature matrix X (same for both fields)
    # -----------------------------------------------------------------------
    P_train_raw = build_param_matrix(TRAIN_CASES, TIME_VALUES)
    P_val_raw   = build_param_matrix(VAL_CASES,   TIME_VALUES)
    P_test_raw  = build_param_matrix(TEST_CASES,  TIME_VALUES)

    F_train = physics_features(P_train_raw)
    F_val   = physics_features(P_val_raw)
    F_test  = physics_features(P_test_raw)

    in_sc   = MeanNormalizer().fit(F_train)
    X_train = in_sc.transform(F_train)
    X_val   = in_sc.transform(F_val)
    X_test  = in_sc.transform(F_test)

    print(f"\n  Physics features on training set:")
    for i, lbl in enumerate(["log10(Kn)", "Ma", "tau=t·U/L"]):
        print(f"    {lbl:12s}: [{F_train[:,i].min():.3f}, {F_train[:,i].max():.3f}]")

    # -----------------------------------------------------------------------
    # Scale each field's coefficients separately, then concatenate
    # Separate scalers ensure neither field dominates the training loss
    # -----------------------------------------------------------------------
    out_sc = {}
    Y_parts_train, Y_parts_val = [], []

    for field in FIELDS:
        sc = MeanNormalizer().fit(pod[field]["alpha_train"])
        out_sc[field] = sc
        Y_parts_train.append(sc.transform(pod[field]["alpha_train"]))
        Y_parts_val.append(sc.transform(pod[field]["alpha_val"]))

    Y_train = np.hstack(Y_parts_train)   # (N_tr, L_st + L_cf)
    Y_val   = np.hstack(Y_parts_val)     # (N_va, L_st + L_cf)

    # -----------------------------------------------------------------------
    # Load actual FOM fields for error computation and plotting
    # -----------------------------------------------------------------------
    print(f"\n  Loading FOM fields from .dat files...")
    fom = {}
    for field in FIELDS:
        fom[field] = {
            "val":  load_fom_fields(VAL_CASES,  TIME_VALUES, field),
            "test": load_fom_fields(TEST_CASES, TIME_VALUES, field),
        }

    x_coords = np.loadtxt(DATA_DIR / "wall_x.txt")
    x_mm     = x_coords * 1000
    plot_dir = Path("plots") / "pod_nn_results"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Train joint surrogates
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Training joint surrogate models  (output dim = {L_joint})")
    print(f"{'='*65}")

    # results[field][split][model] = (fields_pred, fields_true, rel_errs)
    results = {f: {"val": {}, "test": {}} for f in FIELDS}

    def collect(model_predict, model_name):
        """Predict, split, reconstruct and store results for both fields."""
        for X, split in [(X_val, "val"), (X_test, "test")]:
            Y_pred = model_predict(X)           # (N, L_joint)  — scaled space
            alpha_st, alpha_cf = split_and_decode(
                Y_pred, L_st, out_sc["stanton"], out_sc["cf"]
            )
            for field, alpha_pred in [("stanton", alpha_st), ("cf", alpha_cf)]:
                p = pod[field]
                results[field][split][model_name] = reconstruct_fields(
                    alpha_pred, fom[field][split], p["Phi"], p["S_mean"]
                )

    # --- 1. Joint NN ---
    nn_model, _ = train_nn_joint(X_train, Y_train, X_val, Y_val, L_joint, plot_dir)
    collect(lambda X: nn_model.predict(X, verbose=0), "NN")
    nn_model.save(MODEL_DIR / "joint_nn_model.keras")
    np.save(MODEL_DIR / "joint_in_mean.npy",  in_sc.mean_)
    np.save(MODEL_DIR / "joint_in_scale.npy", in_sc.scale_)
    for field in FIELDS:
        np.save(MODEL_DIR / f"joint_{field}_out_mean.npy",  out_sc[field].mean_)
        np.save(MODEL_DIR / f"joint_{field}_out_scale.npy", out_sc[field].scale_)

    # --- 2. Joint RBF ---
    rbf_model = train_rbf_joint(X_train, Y_train, smoothing=1e-3)
    collect(lambda X: rbf_model(X), "RBF")

    # --- 3. Joint GPR ---
    gpr_model = train_gpr_joint(X_train, Y_train)
    collect(lambda X: gpr_model.predict(X), "GPR")

    # -----------------------------------------------------------------------
    # Error summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Reconstruction errors  (||u_ROM − u_FOM|| / ||u_FOM||)")
    print(f"{'='*65}")

    for field in FIELDS:
        print(f"\n  {field.upper()}:")
        for split_label, split_key in [("Validation", "val"), ("Test", "test")]:
            print(f"    {split_label}:")
            for mname, (_, _, rel_errs) in results[field][split_key].items():
                error_summary(mname, rel_errs)

    # -----------------------------------------------------------------------
    # Plots — final time step only
    # -----------------------------------------------------------------------
    for field in FIELDS:
        for split_label, split_key, case_list in [
            ("Validation", "val",  VAL_CASES),
            ("Test",       "test", TEST_CASES),
        ]:
            for mname, (fp, ft, re) in results[field][split_key].items():
                plot_final_time(fp, ft, re, case_list, n_times, x_mm,
                                field, split_label, mname, plot_dir)
            plot_model_comparison(results[field][split_key], case_list,
                                  split_label, n_times, x_mm, field, plot_dir)
            val_means = {m: r[2].mean() for m, r in results[field][split_key].items()}
            plot_error_bar_chart(val_means, field, split_label, plot_dir)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"  {'Field':8s}  {'Model':6s}  {'Val mean%':>10s}  {'Val max%':>9s}  "
          f"{'Test mean%':>11s}  {'Test max%':>10s}")
    print(f"  {'-'*60}")
    for field in FIELDS:
        for mname in results[field]["val"]:
            vm = results[field]["val"][mname][2]
            tm = results[field]["test"][mname][2]
            print(f"  {field.upper():8s}  {mname:6s}  "
                  f"{vm.mean()*100:>10.3f}  {vm.max()*100:>9.3f}  "
                  f"{tm.mean()*100:>11.3f}  {tm.max()*100:>10.3f}")
        print()

    print(f"  Plots  → {plot_dir}/")
    print(f"  Models → {MODEL_DIR}/")
    print(f"  Log    → {LOG_DIR}/training_log_joint.txt")
    print(f"  Finished: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    train_and_evaluate()
