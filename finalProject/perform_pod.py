import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DATA_DIR = Path("data_extraction")
POD_DIR  = Path("pod_results")
POD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset split
# Columns in the .dat files: case1..case17, valcase, test1
# ---------------------------------------------------------------------------
ALL_CASES   = [f"case{i}" for i in range(1, 18)] + ["valcase", "test1"]
VAL_CASES   = ["case4", "case16"]          # held-out for ANN validation
TEST_CASES  = ["valcase", "test1"]         # held-out for final testing
TRAIN_CASES = [c for c in ALL_CASES if c not in VAL_CASES + TEST_CASES]

# --- Time step selection ---
# Change TIME_STEPS to study different combinations.
# Indices 1-10 map to the physical times below; everything else propagates automatically.
TIME_STEPS = list(range(3, 11))

# Full index → physical time [s] mapping (do not edit)
_ALL_TIMES = {
    1: 5e-05, 2: 1e-04, 3: 1.5e-04, 4: 2e-04,  5: 2.5e-04,
    6: 3e-04, 7: 3.5e-04, 8: 4e-04, 9: 4.5e-04, 10: 5e-04,
}
TIME_VALUES = [_ALL_TIMES[i] for i in TIME_STEPS]  # derived automatically

# POD energy threshold
ENERGY_THRESHOLD = 0.999

# Fields to analyse
QUANTITIES = {
    "stanton": {"dir": "St",  "prefix": "stanton"},
    "cf":      {"dir": "Cf",  "prefix": "cf"},
}


def get_col_indices(case_list):
    """Return column indices for case_list within ALL_CASES."""
    return [ALL_CASES.index(c) for c in case_list]


def perform_pod_analysis(field_name, info, x_coords):
    print(f"\n{'='*60}")
    print(f"  POD Analysis — {field_name.upper()}")
    print(f"{'='*60}")
    print(f"  Training  ({len(TRAIN_CASES):2d} cases): {TRAIN_CASES}")
    print(f"  Validation ({len(VAL_CASES):2d} cases): {VAL_CASES}")
    print(f"  Test       ({len(TEST_CASES):2d} cases): {TEST_CASES}")
    print(f"  Time steps: {TIME_STEPS}  ({len(TIME_STEPS)} selected)")

    tr_idx  = get_col_indices(TRAIN_CASES)
    val_idx = get_col_indices(VAL_CASES)
    te_idx  = get_col_indices(TEST_CASES)

    # ------------------------------------------------------------------
    # Load all time steps and separate by split
    # Column ordering after hstack: (t1,c1), (t1,c2), ..., (tN,cM)
    # ------------------------------------------------------------------
    S_train_cols, S_val_cols, S_test_cols = [], [], []
    loaded_times        = []   # step indices actually found on disk
    loaded_time_values  = []   # corresponding physical times [s]

    for t in TIME_STEPS:
        fpath = DATA_DIR / info["dir"] / f"{info['prefix']}_t{t}.dat"
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found — skipping t{t}.")
            continue

        data = np.loadtxt(fpath)          # (399, 18)  — header line skipped by loadtxt
        S_train_cols.append(data[:, tr_idx])    # (399, 15)
        S_val_cols.append(data[:, val_idx])     # (399, 2)
        S_test_cols.append(data[:, te_idx])     # (399, 1)
        loaded_times.append(t)
        loaded_time_values.append(_ALL_TIMES[t])

    S_train = np.hstack(S_train_cols)  # (399, 15*10 = 150)
    S_val   = np.hstack(S_val_cols)   # (399,  2*10 =  20)
    S_test  = np.hstack(S_test_cols)  # (399,  1*10 =  10)

    print(f"\n  Snapshot matrix shapes:")
    print(f"    Training   : {S_train.shape}")
    print(f"    Validation : {S_val.shape}")
    print(f"    Test       : {S_test.shape}")

    # ------------------------------------------------------------------
    # Mean computed from training data only — proper ML practice
    # ------------------------------------------------------------------
    S_mean = S_train.mean(axis=1, keepdims=True)   # (399, 1)

    S_train_c = S_train - S_mean
    S_val_c   = S_val   - S_mean
    S_test_c  = S_test  - S_mean

    # ------------------------------------------------------------------
    # SVD on training fluctuations
    # ------------------------------------------------------------------
    U_svd, Sigma, _ = np.linalg.svd(S_train_c, full_matrices=False)

    eigenvalues   = Sigma ** 2
    cumul_energy  = np.cumsum(eigenvalues) / eigenvalues.sum()
    mask = cumul_energy >= ENERGY_THRESHOLD
    L = int(np.argmax(mask)) + 1 if mask.any() else len(Sigma)
    print(f"\n  Modes for {ENERGY_THRESHOLD*100:.1f}% energy threshold: L = {L}")
    print(f"  Captured energy at L = {L}: {cumul_energy[L-1]*100:.5f}%")

    Phi = U_svd[:, :L]    # (399, L) — reduced basis

    # ------------------------------------------------------------------
    # Project each split onto the basis
    # ------------------------------------------------------------------
    alpha_train = Phi.T @ S_train_c   # (L, 150)
    alpha_val   = Phi.T @ S_val_c    # (L,  20)
    alpha_test  = Phi.T @ S_test_c   # (L,  10)

    # Projection errors on training set
    S_train_recon = Phi @ alpha_train + S_mean
    proj_errs = np.linalg.norm(S_train - S_train_recon, axis=0) / (
                np.linalg.norm(S_train, axis=0) + 1e-20)
    print(f"  Training projection error: "
          f"mean = {proj_errs.mean()*100:.4f}%,  "
          f"max  = {proj_errs.max()*100:.4f}%")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    np.save(POD_DIR / f"{field_name}_basis.npy",       Phi)
    np.save(POD_DIR / f"{field_name}_mean.npy",        S_mean)
    np.save(POD_DIR / f"{field_name}_alpha_train.npy", alpha_train)
    np.save(POD_DIR / f"{field_name}_alpha_val.npy",   alpha_val)
    np.save(POD_DIR / f"{field_name}_alpha_test.npy",  alpha_test)

    np.save(POD_DIR / "train_cases.npy",  np.array(TRAIN_CASES))
    np.save(POD_DIR / "val_cases.npy",    np.array(VAL_CASES))
    np.save(POD_DIR / "test_cases.npy",   np.array(TEST_CASES))
    np.save(POD_DIR / "time_values.npy",  np.array(loaded_time_values))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    x_mm = x_coords * 1000

    # 1. Energy capture
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumul_energy) + 1), cumul_energy * 100, 'bo-', markersize=5)
    ax.axhline(ENERGY_THRESHOLD * 100, color='r', linestyle='--',
               label=f'{ENERGY_THRESHOLD*100:.1f}% threshold')
    ax.axvline(L, color='g', linestyle=':', label=f'$L = {L}$')
    ax.set_xlabel("Number of Modes", fontsize=18)
    ax.set_ylabel("Cumulative Energy [%]", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(POD_DIR / f"{field_name}_energy.png", dpi=300)
    plt.close(fig)

    # 2. Singular value decay
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(range(1, len(Sigma) + 1), Sigma, 'b.-', markersize=6)
    ax.axvline(L, color='r', linestyle='--', label=f'$L = {L}$')
    ax.set_xlabel("Mode Index", fontsize=18)
    ax.set_ylabel("Singular Value", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(POD_DIR / f"{field_name}_singular_values.png", dpi=300)
    plt.close(fig)

    # 3. First 5 basis functions
    n_plot  = min(5, L)
    styles  = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'v', 'd']
    colors  = plt.cm.viridis(np.linspace(0, 0.8, n_plot))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(n_plot):
        ax.plot(x_mm, Phi[:, i], color=colors[i], linestyle=styles[i],
                marker=markers[i], markevery=30, markersize=5, linewidth=1.6,
                label=f"Mode {i+1}")
    ax.set_xlabel("x [mm]", fontsize=18)
    ax.set_ylabel("Mode Amplitude", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(POD_DIR / f"{field_name}_modes.png", dpi=300)
    plt.close(fig)

    return L, cumul_energy[L - 1]


if __name__ == "__main__":
    x_coords = np.loadtxt(DATA_DIR / "wall_x.txt")
    print(f"Spatial grid: {x_coords.shape[0]} points, "
          f"x = [{x_coords[0]*1000:.3f}, {x_coords[-1]*1000:.3f}] mm")

    results = {}
    for q_name, q_info in QUANTITIES.items():
        L, energy = perform_pod_analysis(q_name, q_info, x_coords)
        results[q_name] = {"L": L, "energy_pct": energy * 100}

    print(f"\n{'='*60}")
    print("  POD Analysis Complete")
    print(f"{'='*60}")
    for q, r in results.items():
        print(f"  {q.capitalize():10s}: L = {r['L']:3d} modes  "
              f"(energy = {r['energy_pct']:.5f}%)")
    print(f"\nOutputs saved to: {POD_DIR}/")
    print("Next step: run train_and_evaluate.py")
