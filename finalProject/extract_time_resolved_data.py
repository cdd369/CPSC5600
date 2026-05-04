import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy.interpolate import interp1d

# Physical Constants
R_ARGON = 208.14
CP_ARGON = 2.5 * R_ARGON
T_WALL = 292.0
H_WALL = CP_ARGON * T_WALL

# Case Parameters (Freestream conditions from casesList.md)
CASES = {
    "case1":  {"T": 80,  "p": 2.020, "U": 1600, "regime": "slip"},
    "case2":  {"T": 120, "p": 2.200, "U": 1800, "regime": "slip"},
    "case3":  {"T": 160, "p": 2.260, "U": 2000, "regime": "slip"},
    "case4":  {"T": 220, "p": 2.370, "U": 2200, "regime": "slip"},
    "case5":  {"T": 300, "p": 2.710, "U": 2500, "regime": "slip"},
    "case6":  {"T": 400, "p": 3.140, "U": 2800, "regime": "slip"},
    "case7":  {"T": 500, "p": 3.480, "U": 3200, "regime": "slip"},
    "case8":  {"T": 600, "p": 3.770, "U": 3600, "regime": "slip"},
    "case9":  {"T": 700, "p": 4.010, "U": 4000, "regime": "slip"},
    "case10": {"T": 800, "p": 4.240, "U": 4500, "regime": "slip"},
    "case11": {"T": 80,  "p": 0.202, "U": 1500, "regime": "trans"},
    "case12": {"T": 150, "p": 0.289, "U": 1900, "regime": "trans"},
    "case13": {"T": 200, "p": 0.296, "U": 2300, "regime": "trans"},
    "case14": {"T": 280, "p": 0.318, "U": 2700, "regime": "trans"},
    "case15": {"T": 380, "p": 0.360, "U": 3100, "regime": "trans"},
    "case16": {"T": 480, "p": 0.392, "U": 3500, "regime": "trans"},
    "case17": {"T": 580, "p": 0.418, "U": 3900, "regime": "trans"},
    "case18": {"T": 680, "p": 0.440, "U": 4300, "regime": "trans"},
    "case19": {"T": 780, "p": 0.459, "U": 4700, "regime": "trans"},
    "case20": {"T": 870, "p": 0.466, "U": 4900, "regime": "trans"},
    "valcase": {"T": 64.5, "p": 3.73,  "U": 1893.7, "regime": "validation"},
    # T-S slip test case (Kn=0.050, between case5 and case6)
    "test1":  {"T": 350,  "p": 2.929, "U": 2650,   "regime": "slip",
               "path": Path("../cases/testcases/test1")},
}

VAL_DIR   = Path("../cases/validation_case")
SLIP_DIR  = Path("../cases/slip_regime")
TRANS_DIR = Path("../cases/transitional_regime")
OUTPUT_DIR = Path("data_extraction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for sub in ["heatflux", "shearStressX", "St", "Cf"]:
    (OUTPUT_DIR / sub).mkdir(exist_ok=True)

def parse_openfoam_list(path, value_kind="scalar", patch_name="bottom"):
    """Surgical extraction of OpenFOAM list data."""
    if not os.path.exists(path):
        return None
        
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Locate patch
    start_idx = -1
    for i, line in enumerate(lines):
        if f"    {patch_name}" == line.strip("\n") or f"	{patch_name}" == line.strip("\n") or patch_name == line.strip():
            # Check if it's the start of a boundary field entry
            if i+1 < len(lines) and "{" in lines[i+1]:
                start_idx = i
                break
    
    if start_idx == -1:
        # Try a more relaxed search if exact match fails
        for i, line in enumerate(lines):
            if patch_name in line and "{" in lines[min(i+1, len(lines)-1)]:
                start_idx = i
                break
                
    if start_idx == -1:
        return None
        
    # Find the count and the opening parenthesis
    data_start = -1
    count = -1
    for i in range(start_idx, len(lines)):
        if "(" in lines[i]:
            # The count is usually on the line before "(" or on the same line if it's nonuniform List<scalar> 399 (
            line = lines[i].strip()
            match = re.search(r'>\s+(\d+)\s*\(', line)
            if match:
                count = int(match.group(1))
                data_start = i + 1
            else:
                try:
                    count = int(lines[i-1].strip())
                    data_start = i + 1
                except ValueError:
                    # Maybe it's further up?
                    pass
            if data_start != -1:
                break
    
    if data_start == -1:
        return None
    
    values = []
    for i in range(data_start, data_start + count):
        line = lines[i].strip()
        if value_kind == "scalar":
            values.append(float(line))
        else:
            # Extract x-component of the vector (tau_x)
            vec = line.strip("()").split()
            values.append(float(vec[0]))
            
    return np.array(values)

def load_slip_coords():
    path = SLIP_DIR / "wall_x.txt"
    with open(path, 'r') as f:
        lines = f.readlines()
    count = int(lines[1].strip())
    coords = [float(l.strip()) for l in lines[3:3+count]]
    return np.array(coords)

def load_trans_coords():
    path = TRANS_DIR / "wall_x.txt"
    with open(path, 'r') as f:
        lines = f.readlines()
    # Skip "X" header
    coords = [float(l.strip()) for l in lines[1:] if l.strip() and l.strip() != ")"]
    return np.array(coords)

def process_data():
    x_slip = load_slip_coords()
    x_trans = load_trans_coords()
    
    # Target grid: Use slip grid as baseline
    x_target = x_slip
    np.savetxt(OUTPUT_DIR / "wall_x.txt", x_target, header="X_target")
    
    # Times to process (ignore any extra ones like 0.001)
    # Based on casesList.md, we want 5 snapshots, but we have 10 in the existing files.
    # We'll use the 10 common ones: 5e-05, 0.0001, 0.00015, ..., 0.0005
    times = ["5e-05", "0.0001", "0.00015", "0.0002", "0.00025", "0.0003", "0.00035", "0.0004", "0.00045", "0.0005"]
    
    print(f"Targeting {len(times)} time snapshots.")
    
    # Define ordered case names to ensure consistent output column ordering
    ordered_case_names = [f"case{i}" for i in range(1, 21)] + ["valcase", "test1"]
    
    for t_idx, t_val in enumerate(times, 1):
        hf_list = []
        ws_list = []
        st_list = []
        cf_list = []
        available_cases = []
        
        for cname in ordered_case_names:
            if cname not in CASES:
                continue
                
            params = CASES[cname]
            regime = params["regime"]
            
            if regime == "trans":
                patch  = "plate"
                x_orig = x_trans
                c_path = TRANS_DIR / cname
            elif regime == "slip":
                patch  = "bottom"
                x_orig = x_slip
                # Use explicit path override if present (e.g. test cases)
                c_path = params.get("path", SLIP_DIR / cname)
            elif regime == "validation":
                patch  = "bottom"
                x_orig = x_slip
                c_path = VAL_DIR
            
            hf_path = c_path / t_val / "wallHeatFlux"
            ws_path = c_path / t_val / "wallShearStress"
            
            if not hf_path.exists() or not ws_path.exists():
                continue
                
            available_cases.append(cname)
            
            T_inf, p_inf, U_inf = params["T"], params["p"], params["U"]
            rho_inf = p_inf / (R_ARGON * T_inf)
            q_inf = 0.5 * rho_inf * U_inf**2
            H0 = CP_ARGON * T_inf + 0.5 * U_inf**2
            dH = H0 - H_WALL
            
            # Load Raw
            qw_raw = parse_openfoam_list(hf_path, "scalar", patch)
            tau_x_raw = parse_openfoam_list(ws_path, "vector", patch)
            
            if qw_raw is None or tau_x_raw is None:
                print(f"  Warning: Could not parse {cname} at {t_val}")
                available_cases.pop()
                continue
                
            # Note: OpenFOAM wallHeatFlux is negative for heating, wallShearStress is negative for flow in +x
            qw_raw = -qw_raw
            tau_x_raw = -tau_x_raw
            
            # Interpolate to target grid
            f_qw = interp1d(x_orig, qw_raw, kind='cubic', fill_value="extrapolate")
            f_tx = interp1d(x_orig, tau_x_raw, kind='cubic', fill_value="extrapolate")
            
            qw = f_qw(x_target)
            tau_x = f_tx(x_target)
            
            # Non-dimensionalize
            St = qw / (rho_inf * U_inf * dH)
            Cf = tau_x / q_inf
            
            hf_list.append(qw)
            ws_list.append(tau_x)
            st_list.append(St)
            cf_list.append(Cf)
            
        if not available_cases:
            continue
            
        header = "   ".join(available_cases)
        np.savetxt(OUTPUT_DIR / "heatflux" / f"heatflux_t{t_idx}.dat", np.column_stack(hf_list), header=header, fmt="%.8e")
        np.savetxt(OUTPUT_DIR / "shearStressX" / f"shearStressX_t{t_idx}.dat", np.column_stack(ws_list), header=header, fmt="%.8e")
        np.savetxt(OUTPUT_DIR / "St" / f"stanton_t{t_idx}.dat", np.column_stack(st_list), header=header, fmt="%.8e")
        np.savetxt(OUTPUT_DIR / "Cf" / f"cf_t{t_idx}.dat", np.column_stack(cf_list), header=header, fmt="%.8e")
        
        print(f"Generated files for t{t_idx} ({t_val} s) with {len(available_cases)} cases.")

if __name__ == "__main__":
    process_data()
