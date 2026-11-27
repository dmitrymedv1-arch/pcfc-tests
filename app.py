import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_bvp
import streamlit as st

st.set_page_config(
    page_title="PCEC / PCFC Multiphysics Membrane Model",
    page_icon="⚡",
    layout="wide"
)

_trapz = np.trapezoid if hasattr(np, "trapezoid") else getattr(np, "trapz")

R_GAS = 8.314462618
F_CONST = 96485.33212
DH_W = -248110.0
DS_W = -55.48
LHV_H2 = 241800.0
V_MOLAR_STP = 22414.0

MATERIAL_PRESETS = {
    "BCZYYb7111 (Thatte 2024 / Restrepo 2026)": {
        "Vm": 5.13e-5,
        "dH_H2O": -93300.0,  "dS_H2O": -100.0,
        "dH_O2":  114880.0,  "dS_O2":  -45.89,
        "dH_trap": -90000.0, "dS_trap": -14.41,
        "D0_p": 1.02e-7,     "E_p": 42650.0,
        "D0_v": 1.73e-7,     "E_v": 59700.0,
        "D0_h": 2.41e-5,     "E_h": 8400.0,
        "default_cR": 0.20,
    },
    "BZY20 (Moranti 2026)": {
        "Vm": 4.57e-5,
        "dH_H2O": -93300.0,  "dS_H2O": -100.0,
        "dH_O2":  115310.0,  "dS_O2":  -45.89,
        "dH_trap": -90000.0, "dS_trap": -14.41,
        "D0_p": 5.18e-7,     "E_p": 60660.0,
        "D0_v": 2.03e-7,     "E_v": 85190.0,
        "D0_h": 1.38e-5,     "E_h": 7180.0,
        "default_cR": 0.20,
    },
    "BCZY652015 (Moranti 2026)": {
        "Vm": 4.57e-5,
        "dH_H2O": -123000.0, "dS_H2O": -130.95,
        "dH_O2":  132570.0,  "dS_O2":  -119.93,
        "dH_trap": -90000.0, "dS_trap": -14.41,
        "D0_p": 1.63e-7,     "E_p": 47190.0,
        "D0_v": 1.36e-7,     "E_v": 68060.0,
        "D0_h": 2.50e-3,     "E_h": 9730.0,
        "default_cR": 0.15,
    },
    "BZY10 (Restrepo 2026)": {
        "Vm": 4.57e-5,
        "dH_H2O": -79400.0,  "dS_H2O": -88.90,
        "dH_O2":  119000.0,  "dS_O2":  -45.00,
        "dH_trap": 0.0,      "dS_trap": -200.0,
        "D0_p": 1.32e-7,     "E_p": 47000.0,
        "D0_v": 8.90e-8,     "E_v": 75000.0,
        "D0_h": 1.15e-4,     "E_h": 12000.0,
        "default_cR": 0.10,
    },
}


# =============================================================================
# 2. PHYSICS ENGINE
# =============================================================================
def get_material_props(T_K: float, mat: dict):
    K_W = np.exp(DS_W / R_GAS - DH_W / (R_GAS * T_K))
    K_H2O = np.exp(mat["dS_H2O"] / R_GAS - mat["dH_H2O"] / (R_GAS * T_K))
    K_O2 = np.exp(mat["dS_O2"] / R_GAS - mat["dH_O2"] / (R_GAS * T_K))
    dH_trap = mat.get("dH_trap", 0.0)
    dS_trap = mat.get("dS_trap", -200.0)
    K_trap = np.exp(dS_trap / R_GAS - dH_trap / (R_GAS * T_K))
    D_p = mat["D0_p"] * np.exp(-mat["E_p"] / (R_GAS * T_K))
    D_v = mat["D0_v"] * np.exp(-mat["E_v"] / (R_GAS * T_K))
    D_h = mat["D0_h"] * np.exp(-mat["E_h"] / (R_GAS * T_K))
    return K_W, K_H2O, K_O2, K_trap, D_p, D_v, D_h


def solve_boundary_defects(
    p_H2O: float, p_O2: float, c_R: float, K_H2O: float, K_O2: float,
    use_trapping: bool = False, K_trap: float = 0.0
):
    p_H2O = max(p_H2O, 1e-18)
    p_O2 = max(p_O2, 1e-35)

    alpha = np.sqrt((K_O2 * np.sqrt(p_O2)) / (K_H2O * p_H2O))
    denom = 1.0 + alpha

    if use_trapping and K_trap > 1e-12:
        A_q = denom * K_trap * alpha

        def _p_from_v(v):
            B_q = denom + 2.0 * v * K_trap * alpha
            C_q = 2.0 * v - c_R
            return (-B_q + np.sqrt(B_q * B_q - 4.0 * A_q * C_q)) / (2.0 * A_q)

        def residual(v):
            p = _p_from_v(v)
            o = 3.0 - c_R + v
            return p * p - K_H2O * p_H2O * v * o

        v_max = 0.5 * c_R
        v_sol = brentq(residual, 1e-18 * v_max, v_max * (1.0 - 1e-15), xtol=1e-14, rtol=1e-13)
        p_sol = _p_from_v(v_sol)
        h_sol = alpha * p_sol
        o_sol = 3.0 - c_R + v_sol
        return p_sol, v_sol, h_sol, o_sol

    def residual(v):
        p = (c_R - 2.0 * v) / denom
        o = 3.0 - c_R + v
        return p * p - K_H2O * p_H2O * v * o

    v_max = 0.5 * c_R
    v_sol = brentq(residual, 1e-18 * v_max, v_max * (1.0 - 1e-15), xtol=1e-14, rtol=1e-13)
    p_sol = (c_R - 2.0 * v_sol) / denom
    h_sol = alpha * p_sol
    o_sol = 3.0 - c_R + v_sol
    return p_sol, v_sol, h_sol, o_sol


def bernoulli_flux(u: float, c_f: float, c_o: float) -> float:
    if abs(u) < 1e-6:
        return (c_o - c_f) + 0.5 * u * (c_o + c_f) + (u * u / 12.0) * (c_o - c_f)
    elif u > 0:
        eu = np.exp(-u)
        return u * (c_o - c_f * eu) / (1.0 - eu)
    else:
        eu = np.exp(u)
        return u * (c_o * eu - c_f) / (eu - 1.0)


def profile_exp(x_norm: np.ndarray, u: float, c_f: float, c_o: float) -> np.ndarray:
    if abs(u) < 1e-6:
        return c_f + (c_o - c_f) * x_norm
    elif u > 0:
        return c_f + (c_o - c_f) * (1.0 - np.exp(-u * x_norm)) / (1.0 - np.exp(-u))
    else:
        return c_o + (c_f - c_o) * (1.0 - np.exp(u * (1.0 - x_norm))) / (1.0 - np.exp(u))


def compute_gas_fluxes(J_p, J_v, p_H2_f, p_H2O_f, p_O2_f):
    p_H2O_f = max(p_H2O_f, 1e-12)
    p_O2_f = max(p_O2_f, 1e-35)
    p_H2_f = max(p_H2_f, 1e-12)

    term_p = 2.0 + 2.0 * (p_H2_f / p_H2O_f) + 0.5 * (p_H2_f / p_O2_f)
    term_v = 1.0 + 4.0 * (p_O2_f / p_H2O_f) + 4.0 * (p_O2_f / p_H2_f)

    J_H2O = J_p / term_p - J_v / term_v
    J_H2 = 0.5 * J_p - J_H2O
    J_O2 = -0.5 * J_v - 0.5 * J_H2O
    return J_H2, J_H2O, J_O2


def compute_dgm_overpotentials(T_K: float, J_H2: float, p_H2_f: float, p_H2O_o: float, p_O2_o: float):
    L_ed_f = 600e-6
    L_ed_o = 60e-6
    phi_tau = 0.4 / 3.0
    r_p = 0.5e-6
    p_tot = 101325.0

    D_H2_H2O = phi_tau * 9.15e-5 * (T_K / 307.1) ** 1.75
    D_O2_H2O = phi_tau * 2.82e-5 * (T_K / 308.1) ** 1.75
    D_Kn_H2 = phi_tau * (2.0 / 3.0) * r_p * np.sqrt(8.0 * R_GAS * T_K / (np.pi * 2.016e-3))
    D_Kn_H2O = phi_tau * (2.0 / 3.0) * r_p * np.sqrt(8.0 * R_GAS * T_K / (np.pi * 18.015e-3))
    D_Kn_O2 = phi_tau * (2.0 / 3.0) * r_p * np.sqrt(8.0 * R_GAS * T_K / (np.pi * 31.999e-3))

    D_eff_H2 = 1.0 / (1.0 / D_H2_H2O + 1.0 / D_Kn_H2)
    D_eff_H2O = 1.0 / (1.0 / D_O2_H2O + 1.0 / D_Kn_H2O)
    D_eff_O2 = 1.0 / (1.0 / D_O2_H2O + 1.0 / D_Kn_O2)

    p_H2_tpb = np.clip(p_H2_f - J_H2 * R_GAS * T_K * L_ed_f / (p_tot * D_eff_H2), 1e-4, 0.999)
    p_H2O_o_tpb = np.clip(p_H2O_o + J_H2 * R_GAS * T_K * L_ed_o / (p_tot * D_eff_H2O), 1e-4, 0.999)
    p_O2_o_tpb = np.clip(p_O2_o - 0.5 * J_H2 * R_GAS * T_K * L_ed_o / (p_tot * D_eff_O2), 1e-4, 0.999)

    eta_conc_H = -(R_GAS * T_K / (2.0 * F_CONST)) * np.log(p_H2_tpb / max(p_H2_f, 1e-6))
    eta_conc_O = -(R_GAS * T_K / (2.0 * F_CONST)) * np.log(
        np.sqrt(p_O2_o_tpb / max(p_O2_o, 1e-6)) * (max(p_H2O_o, 1e-6) / p_H2O_o_tpb)
    )
    return float(eta_conc_H), float(eta_conc_O)


def _solve_bvp_with_homotopy(
    x_norm, pf, vf, hf, po, vo, ho, c_R,
    D_p, D_v, D_h, L_m, Vm, T_K, i_A_m2,
    cp_init, cv_init, ch_init, Jp_init, Jv_init, Jh_init, u1_init
):
    Dp_nd = 1.0
    Dv_nd = D_v / D_p
    Dh_nd = D_h / D_p
    eps_reg = max(100.0 * Vm * R_GAS * T_K / (F_CONST ** 2 * L_m ** 2), 1e-4)
    i_target_nd = i_A_m2 * L_m * Vm / (D_p * F_CONST)

    def make_ode():
        def ode_rhs(x, y):
            cp, cv, ch, Jp_n, Jv_n, Jh_n, Phi_n, dPhi_n = y
            dcp = -Jp_n / Dp_nd - 1.0 * cp * dPhi_n
            dcv = -Jv_n / Dv_nd - 2.0 * cv * dPhi_n
            dch = -Jh_n / Dh_nd - 1.0 * ch * dPhi_n
            zeros = np.zeros_like(x)
            d2Phi = -(cp + 2.0 * cv + ch - c_R) / eps_reg
            return np.vstack([dcp, dcv, dch, zeros, zeros, zeros, dPhi_n, d2Phi])
        return ode_rhs

    def make_bc(i_nd_val):
        def bc_func(ya, yb):
            return np.array([
                ya[0] - pf, ya[1] - vf, ya[2] - hf,
                yb[0] - po, yb[1] - vo, yb[2] - ho,
                ya[6] - 0.0,
                (yb[3] + 2.0 * yb[4] + yb[5]) - i_nd_val,
            ])
        return bc_func

    y_guess = np.vstack([
        cp_init, cv_init, ch_init,
        np.full_like(x_norm, Jp_init * L_m * Vm / D_p),
        np.full_like(x_norm, Jv_init * L_m * Vm / D_p),
        np.full_like(x_norm, Jh_init * L_m * Vm / D_p),
        u1_init * x_norm,
        np.full_like(x_norm, u1_init),
    ])

    ode_rhs = make_ode()
    max_nodes = 2000

    sol = solve_bvp(ode_rhs, make_bc(i_target_nd), x_norm, y_guess, tol=1e-4, max_nodes=max_nodes)
    if sol.success:
        return sol, True, sol.x.size >= int(0.9 * max_nodes)

    x_curr = x_norm
    y_curr = y_guess
    for frac in [0.2, 0.5, 0.8, 1.0]:
        sol = solve_bvp(ode_rhs, make_bc(i_target_nd * frac), x_curr, y_curr, tol=1e-4, max_nodes=max_nodes)
        if not sol.success:
            return sol, False, False
        x_curr = sol.x
        y_curr = sol.y

    return sol, True, sol.x.size >= int(0.9 * max_nodes)


def _solve_single_point_core(
    i_A_cm2: float,
    T_C: float,
    L_um: float,
    Rp_H: float,
    Rp_O: float,
    p_H2_f: float,
    p_H2O_f: float,
    p_H2O_o: float,
    c_R: float,
    mat: dict,
    solver_mode: str = "fast",
    use_arcsinh_shift: bool = True,
    n_grid: int = 80,
    use_trapping: bool = False,
    use_dgm: bool = False,
    beta_p: float = 0.035,
    beta_v: float = -0.015,
):
    T_K = T_C + 273.15
    L_m = L_um * 1e-6
    Vm = mat["Vm"]
    i_A_m2 = i_A_cm2 * 1e4

    K_W, K_H2O, K_O2, K_trap, D_p, D_v, D_h = get_material_props(T_K, mat)

    p_O2_o = max(0.21 * (1.0 - p_H2O_o), 1e-12)
    p_O2_f = (p_H2O_f / (K_W * max(p_H2_f, 1e-12))) ** 2

    pf_eq, vf_eq, hf_eq, of_eq = solve_boundary_defects(p_H2O_f, p_O2_f, c_R, K_H2O, K_O2, use_trapping, K_trap)
    po_eq, vo_eq, ho_eq, oo_eq = solve_boundary_defects(p_H2O_o, p_O2_o, c_R, K_H2O, K_O2, use_trapping, K_trap)

    E_eq = (R_GAS * T_K / F_CONST) * np.log((ho_eq / oo_eq) / (hf_eq / of_eq))

    eta_H = (R_GAS * T_K / F_CONST) * np.arcsinh((i_A_cm2 * Rp_H * F_CONST) / (R_GAS * T_K)) if Rp_H > 0 else 0.0
    eta_O = (R_GAS * T_K / F_CONST) * np.arcsinh((i_A_cm2 * Rp_O * F_CONST) / (R_GAS * T_K)) if Rp_O > 0 else 0.0

    if use_arcsinh_shift and (Rp_H > 0 or Rp_O > 0):
        shift_f = np.clip(4.0 * F_CONST * eta_H / (R_GAS * T_K), -25.0, 25.0)
        shift_o = np.clip(-4.0 * F_CONST * eta_O / (R_GAS * T_K), -25.0, 25.0)
        pf, vf, hf, of = solve_boundary_defects(p_H2O_f, p_O2_f * np.exp(shift_f), c_R, K_H2O, K_O2, use_trapping, K_trap)
        po, vo, ho, oo = solve_boundary_defects(p_H2O_o, p_O2_o * np.exp(shift_o), c_R, K_H2O, K_O2, use_trapping, K_trap)
    else:
        pf, vf, hf, of = pf_eq, vf_eq, hf_eq, of_eq
        po, vo, ho, oo = po_eq, vo_eq, ho_eq, oo_eq

    cf_p, cf_v, cf_h = pf / Vm, vf / Vm, hf / Vm
    co_p, co_v, co_h = po / Vm, vo / Vm, ho / Vm

    def total_current_alg(dPhi_V):
        u1 = F_CONST * dPhi_V / (R_GAS * T_K)
        u2 = 2.0 * u1
        Jp = -(D_p / L_m) * bernoulli_flux(u1, cf_p, co_p)
        Jv = -(D_v / L_m) * bernoulli_flux(u2, cf_v, co_v)
        Jh = -(D_h / L_m) * bernoulli_flux(u1, cf_h, co_h)
        return F_CONST * (Jp + 2.0 * Jv + Jh) - i_A_m2

    lo, hi = -3.0, 3.0
    f_lo, f_hi = total_current_alg(lo), total_current_alg(hi)
    for _ in range(12):
        if f_lo * f_hi < 0:
            break
        lo *= 2.0
        hi *= 2.0
        f_lo, f_hi = total_current_alg(lo), total_current_alg(hi)

    dPhi_alg = brentq(total_current_alg, lo, hi, xtol=1e-10, rtol=1e-10)
    u1_alg = F_CONST * dPhi_alg / (R_GAS * T_K)
    u2_alg = 2.0 * u1_alg

    x_norm = np.linspace(0.0, 1.0, n_grid)
    cp_prof = profile_exp(x_norm, u1_alg, pf, po)
    cv_prof = profile_exp(x_norm, u2_alg, vf, vo)
    ch_prof = profile_exp(x_norm, u1_alg, hf, ho)
    Phi_prof = dPhi_alg * x_norm
    dPhi_dx_prof = np.full_like(x_norm, dPhi_alg / L_m)
    E_field_prof = -dPhi_dx_prof

    J_p = -(D_p / L_m) * bernoulli_flux(u1_alg, cf_p, co_p)
    J_v = -(D_v / L_m) * bernoulli_flux(u2_alg, cf_v, co_v)
    J_h = -(D_h / L_m) * bernoulli_flux(u1_alg, cf_h, co_h)

    bvp_converged = True
    bvp_mesh_warn = False
    solver_used = "Algebraic (Constant Field)"

    if solver_mode == "bvp":
        sol, ok, mesh_warn = _solve_bvp_with_homotopy(
            x_norm, pf, vf, hf, po, vo, ho, c_R,
            D_p, D_v, D_h, L_m, Vm, T_K, i_A_m2,
            cp_prof, cv_prof, ch_prof, J_p, J_v, J_h, u1_alg
        )
        if ok:
            y_sol = sol.sol(x_norm)
            cp_prof = np.maximum(y_sol[0], 1e-18)
            cv_prof = np.maximum(y_sol[1], 1e-18)
            ch_prof = np.maximum(y_sol[2], 1e-18)
            J_p = float(np.mean(y_sol[3])) * D_p / (L_m * Vm)
            J_v = float(np.mean(y_sol[4])) * D_p / (L_m * Vm)
            J_h = float(np.mean(y_sol[5])) * D_p / (L_m * Vm)
            Phi_prof = y_sol[6] * (R_GAS * T_K / F_CONST)
            dPhi_dx_prof = y_sol[7] * (R_GAS * T_K / (F_CONST * L_m))
            E_field_prof = -dPhi_dx_prof
            dPhi_final = float(Phi_prof[-1] - Phi_prof[0])
            bvp_mesh_warn = mesh_warn
            solver_used = "1D BVP (Exact)"
        else:
            bvp_converged = False
            dPhi_final = dPhi_alg
            solver_used = "Algebraic Fallback"
    else:
        dPhi_final = dPhi_alg

    co_prof = np.maximum(3.0 - cp_prof - cv_prof - ch_prof, 1e-12)

    pref = (F_CONST ** 2 / (R_GAS * T_K * Vm)) * 1e-2
    sig_p_prof = pref * (1.0 ** 2) * cp_prof * D_p
    sig_v_prof = pref * (2.0 ** 2) * cv_prof * D_v
    sig_h_prof = pref * (1.0 ** 2) * ch_prof * D_h
    sig_tot_prof = sig_p_prof + sig_v_prof + sig_h_prof

    L_cm = L_um * 1e-4
    ASR_ohm = float(_trapz(1.0 / np.maximum(sig_tot_prof, 1e-15), x_norm) * L_cm)

    rho_p_int = float(_trapz(1.0 / np.maximum(sig_p_prof, 1e-18), x_norm))
    rho_v_int = float(_trapz(1.0 / np.maximum(sig_v_prof, 1e-18), x_norm))
    rho_h_int = float(_trapz(1.0 / np.maximum(sig_h_prof, 1e-18), x_norm))
    sig_p_eff = 1.0 / rho_p_int
    sig_v_eff = 1.0 / rho_v_int
    sig_h_eff = 1.0 / rho_h_int
    sig_sum_eff = sig_p_eff + sig_v_eff + sig_h_eff
    t_p_rho = sig_p_eff / sig_sum_eff
    t_v_rho = sig_v_eff / sig_sum_eff
    t_h_rho = sig_h_eff / sig_sum_eff

    x_m = x_norm * L_m
    dcp_dx = np.gradient(cp_prof / Vm, x_m)
    dcv_dx = np.gradient(cv_prof / Vm, x_m)
    dch_dx = np.gradient(ch_prof / Vm, x_m)

    i_p_diff_prof = (-1.0 * F_CONST * D_p * dcp_dx) * 1e-4
    i_p_mig_prof = (-1.0 * F_CONST * D_p * (cp_prof / Vm) * (1.0 * F_CONST / (R_GAS * T_K)) * dPhi_dx_prof) * 1e-4
    i_v_diff_prof = (-2.0 * F_CONST * D_v * dcv_dx) * 1e-4
    i_v_mig_prof = (-2.0 * F_CONST * D_v * (cv_prof / Vm) * (2.0 * F_CONST / (R_GAS * T_K)) * dPhi_dx_prof) * 1e-4
    i_h_diff_prof = (-1.0 * F_CONST * D_h * dch_dx) * 1e-4
    i_h_mig_prof = (-1.0 * F_CONST * D_h * (ch_prof / Vm) * (1.0 * F_CONST / (R_GAS * T_K)) * dPhi_dx_prof) * 1e-4

    Pe_p_prof = i_p_mig_prof / np.where(np.abs(i_p_diff_prof) > 1e-12, i_p_diff_prof, np.nan)
    Pe_v_prof = i_v_mig_prof / np.where(np.abs(i_v_diff_prof) > 1e-12, i_v_diff_prof, np.nan)
    Pe_h_prof = i_h_mig_prof / np.where(np.abs(i_h_diff_prof) > 1e-12, i_h_diff_prof, np.nan)

    p_H2O_int = (cp_prof ** 2) / (co_prof * cv_prof * K_H2O)
    p_O2_int = ((ch_prof ** 2) / (co_prof * cv_prof * K_O2)) ** 2
    p_H2_int = p_H2O_int / (K_W * np.sqrt(np.maximum(p_O2_int, 1e-35)))
    p_O2_max = float(np.max(p_O2_int))

    i_p = (1.0 * F_CONST * J_p) * 1e-4
    i_v = (2.0 * F_CONST * J_v) * 1e-4
    i_h = (1.0 * F_CONST * J_h) * 1e-4
    i_tot = i_p + i_v + i_h

    q_diss_p = (i_p ** 2) / np.maximum(sig_p_prof, 1e-15)
    q_diss_v = (i_v ** 2) / np.maximum(sig_v_prof, 1e-15)
    q_diss_h = (i_h ** 2) / np.maximum(sig_h_prof, 1e-15)
    q_diss_tot = q_diss_p + q_diss_v + q_diss_h

    eps_chem_abs = (beta_p * cp_prof + beta_v * cv_prof) * 100.0

    I_zero_dPhi = (
        -F_CONST * (D_p * 1.0 * (co_p - cf_p) + D_v * 2.0 * (co_v - cf_v) + D_h * 1.0 * (co_h - cf_h)) / L_m
    ) * 1e-4

    sum_abs_i = abs(i_p) + abs(i_v) + abs(i_h)
    r_leak = (abs(i_h) / sum_abs_i * 100.0) if sum_abs_i > 1e-15 else 0.0

    J_H2, J_H2O, J_O2 = compute_gas_fluxes(J_p, J_v, p_H2_f, p_H2O_f, p_O2_f)
    eta_conc_H, eta_conc_O = compute_dgm_overpotentials(T_K, J_H2, p_H2_f, p_H2O_o, p_O2_o) if use_dgm else (0.0, 0.0)

    if use_arcsinh_shift and (Rp_H > 0 or Rp_O > 0):
        E_cell_shifted = (R_GAS * T_K / F_CONST) * np.log((ho / oo) / (hf / of))
        V_cell = E_cell_shifted + dPhi_final - (eta_conc_H + eta_conc_O)
    else:
        V_cell = E_eq + dPhi_final - (eta_H + eta_O + eta_conc_H + eta_conc_O)

    i_H2_equiv = (2.0 * F_CONST * J_H2) * 1e-4

    if i_A_cm2 > 1e-6:
        FE = (i_tot / i_H2_equiv) if abs(i_H2_equiv) > 1e-12 else 0.0
        mode_str = "PCFC (Fuel Cell)"
    elif i_A_cm2 < -1e-6:
        FE = (i_H2_equiv / i_tot) if abs(i_tot) > 1e-12 else 0.0
        mode_str = "PCEC (Electrolyzer)"
    else:
        FE = 0.0
        mode_str = "OCV (Open Circuit)"

    t_p_cond = float(np.mean(sig_p_prof / sig_tot_prof))
    t_v_cond = float(np.mean(sig_v_prof / sig_tot_prof))
    t_h_cond = float(np.mean(sig_h_prof / sig_tot_prof))

    J_H2_cm2 = J_H2 * 1e-4
    J_H2_Nml_min = J_H2_cm2 * V_MOLAR_STP * 60.0
    q_dot = -i_tot * V_cell + J_H2_cm2 * abs(DH_W)
    V_TN_theo = abs(DH_W) / (2.0 * F_CONST)
    V_TN_act = max(FE, 0.0) * V_TN_theo if i_A_cm2 < -1e-6 else V_TN_theo

    P_elec = i_tot * V_cell
    if i_A_cm2 > 1e-6:
        eff_LHV = P_elec / (J_H2_cm2 * LHV_H2) if J_H2_cm2 > 1e-12 else 0.0
    elif i_A_cm2 < -1e-6:
        Q_in = max(0.0, -q_dot)
        denom_ec = abs(P_elec) + Q_in
        eff_LHV = (abs(J_H2_cm2) * LHV_H2) / denom_ec if denom_ec > 1e-12 else 0.0
    else:
        eff_LHV = 0.0

    return {
        "mode": mode_str,
        "solver_used": solver_used,
        "bvp_converged": bvp_converged,
        "bvp_mesh_warn": bvp_mesh_warn,
        "T_C": T_C, "T_cell_eff": T_C, "i": i_tot, "V_cell": V_cell, "E_eq": E_eq,
        "dPhi": dPhi_final, "eta_H": eta_H, "eta_O": eta_O,
        "eta_conc_H": eta_conc_H, "eta_conc_O": eta_conc_O,
        "i_p": i_p, "i_v": i_v, "i_h": i_h, "r_leak": r_leak,
        "t_p": t_p_cond, "t_v": t_v_cond, "t_h": t_h_cond,
        "t_p_rho": t_p_rho, "t_v_rho": t_v_rho, "t_h_rho": t_h_rho,
        "sig_p_eff": sig_p_eff, "sig_v_eff": sig_v_eff, "sig_h_eff": sig_h_eff,
        "ASR_ohm": ASR_ohm, "p_O2_max": p_O2_max, "I_zero_dPhi": I_zero_dPhi,
        "FE": FE * 100.0, "eff_LHV": eff_LHV * 100.0,
        "P_elec": P_elec, "q_dot": q_dot,
        "V_TN_theo": V_TN_theo, "V_TN_act": V_TN_act,
        "J_H2": J_H2_cm2, "J_H2_Nml_min": J_H2_Nml_min,
        "J_H2O": J_H2O * 1e-4, "J_O2": J_O2 * 1e-4,
        "x_norm": x_norm, "cp": cp_prof, "cv": cv_prof, "ch": ch_prof, "co": co_prof,
        "Phi": Phi_prof, "E_field": E_field_prof,
        "i_p_diff_prof": i_p_diff_prof, "i_p_mig_prof": i_p_mig_prof,
        "i_v_diff_prof": i_v_diff_prof, "i_v_mig_prof": i_v_mig_prof,
        "i_h_diff_prof": i_h_diff_prof, "i_h_mig_prof": i_h_mig_prof,
        "Pe_p": Pe_p_prof, "Pe_v": Pe_v_prof, "Pe_h": Pe_h_prof,
        "q_diss_p": q_diss_p, "q_diss_v": q_diss_v, "q_diss_h": q_diss_h, "q_diss_tot": q_diss_tot,
        "eps_chem_abs": eps_chem_abs,
        "sig_p": sig_p_prof, "sig_v": sig_v_prof, "sig_h": sig_h_prof, "sig_tot": sig_tot_prof,
        "p_H2O_int": p_H2O_int, "p_O2_int": p_O2_int, "p_H2_int": p_H2_int,
    }


def solve_single_point(
    i_A_cm2: float,
    T_C: float,
    L_um: float,
    Rp_H: float,
    Rp_O: float,
    p_H2_f: float,
    p_H2O_f: float,
    p_H2O_o: float,
    c_R: float,
    mat: dict,
    solver_mode: str = "fast",
    use_arcsinh_shift: bool = True,
    n_grid: int = 80,
    use_trapping: bool = False,
    use_dgm: bool = False,
    use_self_heating: bool = False,
    h_cool: float = 0.02,
):
    if not use_self_heating or h_cool <= 0:
        return _solve_single_point_core(
            i_A_cm2, T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R,
            mat, solver_mode, use_arcsinh_shift, n_grid, use_trapping, use_dgm
        )

    T_curr = T_C
    for _ in range(15):
        r_tmp = _solve_single_point_core(
            i_A_cm2, T_curr, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R,
            mat, "fast", use_arcsinh_shift, 20, use_trapping, use_dgm
        )
        T_next = float(np.clip(T_C + r_tmp["q_dot"] / h_cool, 350.0, 950.0))
        if abs(T_next - T_curr) < 0.05:
            T_curr = T_next
            break
        T_curr = 0.6 * T_curr + 0.4 * T_next

    res = _solve_single_point_core(
        i_A_cm2, T_curr, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R,
        mat, solver_mode, use_arcsinh_shift, n_grid, use_trapping, use_dgm
    )
    res["T_cell_eff"] = T_curr
    return res


def find_special_points(
    T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat,
    use_arcsinh_shift, use_trapping=False, use_dgm=False, use_self_heating=False, h_cool=0.02
):
    def ip_func(iv):
        return solve_single_point(
            iv, T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat,
            "fast", use_arcsinh_shift, 15, use_trapping, use_dgm, use_self_heating, h_cool
        )["i_p"]

    i_trans = np.nan
    try:
        for span in [0.5, 2.0, 10.0]:
            if ip_func(-span) * ip_func(span) < 0:
                i_trans = brentq(ip_func, -span, span, xtol=1e-4)
                break
    except ValueError:
        pass

    def q_func(iv):
        return solve_single_point(
            iv, T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat,
            "fast", use_arcsinh_shift, 15, use_trapping, use_dgm, use_self_heating, h_cool
        )["q_dot"]

    i_TN, V_TN_pt = np.nan, np.nan
    try:
        i_hi_tn = -0.02
        q_hi = q_func(i_hi_tn)
        for i_lo_tn in [-1.5, -4.0, -10.0, -25.0, -60.0, -100.0]:
            q_lo = q_func(i_lo_tn)
            if q_lo * q_hi < 0:
                i_TN = brentq(q_func, i_lo_tn, i_hi_tn, xtol=1e-4)
                r_tn = solve_single_point(
                    i_TN, T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat,
                    "fast", use_arcsinh_shift, 15, use_trapping, use_dgm, use_self_heating, h_cool
                )
                V_TN_pt = r_tn["V_cell"]
                break
    except ValueError:
        pass

    i_fc_scan = np.linspace(0.05, 5.0, 40)
    p_vals = []
    for iv in i_fc_scan:
        r = solve_single_point(
            float(iv), T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat,
            "fast", use_arcsinh_shift, 15, use_trapping, use_dgm, use_self_heating, h_cool
        )
        p_vals.append(r["P_elec"] if r["V_cell"] > 0 else 0.0)
    idx_max = int(np.argmax(p_vals))
    P_max = float(p_vals[idx_max])
    i_Pmax = float(i_fc_scan[idx_max])

    return P_max, i_Pmax, i_TN, V_TN_pt, i_trans


def solve_current_for_voltage(
    V_target, T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat,
    use_arcsinh_shift, use_trapping, use_dgm
):
    def f_v(iv):
        return _solve_single_point_core(
            iv, T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat,
            "fast", use_arcsinh_shift, 15, use_trapping, use_dgm
        )["V_cell"] - V_target

    lo, hi = -8.0, 4.0
    if f_v(lo) * f_v(hi) > 0:
        return np.nan
    return float(brentq(f_v, lo, hi, xtol=1e-5))


# =============================================================================
# 2.5 CACHED COMPUTATION WRAPPERS
# =============================================================================
@st.cache_data(show_spinner=False)
def cached_single_point(
    i_A_cm2, T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R,
    mat_tuple, solver_mode, use_arcsinh_shift, use_trapping, use_dgm, use_self_heating, h_cool, n_grid=80
):
    mat = dict(mat_tuple)
    return solve_single_point(
        i_A_cm2, T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R,
        mat, solver_mode, use_arcsinh_shift, n_grid, use_trapping, use_dgm, use_self_heating, h_cool
    )


@st.cache_data(show_spinner="Computing I–V sweep...")
def cached_iv_sweep(
    i_min, i_max, n_pts, T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R,
    mat_tuple, solver_mode, use_arcsinh_shift, use_trapping, use_dgm, use_self_heating, h_cool
):
    mat = dict(mat_tuple)
    i_grid = np.linspace(i_min, i_max, n_pts)
    sweep_rows = []
    for iv in i_grid:
        r = solve_single_point(
            float(iv), T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R,
            mat, solver_mode, use_arcsinh_shift, 40, use_trapping, use_dgm, use_self_heating, h_cool
        )
        sweep_rows.append({k: r[k] for k in [
            "i", "V_cell", "E_eq", "dPhi", "eta_H", "eta_O", "eta_conc_H", "eta_conc_O",
            "i_p", "i_v", "i_h", "r_leak", "t_p", "t_v", "t_h",
            "t_p_rho", "t_v_rho", "t_h_rho", "sig_p_eff", "sig_h_eff",
            "ASR_ohm", "p_O2_max", "T_cell_eff",
            "FE", "eff_LHV", "P_elec", "q_dot", "V_TN_theo", "V_TN_act"
        ]})
    return pd.DataFrame(sweep_rows)


@st.cache_data(show_spinner="Computing multiparametric analysis...")
def cached_multiparametric(
    param_key, v_min, v_max, n_vals, i_min, i_max, i_work,
    T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
    binary_fuel, mat_tuple, use_arcsinh_shift, use_trapping, use_dgm, use_self_heating, h_cool
):
    mat = dict(mat_tuple)
    var_values = np.linspace(v_min, v_max, n_vals)
    i_multi = np.linspace(i_min, i_max, 33)
    multi_iv = {}
    summary_rows = []

    for val in var_values:
        T_v = float(val) if param_key == "T_C" else T_work
        L_v = float(val) if param_key == "L_um" else L_work
        cR_v = float(val) if param_key == "c_R" else c_R_work
        pH2_v = float(val) if param_key == "p_H2_f" else p_H2_f
        pH2Of_v = max(1.0 - pH2_v, 1e-4) if (param_key == "p_H2_f" and binary_fuel) else p_H2O_f
        pH2Oo_v = float(val) if param_key == "p_H2O_o" else p_H2O_o
        RpH_v = float(val) if param_key == "Rp" else Rp_H
        RpO_v = float(val) if param_key == "Rp" else Rp_O

        rows_v = []
        for iv in i_multi:
            r = solve_single_point(
                float(iv), T_v, L_v, RpH_v, RpO_v, pH2_v, pH2Of_v, pH2Oo_v, cR_v,
                mat, "fast", use_arcsinh_shift, 25, use_trapping, use_dgm, use_self_heating, h_cool
            )
            rows_v.append({k: r[k] for k in ["i", "V_cell", "i_h", "FE", "P_elec"]})
        multi_iv[float(val)] = pd.DataFrame(rows_v)

        r_ocv = solve_single_point(
            0.0, T_v, L_v, RpH_v, RpO_v, pH2_v, pH2Of_v, pH2Oo_v, cR_v,
            mat, "fast", use_arcsinh_shift, 25, use_trapping, use_dgm, use_self_heating, h_cool
        )
        r_wk = solve_single_point(
            i_work, T_v, L_v, RpH_v, RpO_v, pH2_v, pH2Of_v, pH2Oo_v, cR_v,
            mat, "fast", use_arcsinh_shift, 25, use_trapping, use_dgm, use_self_heating, h_cool
        )
        P_max, i_Pmax, i_TN, V_TN_pt, i_trans = find_special_points(
            T_v, L_v, RpH_v, RpO_v, pH2_v, pH2Of_v, pH2Oo_v, cR_v, mat,
            use_arcsinh_shift, use_trapping, use_dgm, use_self_heating, h_cool
        )

        summary_rows.append({
            "param": float(val),
            "P_max": P_max, "i_Pmax": i_Pmax,
            "i_TN": i_TN, "V_TN_pt": V_TN_pt, "V_TN_act_wk": r_wk["V_TN_act"],
            "J_H2": abs(r_wk["J_H2"]), "J_H2_Nml_min": abs(r_wk["J_H2_Nml_min"]),
            "FE_wk": r_wk["FE"], "eff_wk": r_wk["eff_LHV"],
            "E_eq": r_ocv["E_eq"], "OCV": r_ocv["V_cell"],
            "RdOCV_mV": (r_ocv["E_eq"] - r_ocv["V_cell"]) * 1e3,
            "i_leak_OCV": abs(r_ocv["i_h"]), "i_trans": i_trans,
            "ASR_ohm": r_wk["ASR_ohm"], "t_p_wk": r_wk["t_p"] * 100.0,
            "t_p_rho_wk": r_wk["t_p_rho"] * 100.0,
            "q_dot_wk": r_wk["q_dot"], "V_cell_wk": r_wk["V_cell"],
        })

    return var_values, multi_iv, pd.DataFrame(summary_rows)


@st.cache_data(show_spinner="Computing 2D/3D parametric maps...")
def cached_heatmaps(
    res_hm, T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
    mat_tuple, use_arcsinh_shift, use_trapping, use_dgm, use_self_heating, h_cool
):
    mat = dict(mat_tuple)
    T_hm = np.linspace(480, 700, res_hm)
    i_hm_ec = np.linspace(-1.8, -0.1, res_hm)
    L_hm = np.linspace(3.0, 50.0, res_hm)
    x_hm = np.linspace(0.0, 1.0, res_hm)

    FE_map = np.zeros((res_hm, res_hm))
    Q_map = np.zeros((res_hm, res_hm))
    Leak_map = np.zeros((res_hm, res_hm))
    V_map = np.zeros((res_hm, res_hm))
    pO2_max_map = np.zeros((res_hm, res_hm))
    Eff_L_map = np.zeros((res_hm, res_hm))
    pO2_map = np.zeros((res_hm, res_hm))

    for idx_t, tv in enumerate(T_hm):
        for idx_i, iv in enumerate(i_hm_ec):
            r = solve_single_point(
                float(iv), float(tv), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
                mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm, use_self_heating, h_cool
            )
            FE_map[idx_t, idx_i] = np.clip(r["FE"], 0, 100)
            Q_map[idx_t, idx_i] = r["q_dot"]
            Leak_map[idx_t, idx_i] = abs(r["i_h"]) * 1e3
            V_map[idx_t, idx_i] = r["V_cell"]
            pO2_max_map[idx_t, idx_i] = r["p_O2_max"]

    for idx_l, lv in enumerate(L_hm):
        for idx_i, iv in enumerate(i_hm_ec):
            r = solve_single_point(
                float(iv), T_work, float(lv), Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
                mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm, use_self_heating, h_cool
            )
            Eff_L_map[idx_l, idx_i] = np.clip(r["eff_LHV"], 0, 100)

    for idx_i, iv in enumerate(i_hm_ec):
        r = solve_single_point(
            float(iv), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
            mat, "fast", use_arcsinh_shift, res_hm, use_trapping, use_dgm, use_self_heating, h_cool
        )
        pO2_map[:, idx_i] = np.log10(np.maximum(r["p_O2_int"], 1e-30))

    return T_hm, i_hm_ec, L_hm, x_hm, FE_map, Q_map, Leak_map, V_map, pO2_max_map, Eff_L_map, pO2_map


@st.cache_data(show_spinner="Computing Restrepo / Thatte / Putilov benchmarks...")
def cached_literature_insights(
    T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
    mat_tuple, use_arcsinh_shift, use_trapping, use_dgm
):
    mat = dict(mat_tuple)

    # 1. Zero voltage drop current I_0(T, p_O2) (Restrepo 2026 Fig. 14)
    T_i0 = np.linspace(450, 700, 18)
    pO2_i0 = np.linspace(0.05, 0.95, 18)
    I0_map = np.zeros((len(T_i0), len(pO2_i0)))
    for it, tv in enumerate(T_i0):
        for ip, po2 in enumerate(pO2_i0):
            pH2O_bal = max(1.0 - po2, 0.02)
            r = _solve_single_point_core(
                0.0, float(tv), L_work, 0.0, 0.0, p_H2_f, p_H2O_f, pH2O_bal, c_R_work,
                mat, "fast", False, 15, use_trapping, False
            )
            I0_map[it, ip] = abs(r["I_zero_dPhi"])

    # 2. Optimum membrane thickness L_opt(T) for PCEC (-0.8) and PCFC (+0.5) (Restrepo 2026 Fig. 13a)
    T_lopt = np.linspace(480, 680, 11)
    Lopt_ec, eff_ec_opt, Lopt_fc, eff_fc_opt = [], [], [], []
    for tv in T_lopt:
        res_ec = minimize_scalar(
            lambda l_val: -_solve_single_point_core(
                -0.8, float(tv), float(l_val), Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
                mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm
            )["eff_LHV"],
            bounds=(1.0, 80.0), method="bounded"
        )
        Lopt_ec.append(res_ec.x)
        eff_ec_opt.append(-res_ec.fun)

        res_fc = minimize_scalar(
            lambda l_val: -_solve_single_point_core(
                0.5, float(tv), float(l_val), Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
                mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm
            )["eff_LHV"],
            bounds=(1.0, 80.0), method="bounded"
        )
        Lopt_fc.append(res_fc.x)
        eff_fc_opt.append(-res_fc.fun)

    # 3. Actual Thermoneutral Voltage Map V_TN^act(T) (Thatte 2024 Fig. 7a)
    T_tn = np.linspace(500, 680, 12)
    tn_data = {}
    for x_st in [0.20, 0.50, 0.90]:
        v_lo, v_mid, v_hi = [], [], []
        for tv in T_tn:
            r_lo = _solve_single_point_core(-0.4, float(tv), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, x_st, c_R_work, mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm)
            r_mid = _solve_single_point_core(-1.0, float(tv), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, x_st, c_R_work, mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm)
            r_hi = _solve_single_point_core(-2.0, float(tv), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, x_st, c_R_work, mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm)
            v_lo.append(r_lo["V_TN_act"])
            v_mid.append(r_mid["V_TN_act"])
            v_hi.append(r_hi["V_TN_act"])
        tn_data[x_st] = (np.array(v_lo), np.array(v_mid), np.array(v_hi))

    # 4. Working Mode (U=0.7V) vs OCV Comparison across fuel humidity (Putilov 2020 Fig. 11)
    pH2O_f_grid = np.linspace(0.02, 0.50, 13)
    putilov_rows = []
    for ph in pH2O_f_grid:
        ph2 = max(1.0 - ph, 0.05)
        r_oc = _solve_single_point_core(0.0, T_work, L_work, Rp_H, Rp_O, ph2, ph, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 20, use_trapping, use_dgm)
        i_07 = solve_current_for_voltage(0.70, T_work, L_work, Rp_H, Rp_O, ph2, ph, p_H2O_o, c_R_work, mat, use_arcsinh_shift, use_trapping, use_dgm)
        if np.isnan(i_07):
            i_07 = 0.4
        r_op = _solve_single_point_core(i_07, T_work, L_work, Rp_H, Rp_O, ph2, ph, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 20, use_trapping, use_dgm)
        putilov_rows.append({
            "pH2O_f": ph,
            "sig_p_ocv": r_oc["sig_p_eff"] * 1e3, "sig_p_op": r_op["sig_p_eff"] * 1e3,
            "sig_h_ocv": r_oc["sig_h_eff"] * 1e3, "sig_h_op": r_op["sig_h_eff"] * 1e3,
            "tp_ocv": r_oc["t_p_rho"] * 100.0, "tp_op": r_op["t_p_rho"] * 100.0,
            "ih_ocv": abs(r_oc["i_h"]) * 1e3, "ih_op": abs(r_op["i_h"]) * 1e3,
        })

    # 5. Trapping vs No Trapping (Moranti 2026 Fig. 7)
    i_trap = np.linspace(-1.5, -0.1, 18)
    v_tr, fe_tr, v_no, fe_no = [], [], [], []
    for iv in i_trap:
        rt = _solve_single_point_core(float(iv), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 15, True, use_dgm)
        rn = _solve_single_point_core(float(iv), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 15, False, use_dgm)
        v_tr.append(rt["V_cell"]); fe_tr.append(rt["FE"])
        v_no.append(rn["V_cell"]); fe_no.append(rn["FE"])

    return (
        T_i0, pO2_i0, I0_map,
        T_lopt, np.array(Lopt_ec), np.array(eff_ec_opt), np.array(Lopt_fc), np.array(eff_fc_opt),
        T_tn, tn_data,
        pd.DataFrame(putilov_rows),
        i_trap, np.array(v_tr), np.array(fe_tr), np.array(v_no), np.array(fe_no)
    )


@st.cache_data(show_spinner="Computing 1+1D along-channel distribution...")
def cached_channel_1d(
    V_ch, U_target, T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
    mat_tuple, use_arcsinh_shift, use_trapping, use_dgm, n_seg=25
):
    mat = dict(mat_tuple)
    y_norm = np.linspace(0.0, 1.0, n_seg)
    rows = []
    # Co-flow PCEC along channel: steam is depleted from p_H2O_o to p_H2O_o * (1 - U_target)
    for y in y_norm:
        u_loc = U_target * y
        p_st_loc = max(p_H2O_o * (1.0 - u_loc), 0.02)
        p_h2_loc = min(p_H2_f + (1.0 - p_H2_f) * 0.5 * u_loc, 0.995)
        p_h2of_loc = max(1.0 - p_h2_loc, 0.005)
        i_loc = solve_current_for_voltage(
            V_ch, T_work, L_work, Rp_H, Rp_O, p_h2_loc, p_h2of_loc, p_st_loc, c_R_work,
            mat, use_arcsinh_shift, use_trapping, use_dgm
        )
        if np.isnan(i_loc):
            i_loc = -0.5
        r = _solve_single_point_core(
            i_loc, T_work, L_work, Rp_H, Rp_O, p_h2_loc, p_h2of_loc, p_st_loc, c_R_work,
            mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm
        )
        rows.append({
            "y": y, "p_H2O_o": p_st_loc, "p_H2_f": p_h2_loc,
            "i_loc": r["i"], "FE_loc": r["FE"], "V_TN_act": r["V_TN_act"], "q_dot": r["q_dot"]
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing novel research insights...")
def cached_novel_insights(
    T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
    mat_tuple, use_arcsinh_shift, use_trapping, use_dgm
):
    mat = dict(mat_tuple)

    # 1. Thermal Instability Criterion h_min = d(q_dot)/dT at constant V_cell
    T_inst = np.linspace(500, 680, 14)
    V_inst = np.linspace(1.10, 1.50, 14)
    hmin_map = np.zeros((len(T_inst), len(V_inst)))
    dT = 2.0
    for it, tv in enumerate(T_inst):
        for iv, vv in enumerate(V_inst):
            i_m = solve_current_for_voltage(float(vv), float(tv - dT), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, use_arcsinh_shift, use_trapping, use_dgm)
            i_p = solve_current_for_voltage(float(vv), float(tv + dT), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, use_arcsinh_shift, use_trapping, use_dgm)
            if np.isnan(i_m) or np.isnan(i_p):
                hmin_map[it, iv] = np.nan
            else:
                qm = _solve_single_point_core(i_m, float(tv - dT), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 12, use_trapping, use_dgm)["q_dot"]
                qp = _solve_single_point_core(i_p, float(tv + dT), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 12, use_trapping, use_dgm)["q_dot"]
                hmin_map[it, iv] = (qp - qm) / (2.0 * dT)

    # 2. Reversible Round-Trip Efficiency eta_RT(i_EC, i_FC)
    i_ec_rt = np.linspace(-1.5, -0.15, 16)
    i_fc_rt = np.linspace(0.15, 1.2, 16)
    RT_map = np.zeros((len(i_fc_rt), len(i_ec_rt)))
    ec_cache = [
        _solve_single_point_core(float(ie), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm)
        for ie in i_ec_rt
    ]
    fc_cache = [
        _solve_single_point_core(float(ifc), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 15, use_trapping, use_dgm)
        for ifc in i_fc_rt
    ]
    for r_i, r_fc in enumerate(fc_cache):
        for c_i, r_ec in enumerate(ec_cache):
            if r_fc["V_cell"] > 0 and r_ec["V_cell"] > 0:
                RT_map[r_i, c_i] = np.clip((r_fc["V_cell"] / r_ec["V_cell"]) * (r_ec["FE"] / 100.0) * (r_fc["FE"] / 100.0) * 100.0, 0, 100)

    # 3. Chemical expansion waterfall profiles across current densities
    r_ocv_ref = _solve_single_point_core(0.0, T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 40, use_trapping, use_dgm)
    i_wf = np.linspace(-1.5, 1.0, 9)
    chem_wf = {}
    for iv in i_wf:
        r_w = _solve_single_point_core(float(iv), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 40, use_trapping, use_dgm)
        chem_wf[float(iv)] = r_w["eps_chem_abs"] - r_ocv_ref["eps_chem_abs"]

    # 4. Elasticity / Tornado sensitivity at i = -0.8 A/cm^2
    base_r = _solve_single_point_core(-0.8, T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 20, use_trapping, use_dgm)
    eps_rel = 0.05
    sens_keys = [
        ("ΔH°(H₂O)", "dH_H2O", "mat"),
        ("ΔH°(O₂)", "dH_O2", "mat"),
        ("E_act(OH•)", "E_p", "mat"),
        ("E_act(O•)", "E_h", "mat"),
        ("Dopant c_R", "c_R", "arg"),
        ("Thickness L", "L_um", "arg"),
    ]
    sens_rows = []
    for label_s, key_s, kind_s in sens_keys:
        mat_p = dict(mat); mat_m = dict(mat)
        cR_p, cR_m, L_p, L_m = c_R_work, c_R_work, L_work, L_work
        if kind_s == "mat":
            mat_p[key_s] = mat[key_s] * (1.0 + eps_rel)
            mat_m[key_s] = mat[key_s] * (1.0 - eps_rel)
        elif key_s == "c_R":
            cR_p = min(c_R_work * (1.0 + eps_rel), 0.35)
            cR_m = max(c_R_work * (1.0 - eps_rel), 0.05)
        elif key_s == "L_um":
            L_p = L_work * (1.0 + eps_rel)
            L_m = L_work * (1.0 - eps_rel)
        rp = _solve_single_point_core(-0.8, T_work, L_p, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, cR_p, mat_p, "fast", use_arcsinh_shift, 20, use_trapping, use_dgm)
        rm = _solve_single_point_core(-0.8, T_work, L_m, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, cR_m, mat_m, "fast", use_arcsinh_shift, 20, use_trapping, use_dgm)
        s_fe = (rp["FE"] - rm["FE"]) / (2.0 * eps_rel * max(abs(base_r["FE"]), 1e-6))
        s_eff = (rp["eff_LHV"] - rm["eff_LHV"]) / (2.0 * eps_rel * max(abs(base_r["eff_LHV"]), 1e-6))
        s_asr = (rp["ASR_ohm"] - rm["ASR_ohm"]) / (2.0 * eps_rel * max(abs(base_r["ASR_ohm"]), 1e-6))
        sens_rows.append({"Parameter": label_s, "S_FE": s_fe, "S_eff": s_eff, "S_ASR": s_asr})

    # 5. Multi-material Radar comparison
    radar_rows = []
    for m_name, m_dict in MATERIAL_PRESETS.items():
        short_name = m_name.split(" (")[0]
        r_ec = _solve_single_point_core(-0.8, T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, m_dict["default_cR"], m_dict, "fast", use_arcsinh_shift, 20, use_trapping, use_dgm)
        r_fc = _solve_single_point_core(0.6, T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, m_dict["default_cR"], m_dict, "fast", use_arcsinh_shift, 20, use_trapping, use_dgm)
        radar_rows.append({
            "Material": short_name,
            "FE (%)": np.clip(r_ec["FE"], 0, 100),
            "Eff LHV (%)": np.clip(r_ec["eff_LHV"], 0, 100),
            "t_p_rho (%)": r_ec["t_p_rho"] * 100.0,
            "Conductivity (1/ASR)": 1.0 / max(r_ec["ASR_ohm"], 1e-4),
            "PCFC Power": max(r_fc["P_elec"], 0.0),
            "Leakage Safety": max(0.0, 100.0 - r_ec["r_leak"]),
        })

    return T_inst, V_inst, hmin_map, i_ec_rt, i_fc_rt, RT_map, r_ocv_ref["x_norm"], chem_wf, pd.DataFrame(sens_rows), pd.DataFrame(radar_rows)


@st.cache_data(show_spinner="Computing Algebraic vs Exact 1D BVP accuracy map...")
def cached_bvp_accuracy_map(
    n_bvp, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
    mat_tuple, use_arcsinh_shift, use_trapping, use_dgm
):
    mat = dict(mat_tuple)
    T_arr = np.linspace(500, 680, n_bvp)
    i_arr = np.linspace(-1.5, -0.2, n_bvp)
    err_fe = np.zeros((n_bvp, n_bvp))
    err_po2 = np.zeros((n_bvp, n_bvp))
    for it, tv in enumerate(T_arr):
        for ii, iv in enumerate(i_arr):
            r_alg = _solve_single_point_core(float(iv), float(tv), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "fast", use_arcsinh_shift, 35, use_trapping, use_dgm)
            r_bvp = _solve_single_point_core(float(iv), float(tv), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat, "bvp", use_arcsinh_shift, 35, use_trapping, use_dgm)
            if r_bvp["bvp_converged"]:
                err_fe[it, ii] = r_alg["FE"] - r_bvp["FE"]
                err_po2[it, ii] = np.log10(max(r_alg["p_O2_max"], 1e-30)) - np.log10(max(r_bvp["p_O2_max"], 1e-30))
            else:
                err_fe[it, ii] = np.nan
                err_po2[it, ii] = np.nan
    return T_arr, i_arr, err_fe, err_po2


# =============================================================================
# 3. SIDEBAR: INPUT PARAMETERS & PUBLICATION PLOT SETTINGS
# =============================================================================
st.sidebar.title("⚙️ Model Parameters")

preset_name = st.sidebar.selectbox("Electrolyte Material", list(MATERIAL_PRESETS.keys()))
mat_params = dict(MATERIAL_PRESETS[preset_name])

with st.sidebar.expander("🔬 Solver & Boundary Conditions", expanded=True):
    solver_label = st.radio(
        "NPP Solver Mode",
        ["Fast (Algebraic, Constant Field)", "Exact (1D BVP, Variable Field)"],
        index=0,
    )
    solver_mode = "fast" if "Fast" in solver_label else "bvp"
    use_arcsinh_shift = st.checkbox(
        "Shift boundary defects via Rp (arcsinh)",
        value=True,
        help="Couples electrode polarization overpotential with boundary defect concentrations and electronic leakage."
    )
    use_trapping = st.checkbox(
        "Enable polaron trapping (Y'–O•)",
        value=False,
        help="Includes acceptor-polaron trapping equilibrium K_p,Trap at membrane boundaries."
    )
    use_dgm = st.checkbox(
        "Enable porous electrode DGM (η_conc)",
        value=False,
        help="Adds simplified Dusty Gas Model concentration overpotential in porous electrodes."
    )
    use_self_heating = st.checkbox(
        "Enable 0D self-heating T_cell(i)",
        value=False,
        help="Solves T_cell = T_in + q_dot / h_cool to capture exothermic FE peak and I-V flattening."
    )
    h_cool = st.number_input("Cooling coeff. h_cool (W cm⁻² K⁻¹)", 0.002, 0.200, 0.015, 0.002, format="%.3f") if use_self_heating else 0.015

with st.sidebar.expander("⚡ Operating & Geometric Parameters", expanded=True):
    i_work = st.number_input("Current density i (A cm⁻²) [<0 PCEC, >0 PCFC]", -2.5, 2.5, -0.8, 0.05, format="%.2f")
    T_work = st.slider("Temperature T (°C)", 450, 750, 600, 5)
    L_work = st.number_input("Electrolyte thickness L (µm)", 1.0, 500.0, 10.0, 1.0)
    c_R_work = st.slider("Acceptor dopant concentration c_R (-)", 0.10, 0.30, float(mat_params["default_cR"]), 0.01)
    col_rp1, col_rp2 = st.columns(2)
    Rp_H = col_rp1.number_input("Rp(H₂) (Ω cm²)", 0.0, 2.0, 0.05, 0.01, format="%.3f")
    Rp_O = col_rp2.number_input("Rp(O₂) (Ω cm²)", 0.0, 2.0, 0.05, 0.01, format="%.3f")

with st.sidebar.expander("💨 Electrode Gas Compositions (atm)", expanded=True):
    binary_fuel = st.checkbox("Binary H₂/H₂O fuel (p_H₂O = 1 - p_H₂)", value=True)
    p_H2_f = st.slider("Fuel electrode p(H₂) (atm)", 0.05, 0.995, 0.97, 0.01)
    if binary_fuel:
        p_H2O_f = max(1.0 - p_H2_f, 1e-4)
        st.caption(f"Fuel electrode p(H₂O) = {p_H2O_f:.3f} atm")
    else:
        p_H2O_f = st.slider("Fuel electrode p(H₂O) (atm)", 0.005, 0.90, 0.03, 0.005)
        if p_H2_f + p_H2O_f > 1.0001:
            st.error("Fuel electrode p(H₂) + p(H₂O) exceeds 1.0 atm! Clamping p(H₂O) to 1 - p(H₂).")
            p_H2O_f = max(1.0 - p_H2_f, 1e-4)
    p_H2O_o = st.slider("Air/steam electrode p(H₂O) (atm)", 0.01, 0.95, 0.50, 0.01)
    st.caption(f"Air/steam electrode p(O₂) = {0.21 * (1.0 - p_H2O_o):.4f} atm")

with st.sidebar.expander("🧪 Custom Material Overrides", expanded=False):
    mat_params["dH_H2O"] = st.number_input("ΔH°(H₂O) (J mol⁻¹)", value=float(mat_params["dH_H2O"]), step=1000.0)
    mat_params["dS_H2O"] = st.number_input("ΔS°(H₂O) (J mol⁻¹ K⁻¹)", value=float(mat_params["dS_H2O"]), step=1.0)
    mat_params["dH_O2"] = st.number_input("ΔH°(O₂) (J mol⁻¹)", value=float(mat_params["dH_O2"]), step=1000.0)
    mat_params["dS_O2"] = st.number_input("ΔS°(O₂) (J mol⁻¹ K⁻¹)", value=float(mat_params["dS_O2"]), step=1.0)
    mat_params["E_p"] = st.number_input("E_act(OH•) (J mol⁻¹)", value=float(mat_params["E_p"]), step=500.0)
    mat_params["E_h"] = st.number_input("E_act(O•) (J mol⁻¹)", value=float(mat_params["E_h"]), step=500.0)

with st.sidebar.expander("🎨 Publication Plot Settings", expanded=False):
    fs_label = st.slider("Axis label font size", 8, 18, 11)
    fs_tick = st.slider("Tick label font size", 7, 16, 10)
    fs_title = st.slider("Title font size", 9, 20, 12)
    fs_legend = st.slider("Legend font size", 7, 16, 9)
    lw = st.slider("Line width", 1.0, 4.0, 2.0, 0.25)
    show_markers = st.checkbox("Show markers on curves", value=False)
    ms = st.slider("Marker size", 2, 10, 5)
    show_grid = st.checkbox("Show grid lines", value=True)
    show_contours = st.checkbox("Show contour lines & labels on heatmaps", value=True)
    pO2_crit_atm = st.number_input("Critical internal p(O₂) threshold (atm)", 0.1, 100.0, 1.0, 0.5)
    cmap_multi = st.selectbox("Colormap (multi-curve series)", ["coolwarm", "viridis", "plasma", "turbo", "tab10"])
    cmap_heat = st.selectbox("Colormap (2D/3D heatmaps)", ["inferno", "viridis", "magma", "RdYlBu_r", "plasma"])
    custom_legend_title = st.text_input("Custom legend title (multiparametric)", "")
    view_elev = st.slider("3D elevation angle (°)", 5, 80, 28)
    view_azim = st.slider("3D azimuth angle (°)", -180, 180, -55)

plt.rcParams.update({
    "font.size": fs_tick,
    "axes.labelsize": fs_label,
    "axes.labelweight": "bold",
    "axes.titlesize": fs_title,
    "axes.titleweight": "bold",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelsize": fs_tick,
    "ytick.labelsize": fs_tick,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": fs_legend,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "black",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

EXPORT_REGISTRY = {}


def style_legend(ax, title=None, loc="best", handles=None, labels=None):
    if handles is not None and labels is not None:
        leg = ax.legend(handles, labels, title=title, loc=loc)
    else:
        leg = ax.legend(title=title, loc=loc)
    if leg is not None:
        for text, handle in zip(leg.get_texts(), leg.legend_handles):
            if hasattr(handle, "get_color"):
                text.set_color(handle.get_color())
            elif hasattr(handle, "get_facecolor"):
                fc = handle.get_facecolor()
                if isinstance(fc, tuple):
                    text.set_color(fc)
                elif hasattr(fc, "ndim") and fc.ndim == 2 and len(fc) > 0:
                    text.set_color(fc[0])
                elif hasattr(fc, "ndim") and fc.ndim == 1 and len(fc) in (3, 4):
                    text.set_color(fc)
    return leg


def style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)


def render_plot_with_download(fig, filename_base: str):
    st.pyplot(fig, use_container_width=True)
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=600)
    png_bytes = buf_png.getvalue()

    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg")
    svg_bytes = buf_svg.getvalue()

    EXPORT_REGISTRY[f"figures_png/{filename_base}.png"] = png_bytes
    EXPORT_REGISTRY[f"figures_svg/{filename_base}.svg"] = svg_bytes

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ PNG (600 DPI)", png_bytes, f"{filename_base}.png", "image/png", key=f"png_{filename_base}")
    c2.download_button("⬇️ SVG (Vector)", svg_bytes, f"{filename_base}.svg", "image/svg+xml", key=f"svg_{filename_base}")
    plt.close(fig)


def plot_gradient_membrane_profile(x, y, ylabel, title, cmap_name, filename_base, hline=None, hline_label=None):
    fig = plt.figure(figsize=(6.2, 4.6))
    gs = GridSpec(2, 2, height_ratios=[3.6, 0.9], width_ratios=[20, 1], hspace=0.08, wspace=0.06)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[:, 1])

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vmin, vmax = float(np.nanmin(y)), float(np.nanmax(y))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6
    norm = plt.Normalize(vmin, vmax)
    lc = LineCollection(segs, cmap=cmap_name, norm=norm, linewidth=lw + 1.0)
    lc.set_array(0.5 * (y[:-1] + y[1:]))
    ax_top.add_collection(lc)
    ax_top.set_xlim(0.0, 1.0)
    pad = 0.1 * (vmax - vmin)
    ax_top.set_ylim(vmin - pad, vmax + pad)
    if hline is not None:
        ax_top.axhline(hline, color="black", ls="--", lw=1.0, label=hline_label)
        if hline_label:
            ax_top.legend(loc="best")
    ax_top.set_ylabel(ylabel)
    ax_top.set_title(title)
    ax_top.tick_params(labelbottom=False)
    if show_grid:
        ax_top.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)

    strip = np.tile(y, (12, 1))
    im = ax_bot.imshow(strip, aspect="auto", cmap=cmap_name, norm=norm, extent=[0.0, 1.0, 0.0, 1.0])
    ax_bot.set_yticks([])
    ax_bot.set_xlabel(r"Normalized Coordinate ($x / L$)  [Fuel $\rightarrow$ Air/Steam]")
    ax_bot.text(0.02, 0.5, "Fuel", va="center", ha="left", fontsize=fs_tick - 1, color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45, edgecolor="none"))
    ax_bot.text(0.98, 0.5, "Air/Steam", va="center", ha="right", fontsize=fs_tick - 1, color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45, edgecolor="none"))
    fig.colorbar(im, cax=cax, label=ylabel)
    render_plot_with_download(fig, filename_base)


def plot_2d_or_3d_map(
    x_arr, y_arr, Z, xlabel, ylabel, zlabel, title, cmap_name,
    plot_3d, filename_base, contour_fmt="%.0f", vcenter_zero=False, extra_zero_contour=False
):
    X, Y = np.meshgrid(x_arr, y_arr)
    z_clean = np.nan_to_num(Z, nan=0.0)
    if vcenter_zero:
        vmax = max(abs(float(np.nanmin(Z))), abs(float(np.nanmax(Z))), 0.05)
        vmin = -vmax
    else:
        vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6

    if not plot_3d:
        fig, ax = plt.subplots(figsize=(6, 4.4))
        cf = ax.contourf(X, Y, Z, levels=20, cmap=cmap_name, vmin=vmin, vmax=vmax)
        if show_contours:
            if extra_zero_contour:
                cs0 = ax.contour(X, Y, Z, levels=[0.0], colors="black", linewidths=2.0)
                ax.clabel(cs0, inline=True, fontsize=fs_tick, fmt="0")
            else:
                cs = ax.contour(X, Y, Z, levels=8, colors="white", linewidths=0.7, alpha=0.85)
                ax.clabel(cs, inline=True, fontsize=fs_tick - 1, fmt=contour_fmt)
        fig.colorbar(cf, ax=ax, label=zlabel)
        style_ax(ax, xlabel, ylabel, title)
        render_plot_with_download(fig, filename_base)
    else:
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, z_clean, cmap=cmap_name, vmin=vmin, vmax=vmax, alpha=0.92, edgecolor="none", antialiased=True)
        z_offset = vmin - 0.15 * (vmax - vmin)
        ax.contourf(X, Y, z_clean, zdir="z", offset=z_offset, levels=16, cmap=cmap_name, vmin=vmin, vmax=vmax, alpha=0.65)
        if extra_zero_contour:
            ax.contour(X, Y, z_clean, zdir="z", offset=z_offset, levels=[0.0], colors="black", linewidths=1.8)
        ax.set_zlim(z_offset, vmax)
        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_zlabel(zlabel, labelpad=6)
        ax.set_title(title)
        ax.view_init(elev=view_elev, azim=view_azim)
        fig.colorbar(surf, ax=ax, shrink=0.68, pad=0.10, label=zlabel)
        render_plot_with_download(fig, f"{filename_base}_3d")


def plot_3d_waterfall(x_vals, series_dict, xlabel, ylabel, zlabel, title, cmap_name, filename_base):
    fig = plt.figure(figsize=(6.6, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    keys = sorted(series_dict.keys())
    cmap = plt.get_cmap(cmap_name)
    z_all = np.concatenate([np.asarray(series_dict[k]) for k in keys])
    z_base = float(np.nanmin(z_all))
    for idx, k in enumerate(keys):
        z_vals = np.asarray(series_dict[k])
        y_vals = np.full_like(x_vals, float(k))
        col = cmap(idx / max(len(keys) - 1, 1))
        ax.plot(x_vals, y_vals, z_vals, lw=lw, color=col)
        ax.plot(x_vals, y_vals, np.full_like(x_vals, z_base), lw=0.5, color=col, alpha=0.25)
        for j in range(0, len(x_vals), max(1, len(x_vals) // 10)):
            ax.plot([x_vals[j], x_vals[j]], [k, k], [z_base, z_vals[j]], lw=0.4, color=col, alpha=0.22)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_zlabel(zlabel, labelpad=6)
    ax.set_title(title)
    ax.view_init(elev=view_elev, azim=view_azim)
    render_plot_with_download(fig, filename_base)


# =============================================================================
# 4. HEADER & WORKING POINT METRICS
# =============================================================================
st.title("⚡ Unified PCEC / PCFC Electro-, Mass-, and Heat-Transfer Model")

mat_tuple = tuple(sorted(mat_params.items()))
res_pt = cached_single_point(
    float(i_work), float(T_work), float(L_work), float(Rp_H), float(Rp_O),
    float(p_H2_f), float(p_H2O_f), float(p_H2O_o), float(c_R_work),
    mat_tuple, solver_mode, use_arcsinh_shift, use_trapping, use_dgm, use_self_heating, float(h_cool)
)
res_ocv_pt = cached_single_point(
    0.0, float(T_work), float(L_work), float(Rp_H), float(Rp_O),
    float(p_H2_f), float(p_H2O_f), float(p_H2O_o), float(c_R_work),
    mat_tuple, "fast", use_arcsinh_shift, use_trapping, use_dgm, False, float(h_cool)
)

flags_txt = []
if use_trapping:
    flags_txt.append("Trapping ON")
if use_dgm:
    flags_txt.append("DGM ON")
if use_self_heating:
    flags_txt.append(f"Self-Heating ON (T_eff={res_pt['T_cell_eff']:.1f} °C)")
flags_str = " | ".join(flags_txt) if flags_txt else "Isothermal / No Trapping"

st.caption(
    f"Electrolyte: **{preset_name}** | Active Solver: **{res_pt['solver_used']}** | Carriers: OH•, V••, O• | {flags_str}"
)

if solver_mode == "bvp" and not res_pt["bvp_converged"]:
    st.warning("1D BVP solver did not converge at this operating point — metrics and plots reflect **Algebraic Fallback**.")
elif solver_mode == "bvp" and res_pt["bvp_mesh_warn"]:
    st.info("1D BVP adaptive mesh approached max_nodes (2000) due to steep boundary layers.")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Mode / Solver", res_pt["mode"].split()[0], delta=res_pt["solver_used"])
m2.metric("Cell Voltage (V)", f"{res_pt['V_cell']:.4f}", delta=f"E_eq = {res_pt['E_eq']:.3f} V")
m3.metric("Faradaic Eff. (%)", f"{res_pt['FE']:.2f}%", delta=f"t_p(ρ) = {res_pt['t_p_rho']*100:.1f}%")
m4.metric("LHV Efficiency (%)", f"{res_pt['eff_LHV']:.2f}%")
m5.metric("Heat Flux q̇ (W cm⁻²)", f"{res_pt['q_dot']:.3f}", delta="Exothermic" if res_pt["q_dot"] > 0 else "Endothermic")
m6.metric("Leakage i_h (A cm⁻²)", f"{res_pt['i_h']:.4f}", delta=f"r_leak = {res_pt['r_leak']:.1f}%")

# =============================================================================
# 5. MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Spatial Profiles (x/L)",
    "2. I–V, Leakage & Thermal",
    "3. Multiparametric Analysis",
    "4. 2D/3D Parametric Maps",
    "5. Literature Benchmarks",
    "6. Novel Insights & Design"
])

mk = "o" if show_markers else None

# -----------------------------------------------------------------------------
# TAB 1: SPATIAL PROFILES ACROSS MEMBRANE THICKNESS
# -----------------------------------------------------------------------------
with tab1:
    st.subheader(f"Spatial Distributions Across Membrane Thickness (T = {res_pt['T_cell_eff']:.1f} °C, i = {i_work:.2f} A cm⁻²)")
    c1, c2 = st.columns(2)
    x = res_pt["x_norm"]

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, res_pt["cp"], lw=lw, marker=mk, ms=ms, label=r"$[\mathrm{OH_O^\bullet}]_L$ (protons)")
        ax.plot(x, res_pt["cv"], lw=lw, marker=mk, ms=ms, label=r"$[\mathrm{V_O^{\bullet\bullet}}]_L$ (vacancies)")
        ax.plot(x, res_pt["ch"], lw=lw, marker=mk, ms=ms, label=r"$[\mathrm{O_O^\bullet}]_L$ (polarons)")
        ax.set_yscale("log")
        style_ax(ax, r"Normalized Coordinate ($x / L$)", "Defect Concentration (formula unit)", "Defect Concentration Profiles")
        style_legend(ax)
        render_plot_with_download(fig, "spatial_defect_profiles")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, res_pt["sig_p"] * 1e3, lw=lw, label=r"$\sigma_p$ (protons)")
        ax.plot(x, res_pt["sig_h"] * 1e3, lw=lw, label=r"$\sigma_h$ (polarons)")
        ax.plot(x, res_pt["sig_v"] * 1e3, lw=lw, label=r"$\sigma_v$ (vacancies)")
        ax.plot(x, res_pt["sig_tot"] * 1e3, lw=lw, ls="--", color="black", label=r"$\sigma_{\mathrm{tot}}$")
        style_ax(ax, r"Normalized Coordinate ($x / L$)", r"Conductivity (mS cm$^{-1}$)", "Partial and Total Conductivities")
        style_legend(ax)
        render_plot_with_download(fig, "spatial_conductivities")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, res_pt["i_h_diff_prof"], lw=lw, color="tab:orange", label=r"Diffusion ($i_{h,\mathrm{diff}}$)")
        ax.plot(x, res_pt["i_h_mig_prof"], lw=lw, color="tab:purple", label=r"Migration ($i_{h,\mathrm{mig}}$)")
        ax.plot(x, res_pt["i_h_diff_prof"] + res_pt["i_h_mig_prof"], lw=lw, ls="--", color="black", label=r"Net leakage ($i_h$)")
        ax.axhline(0.0, color="gray", ls=":", lw=0.8)
        style_ax(ax, r"Normalized Coordinate ($x / L$)", r"Polaron Current Density (A cm$^{-2}$)", "Polaron Leakage Components Across Membrane")
        style_legend(ax)
        render_plot_with_download(fig, "spatial_leakage_components")

        # Gradient-colored Chemical Expansion Strain Profile + 2D Membrane Strip
        delta_eps_chem = res_pt["eps_chem_abs"] - res_ocv_pt["eps_chem_abs"]
        plot_gradient_membrane_profile(
            x, delta_eps_chem,
            r"$\Delta\varepsilon_{\mathrm{chem}}$ vs. OCV (%)",
            "Current-Induced Chemical Expansion Across Membrane",
            "coolwarm", "spatial_chem_expansion", hline=0.0, hline_label="OCV baseline"
        )

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, res_pt["Phi"], lw=lw, color="tab:blue", label=r"Potential ($\Phi$)")
        style_ax(ax, r"Normalized Coordinate ($x / L$)", "Potential (V)", "Electrostatic Potential and Electric Field")
        ax2 = ax.twinx()
        ax2.plot(x, res_pt["E_field"] * 1e-5, lw=lw, ls="--", color="tab:red", label=r"Electric field ($E$)")
        ax2.set_ylabel(r"Electric Field ($10^5$ V m$^{-1}$)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
        render_plot_with_download(fig, "spatial_potential_field")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, np.log10(np.maximum(res_pt["p_O2_int"], 1e-35)), lw=lw, label=r"$\log_{10} p_{\mathrm{O_2}}$")
        ax.plot(x, np.log10(np.maximum(res_pt["p_H2O_int"], 1e-35)), lw=lw, label=r"$\log_{10} p_{\mathrm{H_2O}}$")
        ax.plot(x, np.log10(np.maximum(res_pt["p_H2_int"], 1e-35)), lw=lw, label=r"$\log_{10} p_{\mathrm{H_2}}$")
        ax.axhline(np.log10(pO2_crit_atm), color="crimson", ls=":", lw=1.1, label=f"Critical p(O₂) ({pO2_crit_atm} atm)")
        style_ax(ax, r"Normalized Coordinate ($x / L$)", r"$\log_{10}(p_i\ /\ \mathrm{atm})$", "Internal Gas Partial Pressures")
        style_legend(ax)
        render_plot_with_download(fig, "spatial_internal_pressures")

        # Gradient-colored Local Irreversible Dissipation + 2D Membrane Strip
        plot_gradient_membrane_profile(
            x, res_pt["q_diss_tot"],
            r"Local Dissipation $\dot{q}_{\mathrm{diss}}$ (W cm$^{-3}$)",
            "Local Irreversible Energy Dissipation (Entropy Generation)",
            cmap_heat, "spatial_dissipation"
        )

        # Electrochemical Peclet Number Pe_k = J_mig / J_diff
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, np.clip(res_pt["Pe_p"], -20, 20), lw=lw, label=r"$\mathrm{Pe}_p$ (protons)")
        ax.plot(x, np.clip(res_pt["Pe_h"], -20, 20), lw=lw, color="crimson", label=r"$\mathrm{Pe}_h$ (polarons)")
        ax.plot(x, np.clip(res_pt["Pe_v"], -20, 20), lw=lw, color="tab:green", label=r"$\mathrm{Pe}_v$ (vacancies)")
        ax.axhline(0.0, color="black", ls=":", lw=0.8)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label=r"$|\mathrm{Pe}| = 1$")
        ax.axhline(-1.0, color="gray", ls="--", lw=0.8)
        style_ax(ax, r"Normalized Coordinate ($x / L$)", r"Electrochemical Peclet ($J_{\mathrm{mig}} / J_{\mathrm{diff}}$)", "Migration vs. Diffusion Dominance (Peclet Ratio)")
        style_legend(ax)
        render_plot_with_download(fig, "spatial_peclet")

# -----------------------------------------------------------------------------
# TAB 2: I-V, LEAKAGE, AND THERMAL CHARACTERISTICS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader(f"Electrochemical, Leakage, and Thermal Characteristics (T_in = {T_work} °C)")
    col_i1, col_i2, col_i3 = st.columns(3)
    i_min = col_i1.number_input("Min Current Density (PCEC) (A cm⁻²)", -3.0, -0.1, -1.5, 0.1)
    i_max = col_i2.number_input("Max Current Density (PCFC) (A cm⁻²)", 0.1, 3.0, 1.0, 0.1)
    n_pts = col_i3.slider("Sweep Points", 20, 80, 41)

    df_i = cached_iv_sweep(
        float(i_min), float(i_max), int(n_pts),
        float(T_work), float(L_work), float(Rp_H), float(Rp_O),
        float(p_H2_f), float(p_H2O_f), float(p_H2O_o), float(c_R_work),
        mat_tuple, solver_mode, use_arcsinh_shift, use_trapping, use_dgm, use_self_heating, float(h_cool)
    )
    EXPORT_REGISTRY["data_csv/iv_sweep_data.csv"] = df_i.to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["V_cell"], lw=lw, marker=mk, ms=ms, color="navy", label=r"$V_{\mathrm{cell}}$")
        ax.axvline(0.0, color="black", ls=":", lw=1.0)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", "Cell Voltage (V)", "Polarization Curve and Power Density")
        ax2 = ax.twinx()
        df_fc = df_i[df_i["i"] >= 0]
        ax2.plot(df_fc["i"], df_fc["P_elec"], lw=lw, ls="--", color="crimson", label="PCFC Power Density")
        ax2.set_ylabel(r"Power Density (W cm$^{-2}$)", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
        render_plot_with_download(fig, "iv_power_curve")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["i_p"], lw=lw, label=r"$i_p$ (protons)")
        ax.plot(df_i["i"], df_i["i_h"], lw=lw, label=r"$i_h$ (polarons / leakage)")
        ax.plot(df_i["i"], df_i["i_v"], lw=lw, label=r"$i_v$ (vacancies)")
        ax.axhline(0, color="black", ls=":", lw=0.8)
        ax.axvline(0, color="black", ls=":", lw=0.8)
        style_ax(ax, r"External Current Density (A cm$^{-2}$)", r"Partial Current Density (A cm$^{-2}$)", "Partial Current Densities")
        style_legend(ax)
        render_plot_with_download(fig, "iv_partial_currents")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["t_p"] * 100.0, lw=lw, color="tab:blue", label=r"$t_p^\sigma$ (mean cond.)")
        ax.plot(df_i["i"], df_i["t_p_rho"] * 100.0, lw=lw, ls="--", color="navy", label=r"$t_p^\rho$ (int. resist.)")
        ax.plot(df_i["i"], df_i["t_h"] * 100.0, lw=lw, color="crimson", label=r"$t_h^\sigma$ (polarons)")
        ax.plot(df_i["i"], df_i["t_v"] * 100.0, lw=lw, color="tab:green", label=r"$t_v^\sigma$ (vacancies)")
        ax.axvline(0.0, color="black", ls=":", lw=0.8)
        ax.set_ylim(0, 105)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", "Transport Number (%)", r"Transport Numbers ($t_p^\sigma$ vs. $t_p^\rho$)")
        style_legend(ax)
        render_plot_with_download(fig, "iv_transport_numbers")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["ASR_ohm"], lw=lw, marker=mk, ms=ms, color="darkviolet", label=r"$\mathrm{ASR}_{\mathrm{ohm}}$")
        ax.axvline(0.0, color="black", ls=":", lw=0.8)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", r"Membrane Ohmic ASR ($\Omega$ cm$^2$)", "Current-Dependent Membrane Ohmic Resistance")
        style_legend(ax)
        render_plot_with_download(fig, "iv_ASR_ohm")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        df_ec = df_i[df_i["i"] < -0.05]
        df_fc_p = df_i[df_i["i"] > 0.05]
        ax.plot(df_ec["i"], np.clip(df_ec["eff_LHV"], 0, 105), lw=lw, color="teal", label="PCEC Efficiency (LHV)")
        ax.plot(df_fc_p["i"], np.clip(df_fc_p["eff_LHV"], 0, 105), lw=lw, color="purple", label="PCFC Efficiency (LHV)")
        ax.set_ylim(0, 105)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", "LHV Energy Efficiency (%)", "Energy Conversion Efficiency (LHV)")
        style_legend(ax)
        render_plot_with_download(fig, "iv_energy_efficiency")

        # Stacked Area Chart: Relative Overpotential Shares (%) (Niu 2025 Fig. 7b)
        df_nz = df_i[df_i["i"].abs() > 0.05].copy()
        eta_ohm_abs = (df_nz["i"].abs() * df_nz["ASR_ohm"]).values
        eta_h_abs = df_nz["eta_H"].abs().values
        eta_o_abs = df_nz["eta_O"].abs().values
        eta_c_abs = (df_nz["eta_conc_H"].abs() + df_nz["eta_conc_O"].abs()).values
        tot_eta = np.maximum(eta_ohm_abs + eta_h_abs + eta_o_abs + eta_c_abs, 1e-9)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.stackplot(
            df_nz["i"],
            100.0 * eta_ohm_abs / tot_eta,
            100.0 * eta_h_abs / tot_eta,
            100.0 * eta_o_abs / tot_eta,
            100.0 * eta_c_abs / tot_eta,
            labels=[r"Ohmic ($\eta_{\mathrm{ohm}}$)", r"Fuel act. ($\eta_{\mathrm{H_2}}$)", r"Air act. ($\eta_{\mathrm{O_2}}$)", r"Conc. ($\eta_{\mathrm{conc}}$)"],
            colors=["#3b528b", "#21918c", "#fdae61", "#d7191c"],
            alpha=0.85
        )
        ax.axvline(0.0, color="black", ls=":", lw=0.9)
        ax.set_ylim(0, 100)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", "Relative Loss Share (%)", "Relative Overpotential Breakdown (Stacked Area)")
        ax.legend(loc="lower center", ncol=2)
        render_plot_with_download(fig, "iv_loss_breakdown_pct")

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["dPhi"], lw=lw, label=r"Membrane ($\Delta\Phi$)")
        ax.plot(df_i["i"], -df_i["eta_H"], lw=lw, label=r"Fuel electrode ($-\eta_{\mathrm{H_2}}$)")
        ax.plot(df_i["i"], -df_i["eta_O"], lw=lw, ls="--", label=r"Air electrode ($-\eta_{\mathrm{O_2}}$)")
        if use_dgm:
            ax.plot(df_i["i"], -(df_i["eta_conc_H"] + df_i["eta_conc_O"]), lw=lw, ls=":", color="crimson", label=r"Concentration ($-\eta_{\mathrm{conc}}$)")
        ax.axhline(0, color="black", ls=":", lw=0.8)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", "Voltage Component (V)", "Membrane and Electrode Overpotentials")
        style_legend(ax)
        render_plot_with_download(fig, "iv_voltage_losses")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["i_h"], lw=lw, color="crimson", label=r"Leakage current ($i_h$)")
        ax.axhline(0.0, color="black", ls=":", lw=0.8)
        ax.axvline(0.0, color="black", ls=":", lw=0.8)
        style_ax(ax, r"External Current Density (A cm$^{-2}$)", r"Leakage Current Density (A cm$^{-2}$)", "Electronic Leakage Current and Leakage Fraction")
        ax2 = ax.twinx()
        ax2.plot(df_i["i"], df_i["r_leak"], lw=lw, ls="--", color="darkorange", label=r"Leakage fraction ($r_{\mathrm{leak}}$)")
        ax2.set_ylabel("Leakage Fraction (%)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
        render_plot_with_download(fig, "iv_leakage_analysis")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_ec["i"], np.clip(df_ec["FE"], 0, 105), lw=lw, color="tab:blue", label="FE (PCEC)")
        ax.plot(df_fc_p["i"], np.clip(df_fc_p["FE"], 0, 105), lw=lw, color="tab:green", label="FE (PCFC)")
        ax.axvline(0.0, color="black", ls=":", lw=0.8)
        ax.set_ylim(0, 105)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", "Faradaic Efficiency (%)", "Faradaic Efficiency in PCEC and PCFC Modes")
        style_legend(ax)
        render_plot_with_download(fig, "iv_FE")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["p_O2_max"], lw=lw, marker=mk, ms=ms, color="firebrick", label=r"$\max\,p_{\mathrm{O_2, int}}$")
        ax.axhline(pO2_crit_atm, color="black", ls="--", lw=1.0, label=f"Critical threshold ({pO2_crit_atm} atm)")
        ax.set_yscale("log")
        style_ax(ax, r"Current Density (A cm$^{-2}$)", r"Peak Internal $p_{\mathrm{O_2}}$ (atm)", "Peak Internal Oxygen Pressure (Delamination Risk)")
        style_legend(ax)
        render_plot_with_download(fig, "iv_peak_pO2")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["q_dot"], lw=lw, color="darkorange", label=r"Heat flux ($\dot{q}$)")
        ax.axhline(0.0, color="black", ls="--", lw=1.0, label=r"Thermoneutral ($\dot{q}=0$)")
        ax.fill_between(df_i["i"], df_i["q_dot"], 0, where=(df_i["q_dot"] > 0), color="red", alpha=0.12)
        ax.fill_between(df_i["i"], df_i["q_dot"], 0, where=(df_i["q_dot"] < 0), color="blue", alpha=0.12)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", r"Heat Flux ($\mathrm{W\ cm^{-2}}$)", "Thermal Balance (Endothermic / Exothermic)")
        style_legend(ax)
        render_plot_with_download(fig, "iv_heat_balance")

        # Effective Conductivities vs Cell Voltage (Putilov 2020 Fig. 7)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["V_cell"], df_i["sig_p_eff"] * 1e3, lw=lw, color="tab:blue", label=r"$\sigma_p^{\mathrm{eff}}$ (protons)")
        style_ax(ax, "Cell Voltage (V)", r"Effective Proton Cond. (mS cm$^{-1}$)", r"Effective Partial Conductivities vs. Cell Voltage")
        ax2 = ax.twinx()
        ax2.plot(df_i["V_cell"], df_i["sig_h_eff"] * 1e3, lw=lw, ls="--", color="crimson", label=r"$\sigma_h^{\mathrm{eff}}$ (polarons)")
        ax2.set_ylabel(r"Effective Polaron Cond. (mS cm$^{-1}$)", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
        render_plot_with_download(fig, "iv_putilov_eff_cond")

    st.download_button("📥 Download I–V Sweep Data (CSV)", EXPORT_REGISTRY["data_csv/iv_sweep_data.csv"], "iv_sweep_data.csv", "text/csv")

# -----------------------------------------------------------------------------
# TAB 3: MULTIPARAMETRIC ANALYSIS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Multiparametric Sensitivity Analysis")

    PARAM_OPTIONS = {
        "Temperature (°C)": {"key": "T_C", "min": 450.0, "max": 750.0, "def_min": 500.0, "def_max": 700.0, "unit": "°C", "fmt": "{:.0f} °C"},
        "Electrolyte thickness (µm)": {"key": "L_um", "min": 2.0, "max": 200.0, "def_min": 5.0, "def_max": 50.0, "unit": "µm", "fmt": "{:.1f} µm"},
        "Acceptor dopant concentration (-)": {"key": "c_R", "min": 0.10, "max": 0.30, "def_min": 0.10, "def_max": 0.30, "unit": "-", "fmt": "{:.2f}"},
        "Fuel electrode p(H₂) (atm)": {"key": "p_H2_f", "min": 0.10, "max": 0.99, "def_min": 0.50, "def_max": 0.98, "unit": "atm", "fmt": "{:.2f} atm"},
        "Air electrode p(H₂O) (atm)": {"key": "p_H2O_o", "min": 0.02, "max": 0.95, "def_min": 0.10, "def_max": 0.90, "unit": "atm", "fmt": "{:.2f} atm"},
        "Electrode Rp (Ω cm²)": {"key": "Rp", "min": 0.0, "max": 1.0, "def_min": 0.02, "def_max": 0.20, "unit": "Ω cm²", "fmt": "{:.2f} Ω cm²"},
    }

    p_col1, p_col2, p_col3, p_col4 = st.columns([2, 1, 1, 1])
    var_label = p_col1.selectbox("Select Parameter to Vary", list(PARAM_OPTIONS.keys()))
    p_cfg = PARAM_OPTIONS[var_label]
    v_min = p_col2.number_input("Min", p_cfg["min"], p_cfg["max"], p_cfg["def_min"])
    v_max = p_col3.number_input("Max", p_cfg["min"], p_cfg["max"], p_cfg["def_max"])
    n_vals = p_col4.slider("Number of Steps", 3, 9, 5)
    mp_3d_mode = st.checkbox("Show Families of Curves as 3D Waterfall / Cascade", value=False)

    if v_max <= v_min:
        st.error("Max value must be greater than Min value.")
    else:
        var_values, multi_iv, df_sum = cached_multiparametric(
            p_cfg["key"], float(v_min), float(v_max), int(n_vals),
            float(i_min), float(i_max), float(i_work),
            float(T_work), float(L_work), float(Rp_H), float(Rp_O),
            float(p_H2_f), float(p_H2O_f), float(p_H2O_o), float(c_R_work),
            binary_fuel, mat_tuple, use_arcsinh_shift, use_trapping, use_dgm, use_self_heating, float(h_cool)
        )
        EXPORT_REGISTRY["data_csv/multiparametric_summary.csv"] = df_sum.to_csv(index=False).encode("utf-8")

        cmap = plt.get_cmap(cmap_multi)
        colors = [cmap(k / max(len(var_values) - 1, 1)) for k in range(len(var_values))]
        leg_title_use = custom_legend_title.strip() if custom_legend_title.strip() else var_label

        st.markdown("### Families of Curves vs. Current Density")
        c1, c2 = st.columns(2)
        i_axis_mp = multi_iv[float(var_values[0])]["i"].values
        with c1:
            if mp_3d_mode:
                plot_3d_waterfall(
                    i_axis_mp, {float(v): multi_iv[float(v)]["V_cell"].values for v in var_values},
                    r"Current Density (A cm$^{-2}$)", var_label, "Cell Voltage (V)",
                    f"3D Waterfall: Polarization vs. {var_label}", cmap_multi, "mp_family_IV_3d"
                )
                plot_3d_waterfall(
                    i_axis_mp, {float(v): multi_iv[float(v)]["i_h"].values for v in var_values},
                    r"Current Density (A cm$^{-2}$)", var_label, r"Leakage $i_h$ (A cm$^{-2}$)",
                    f"3D Waterfall: Leakage vs. {var_label}", cmap_multi, "mp_family_leakage_3d"
                )
            else:
                fig, ax = plt.subplots(figsize=(6, 4.2))
                for idx, val in enumerate(var_values):
                    df_v = multi_iv[float(val)]
                    ax.plot(df_v["i"], df_v["V_cell"], lw=lw, marker=mk, ms=ms, color=colors[idx], label=p_cfg["fmt"].format(val))
                ax.axvline(0, color="black", ls=":", lw=0.8)
                style_ax(ax, r"Current Density (A cm$^{-2}$)", "Cell Voltage (V)", f"Polarization Curves vs. {var_label}")
                style_legend(ax, title=leg_title_use)
                render_plot_with_download(fig, "mp_family_IV")

                fig, ax = plt.subplots(figsize=(6, 4.2))
                for idx, val in enumerate(var_values):
                    df_v = multi_iv[float(val)]
                    ax.plot(df_v["i"], df_v["i_h"], lw=lw, marker=mk, ms=ms, color=colors[idx], label=p_cfg["fmt"].format(val))
                ax.axhline(0, color="black", ls=":", lw=0.8)
                ax.axvline(0, color="black", ls=":", lw=0.8)
                style_ax(ax, r"External Current Density (A cm$^{-2}$)", r"Leakage Current Density (A cm$^{-2}$)", f"Electronic Leakage vs. {var_label}")
                style_legend(ax, title=leg_title_use)
                render_plot_with_download(fig, "mp_family_leakage")

        with c2:
            if mp_3d_mode:
                plot_3d_waterfall(
                    i_axis_mp, {float(v): np.clip(multi_iv[float(v)]["FE"].values, 0, 100) for v in var_values},
                    r"Current Density (A cm$^{-2}$)", var_label, "Faradaic Efficiency (%)",
                    f"3D Waterfall: Faradaic Efficiency vs. {var_label}", cmap_multi, "mp_family_FE_3d"
                )
                fc_mask = i_axis_mp >= 0
                plot_3d_waterfall(
                    i_axis_mp[fc_mask], {float(v): multi_iv[float(v)]["P_elec"].values[fc_mask] for v in var_values},
                    r"Current Density (A cm$^{-2}$)", var_label, r"Power Density (W cm$^{-2}$)",
                    f"3D Waterfall: PCFC Power vs. {var_label}", cmap_multi, "mp_family_power_3d"
                )
            else:
                fig, ax = plt.subplots(figsize=(6, 4.2))
                for idx, val in enumerate(var_values):
                    df_v = multi_iv[float(val)]
                    df_ec_v = df_v[df_v["i"] < -0.05]
                    df_fc_v = df_v[df_v["i"] > 0.05]
                    ax.plot(df_ec_v["i"], np.clip(df_ec_v["FE"], 0, 105), lw=lw, color=colors[idx], label=p_cfg["fmt"].format(val))
                    ax.plot(df_fc_v["i"], np.clip(df_fc_v["FE"], 0, 105), lw=lw, ls="--", color=colors[idx])
                ax.set_ylim(0, 105)
                style_ax(ax, r"Current Density (A cm$^{-2}$)", "Faradaic Efficiency (%)", f"Faradaic Efficiency vs. {var_label}")
                style_legend(ax, title=leg_title_use)
                render_plot_with_download(fig, "mp_family_FE")

                fig, ax = plt.subplots(figsize=(6, 4.2))
                for idx, val in enumerate(var_values):
                    df_v = multi_iv[float(val)]
                    df_fc_v = df_v[df_v["i"] >= 0]
                    ax.plot(df_fc_v["i"], df_fc_v["P_elec"], lw=lw, marker=mk, ms=ms, color=colors[idx], label=p_cfg["fmt"].format(val))
                style_ax(ax, r"Current Density (A cm$^{-2}$)", r"Power Density (W cm$^{-2}$)", f"PCFC Power Density vs. {var_label}")
                style_legend(ax, title=leg_title_use)
                render_plot_with_download(fig, "mp_family_power")

        st.markdown(f"### Key Performance Metrics vs. {var_label}")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4.0))
            ax.plot(df_sum["param"], df_sum["P_max"], lw=lw, marker="o", ms=ms, color="crimson", label=r"$P_{\max}$")
            style_ax(ax, var_label, r"Max Power Density (W cm$^{-2}$)", "Maximum PCFC Power Density")
            ax2 = ax.twinx()
            ax2.plot(df_sum["param"], df_sum["i_Pmax"], lw=lw, ls="--", marker="s", ms=ms, color="navy", label=r"$i(P_{\max})$")
            ax2.set_ylabel(r"Current Density at $P_{\max}$ (A cm$^{-2}$)", color="navy")
            ax2.tick_params(axis="y", labelcolor="navy", direction="in")
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
            render_plot_with_download(fig, "mp_Pmax")

            fig, ax = plt.subplots(figsize=(6, 4.0))
            ax.plot(df_sum["param"], df_sum["J_H2_Nml_min"], lw=lw, marker="o", ms=ms, color="tab:blue", label=r"$J_{\mathrm{H_2}}$ (volumetric)")
            style_ax(ax, var_label, r"$\mathrm{H_2}$ Flux (Nml cm$^{-2}$ min$^{-1}$)", f"Hydrogen Flux at i = {i_work:.2f} A cm⁻²")
            ax2 = ax.twinx()
            ax2.plot(df_sum["param"], df_sum["J_H2"] * 1e6, lw=lw, ls="--", marker="^", ms=ms, color="teal", label=r"$J_{\mathrm{H_2}}$ (molar)")
            ax2.set_ylabel(r"Molar $\mathrm{H_2}$ Flux ($\mu$mol cm$^{-2}$ s$^{-1}$)", color="teal")
            ax2.tick_params(axis="y", labelcolor="teal", direction="in")
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
            render_plot_with_download(fig, "mp_H2_flux")

            fig, ax = plt.subplots(figsize=(6, 4.0))
            ax.plot(df_sum["param"], df_sum["E_eq"], lw=lw, marker="o", ms=ms, color="navy", label=r"$E_{\mathrm{eq}}$")
            ax.plot(df_sum["param"], df_sum["OCV"], lw=lw, marker="s", ms=ms, color="tab:green", label="OCV")
            style_ax(ax, var_label, "Potential (V)", "Equilibrium Potential, OCV, and OCV Loss")
            ax2 = ax.twinx()
            ax2.plot(df_sum["param"], df_sum["RdOCV_mV"], lw=lw, ls="--", marker="^", ms=ms, color="crimson", label="RdOCV")
            ax2.set_ylabel("OCV Reduction, RdOCV (mV)", color="crimson")
            ax2.tick_params(axis="y", labelcolor="crimson", direction="in")
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
            render_plot_with_download(fig, "mp_OCV_RdOCV")

            fig, ax = plt.subplots(figsize=(6, 4.0))
            ax.plot(df_sum["param"], df_sum["ASR_ohm"], lw=lw, marker="o", ms=ms, color="darkviolet", label=r"$\mathrm{ASR}_{\mathrm{ohm}}$")
            style_ax(ax, var_label, r"Membrane Ohmic ASR ($\Omega$ cm$^2$)", "Ohmic Resistance and Proton Transport Number")
            ax2 = ax.twinx()
            ax2.plot(df_sum["param"], df_sum["t_p_wk"], lw=lw, ls="--", marker="s", ms=ms, color="tab:blue", label=r"Mean $t_p^\sigma$")
            ax2.plot(df_sum["param"], df_sum["t_p_rho_wk"], lw=lw, ls=":", marker="^", ms=ms, color="navy", label=r"Eff. $t_p^\rho$")
            ax2.set_ylabel("Proton Transport Number (%)", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue", direction="in")
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
            render_plot_with_download(fig, "mp_ASR_tp")

        with c2:
            fig, ax = plt.subplots(figsize=(6, 4.0))
            has_tn = df_sum["i_TN"].notna().any()
            if has_tn:
                ax.plot(df_sum["param"], df_sum["i_TN"], lw=lw, marker="o", ms=ms, color="darkorange", label=r"$i_{\mathrm{TN}}$ (PCEC)")
                style_ax(ax, var_label, r"Thermoneutral Current Density (A cm$^{-2}$)", "Thermoneutral Operating Point (PCEC)")
                ax2 = ax.twinx()
                ax2.plot(df_sum["param"], df_sum["V_TN_pt"], lw=lw, ls="--", marker="s", ms=ms, color="black", label=r"$V_{\mathrm{TN}}^{\mathrm{act}}$")
                ax2.set_ylabel("Actual Thermoneutral Voltage (V)", color="black")
                ax2.tick_params(axis="y", labelcolor="black", direction="in")
                l1, lb1 = ax.get_legend_handles_labels()
                l2, lb2 = ax2.get_legend_handles_labels()
                style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
            else:
                ax.plot(df_sum["param"], df_sum["V_TN_act_wk"], lw=lw, marker="s", ms=ms, color="black", label=r"$V_{\mathrm{TN}}^{\mathrm{act}}$ (at $i_{\mathrm{work}}$)")
                style_ax(ax, var_label, "Actual Thermoneutral Voltage (V)", "Actual Thermoneutral Voltage (i_TN > 100 A/cm²)")
                style_legend(ax)
            render_plot_with_download(fig, "mp_thermoneutral")

            fig, ax = plt.subplots(figsize=(6, 4.0))
            ax.plot(df_sum["param"], df_sum["FE_wk"], lw=lw, marker="o", ms=ms, color="tab:blue", label="Faradaic Efficiency (FE)")
            ax.plot(df_sum["param"], df_sum["eff_wk"], lw=lw, marker="s", ms=ms, color="tab:green", label="LHV Energy Efficiency")
            ax.set_ylim(0, 105)
            style_ax(ax, var_label, "Efficiency (%)", f"Faradaic and Energy Efficiency (i = {i_work:.2f} A cm⁻²)")
            style_legend(ax)
            render_plot_with_download(fig, "mp_FE_eff")

            fig, ax = plt.subplots(figsize=(6, 4.0))
            ax.plot(df_sum["param"], df_sum["i_leak_OCV"] * 1e3, lw=lw, marker="o", ms=ms, color="crimson", label=r"$|i_{\mathrm{leak, OCV}}|$")
            style_ax(ax, var_label, r"OCV Leakage Current (mA cm$^{-2}$)", "OCV Electronic Leakage and Transition Current")
            ax2 = ax.twinx()
            ax2.plot(df_sum["param"], df_sum["i_trans"] * 1e3, lw=lw, ls="--", marker="^", ms=ms, color="navy", label=r"$i_{\mathrm{PCFC}\to\mathrm{PCEC}}$")
            ax2.set_ylabel(r"Transition Current Density (mA cm$^{-2}$)", color="navy")
            ax2.tick_params(axis="y", labelcolor="navy", direction="in")
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
            render_plot_with_download(fig, "mp_leakage_OCV_trans")

            fig, ax = plt.subplots(figsize=(6, 4.0))
            ax.plot(df_sum["param"], df_sum["q_dot_wk"], lw=lw, marker="o", ms=ms, color="darkorange", label=r"Heat flux ($\dot{q}$)")
            ax.axhline(0.0, color="gray", ls=":", lw=0.8)
            style_ax(ax, var_label, r"Heat Flux (W cm$^{-2}$)", f"Heat Flux and Cell Voltage (i = {i_work:.2f} A cm⁻²)")
            ax2 = ax.twinx()
            ax2.plot(df_sum["param"], df_sum["V_cell_wk"], lw=lw, ls="--", marker="s", ms=ms, color="navy", label=r"$V_{\mathrm{cell}}$")
            ax2.set_ylabel("Cell Voltage (V)", color="navy")
            ax2.tick_params(axis="y", labelcolor="navy", direction="in")
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
            render_plot_with_download(fig, "mp_heat_voltage")

        st.download_button("📥 Download Multiparametric Summary (CSV)", EXPORT_REGISTRY["data_csv/multiparametric_summary.csv"], "multiparametric_summary.csv", "text/csv")

# -----------------------------------------------------------------------------
# TAB 4: 2D / 3D PARAMETRIC MAPS
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("2D Contour & 3D Surface Parametric Maps")
    hc1, hc2 = st.columns([2, 1])
    res_hm = hc1.slider("Grid Resolution (N × N)", 12, 35, 20)
    hm_3d = hc2.checkbox("Render as 3D Surfaces (+ bottom contour projection)", value=False)

    T_hm, i_hm_ec, L_hm, x_hm, FE_map, Q_map, Leak_map, V_map, pO2_max_map, Eff_L_map, pO2_map = cached_heatmaps(
        int(res_hm), float(T_work), float(L_work), float(Rp_H), float(Rp_O),
        float(p_H2_f), float(p_H2O_f), float(p_H2O_o), float(c_R_work),
        mat_tuple, use_arcsinh_shift, use_trapping, use_dgm, use_self_heating, float(h_cool)
    )

    c1, c2 = st.columns(2)
    with c1:
        plot_2d_or_3d_map(
            i_hm_ec, T_hm, FE_map,
            r"PCEC Current Density (A cm$^{-2}$)", "Temperature (°C)", "Faradaic Efficiency (%)",
            "Faradaic Efficiency Map", cmap_heat, hm_3d, "heatmap_FE_Ti", contour_fmt="%.0f%%"
        )
        plot_2d_or_3d_map(
            i_hm_ec, T_hm, Leak_map,
            r"PCEC Current Density (A cm$^{-2}$)", "Temperature (°C)", r"Electronic Leakage $|i_h|$ (mA cm$^{-2}$)",
            "Electronic Leakage Current Map", cmap_heat, hm_3d, "heatmap_leakage_Ti", contour_fmt="%.0f"
        )
        plot_2d_or_3d_map(
            i_hm_ec, x_hm, pO2_map,
            r"PCEC Current Density (A cm$^{-2}$)", r"Normalized Coordinate ($x / L$)", r"$\log_{10}(p_{\mathrm{O_2}}\ /\ \mathrm{atm})$",
            f"Internal Oxygen Pressure Map ({T_work} °C)", cmap_heat, hm_3d, "heatmap_pO2_xi", contour_fmt="%.1f"
        )

    with c2:
        plot_2d_or_3d_map(
            i_hm_ec, T_hm, Q_map,
            r"PCEC Current Density (A cm$^{-2}$)", "Temperature (°C)", r"Heat Flux (W cm$^{-2}$)",
            "Thermal Balance and Thermoneutral Boundary", "RdBu_r", hm_3d, "heatmap_Q_Ti",
            vcenter_zero=True, extra_zero_contour=True
        )
        plot_2d_or_3d_map(
            i_hm_ec, L_hm, Eff_L_map,
            r"PCEC Current Density (A cm$^{-2}$)", "Electrolyte Thickness (µm)", "LHV Energy Efficiency (%)",
            f"Optimum Thickness Map ({T_work} °C)", cmap_heat, hm_3d, "heatmap_Eff_Li", contour_fmt="%.0f%%"
        )
        plot_2d_or_3d_map(
            i_hm_ec, T_hm, V_map,
            r"PCEC Current Density (A cm$^{-2}$)", "Temperature (°C)", "Cell Voltage (V)",
            "PCEC Cell Voltage Surface V(T, i)", cmap_heat, hm_3d, "heatmap_V_Ti", contour_fmt="%.2f V"
        )

# -----------------------------------------------------------------------------
# TAB 5: LITERATURE BENCHMARKS (RESTREPO 2026, THATTE 2024, PUTILOV 2020, MORANTI 2026)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Literature Benchmark Diagnostics & 1+1D Channel Profile")
    lit_3d = st.checkbox("Show I₀(T, p_O₂) as 3D Surface", value=False)

    (
        T_i0, pO2_i0, I0_map,
        T_lopt, Lopt_ec, eff_ec_opt, Lopt_fc, eff_fc_opt,
        T_tn, tn_data,
        df_putilov,
        i_trap, v_tr, fe_tr, v_no, fe_no
    ) = cached_literature_insights(
        float(T_work), float(L_work), float(Rp_H), float(Rp_O),
        float(p_H2_f), float(p_H2O_f), float(p_H2O_o), float(c_R_work),
        mat_tuple, use_arcsinh_shift, use_trapping, use_dgm
    )

    c1, c2 = st.columns(2)
    with c1:
        # 1. Zero Membrane Voltage Drop Current I_0(T, p_O2) (Restrepo 2026 Fig. 14)
        plot_2d_or_3d_map(
            pO2_i0, T_i0, I0_map,
            r"Positrode $p_{\mathrm{O_2}}$ (atm)", "Temperature (°C)", r"$|I_0|$ at $\Delta\Phi=0$ (A cm$^{-2}$)",
            f"Zero Membrane Voltage Drop Current I₀ (L = {L_work:.0f} µm)", cmap_heat, lit_3d, "lit_I0_zero_dPhi", contour_fmt="%.2f"
        )

        # 3. Actual Thermoneutral Voltage Map with Current Envelopes (Thatte 2024 Fig. 7a)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        env_cols = {0.20: "tab:red", 0.50: "tab:orange", 0.90: "tab:blue"}
        for x_st, (v_lo, v_mid, v_hi) in tn_data.items():
            col = env_cols[x_st]
            ax.fill_between(T_tn, v_lo, v_hi, color=col, alpha=0.18)
            ax.plot(T_tn, v_mid, lw=lw, color=col, label=f"Steam {int(x_st*100)}% (i = -0.4..-2.0 A cm⁻²)")
        ax.axhline(abs(DH_W) / (2.0 * F_CONST), color="black", ls="--", lw=1.2, label=r"$V_{\mathrm{TN}}^{\mathrm{theo}} = 1.286$ V")
        style_ax(ax, "Temperature (°C)", r"Actual Thermoneutral Voltage $V_{\mathrm{TN}}^{\mathrm{act}}$ (V)", "Thermoneutral Voltage Envelope (Thatte 2024)")
        style_legend(ax)
        render_plot_with_download(fig, "lit_thatte_VTN_map")

        # 5. Trapping vs No Trapping (Moranti 2026 Fig. 7)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(i_trap, v_tr, lw=lw, color="navy", label=r"$V_{\mathrm{cell}}$ (Trapping)")
        ax.plot(i_trap, v_no, lw=lw, ls="--", color="tab:cyan", label=r"$V_{\mathrm{cell}}$ (No Trapping)")
        style_ax(ax, r"PCEC Current Density (A cm$^{-2}$)", "Cell Voltage (V)", "Effect of Polaron Trapping (Moranti 2026)")
        ax2 = ax.twinx()
        ax2.plot(i_trap, fe_tr, lw=lw, color="crimson", label="FE (Trapping)")
        ax2.plot(i_trap, fe_no, lw=lw, ls="--", color="darkorange", label="FE (No Trapping)")
        ax2.set_ylabel("Faradaic Efficiency (%)", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
        render_plot_with_download(fig, "lit_moranti_trapping")

    with c2:
        # 2. Optimum Membrane Thickness L_opt(T) & Max Energy Efficiency (Restrepo 2026 Fig. 13a)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(T_lopt, Lopt_ec, lw=lw, marker="o", ms=ms, color="navy", label=r"$L_{\mathrm{opt}}$ (PCEC, -0.8 A cm$^{-2}$)")
        ax.plot(T_lopt, Lopt_fc, lw=lw, marker="s", ms=ms, color="teal", label=r"$L_{\mathrm{opt}}$ (PCFC, +0.5 A cm$^{-2}$)")
        style_ax(ax, "Temperature (°C)", "Optimum Membrane Thickness (µm)", "Optimum Thickness & Peak Efficiency (Restrepo 2026)")
        ax2 = ax.twinx()
        ax2.plot(T_lopt, eff_ec_opt, lw=lw, ls="--", color="crimson", label=r"$\eta_{\max}$ (PCEC)")
        ax2.plot(T_lopt, eff_fc_opt, lw=lw, ls=":", color="darkorange", label=r"$\eta_{\max}$ (PCFC)")
        ax2.set_ylabel("Max LHV Efficiency (%)", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
        render_plot_with_download(fig, "lit_restrepo_Lopt")

        # 4. Working Mode (U = 0.7 V) vs OCV Comparison (Putilov 2020 Fig. 11)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_putilov["pH2O_f"], df_putilov["ih_ocv"], lw=lw, marker="o", ms=ms, color="crimson", label=r"$|i_h|$ at OCV")
        ax.plot(df_putilov["pH2O_f"], df_putilov["ih_op"], lw=lw, ls="--", marker="s", ms=ms, color="darkorange", label=r"$|i_h|$ at 0.7 V (PCFC)")
        style_ax(ax, r"Inlet Fuel Humidity $p_{\mathrm{H_2O}}^a$ (atm)", r"Hole Leakage Current (mA cm$^{-2}$)", "Working Mode vs. OCV Comparison (Putilov 2020)")
        ax2 = ax.twinx()
        ax2.plot(df_putilov["pH2O_f"], df_putilov["tp_ocv"], lw=lw, color="navy", label=r"$t_p^\rho$ at OCV")
        ax2.plot(df_putilov["pH2O_f"], df_putilov["tp_op"], lw=lw, ls="--", color="tab:blue", label=r"$t_p^\rho$ at 0.7 V")
        ax2.set_ylabel(r"Effective Transport Number $t_p^\rho$ (%)", color="navy")
        ax2.tick_params(axis="y", labelcolor="navy", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
        render_plot_with_download(fig, "lit_putilov_ocv_vs_op")

        # 6. Quasi-2D (1+1D) Along-Channel Distribution (Thatte 2024 Fig. 6/8)
        ch_c1, ch_c2 = st.columns(2)
        V_ch = ch_c1.number_input("Channel Constant Voltage (V)", 1.05, 1.60, 1.28, 0.02)
        U_ch = ch_c2.slider("Steam Utilization along Channel", 0.10, 0.90, 0.60, 0.05)
        df_ch = cached_channel_1d(
            float(V_ch), float(U_ch), float(T_work), float(L_work), float(Rp_H), float(Rp_O),
            float(p_H2_f), float(p_H2O_f), float(p_H2O_o), float(c_R_work),
            mat_tuple, use_arcsinh_shift, use_trapping, use_dgm
        )
        fig, ax = plt.subplots(figsize=(6, 4.0))
        ax.plot(df_ch["y"], df_ch["FE_loc"], lw=lw, color="tab:blue", label=r"Local $\mathrm{FE}(y)$")
        ax.axhline(0.0, color="black", ls=":", lw=0.8)
        style_ax(ax, r"Normalized Channel Length ($y / L_{\mathrm{ch}}$)", "Local Faradaic Efficiency (%)", f"1+1D Along-Channel PCEC Profile (V={V_ch:.2f} V, U={int(U_ch*100)}%)")
        ax2 = ax.twinx()
        ax2.plot(df_ch["y"], df_ch["i_loc"], lw=lw, ls="--", color="crimson", label=r"Local $i(y)$")
        ax2.set_ylabel(r"Local Current Density (A cm$^{-2}$)", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
        render_plot_with_download(fig, "lit_channel_1p1d")

# -----------------------------------------------------------------------------
# TAB 6: NOVEL RESEARCH INSIGHTS & MATERIAL DESIGN
# -----------------------------------------------------------------------------
with tab6:
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export Peak Data** *(for Complete Dataset table)*")
            
            # Create peaks DataFrame with Fmax and C columns
            peaks_data = []
            for p in deconv_result.peaks:
                char = calculate_peak_characteristics(p)
                peaks_data.append({
                    'Peak_ID': p.id,
                    'Center_tau_s': p.center,
                    'Center_log_tau': p.center_log,
                    'Fmax_Hz': char['fmax_hz'],
                    'C_Farad': char['c_farad'],
                    'Amplitude_Ohm': p.amplitude,
                    'Amplitude_Normalized': p.amplitude_norm,
                    'Sigma_log': p.sigma_log,
                    'FWHM': p.fwhm,
                    'Area_Ohm_s': p.area,
                    'Fraction': p.fraction,
                    'Fraction_Percent': p.fraction_percent,
                    'Source': p.source,
                    'Characteristic_Frequency_Hz': p.get_characteristic_frequency(),
                    'Resistance_Contribution_Ohm': p.get_resistance_contribution()
                })
            
            peaks_df = pd.DataFrame(peaks_data)
            csv_peaks = peaks_df.to_csv(index=False)
            st.download_button(
                "📥 Export Peaks as CSV",
                data=csv_peaks,
                file_name=f"deconvolution_peaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("**Export Fitting Data** *(from Gaussian Deconvolution Result)*")
            
            if deconv_result.fit_y_original is not None:
                fit_values = deconv_result.fit_y_original
            else:
                fit_values = np.zeros_like(deconv_result.x_linear)
                for i, tau in enumerate(deconv_result.x_linear):
                    if deconv_result.use_log_x:
                        log_tau = np.log10(tau)
                    else:
                        log_tau = tau
                    
                    total = 0
                    for peak in deconv_result.peaks:
                        total += peak.amplitude * GaussianModelDeconv.gaussian(
                            log_tau, 1.0, peak.center_log, peak.sigma_log
                        )
                    
                    if deconv_result.baseline_params and deconv_result.baseline_method != 'none':
                        if deconv_result.baseline_method == 'constant':
                            total += deconv_result.baseline_params[0]
                        elif deconv_result.baseline_method == 'linear':
                            total += deconv_result.baseline_params[0] + deconv_result.baseline_params[1] * log_tau
                        elif deconv_result.baseline_method == 'quadratic':
                            total += (deconv_result.baseline_params[0] + 
                                     deconv_result.baseline_params[1] * log_tau +
                                     deconv_result.baseline_params[2] * log_tau**2)
                    
                    fit_values[i] = total
            
            residuals = deconv_result.y_original - fit_values
            
            fit_data = pd.DataFrame({
                'tau_s': deconv_result.x_linear,
                'gamma_tau_Ohm': deconv_result.y_original,
                'gamma_fit_Ohm': fit_values,
                'Residuals_Ohm': residuals
            })
            
            csv_fit = fit_data.to_csv(index=False)
            st.download_button(
                "📥 Export Fitting Data as CSV",
                data=csv_fit,
                file_name=f"deconvolution_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # --- Export Normalized Gaussian Deconvolution Result ---
            st.markdown("**Export Normalized Deconvolution Data** *(Normalized spectrum + Total Fit + Peaks)*")
            
            peaks_sorted_exp = sorted(deconv_result.peaks, key=lambda p: p.id)
            max_amp_exp = max(deconv_result.y_original) if len(deconv_result.y_original) > 0 else 1.0
            x_eval = np.log10(np.maximum(deconv_result.x_linear, 1e-15)) if deconv_result.use_log_x else deconv_result.x_linear
            
            norm_dict = {
                'tau_s': deconv_result.x_linear,
                'gamma_tau_norm': deconv_result.y_original / max_amp_exp,
                'Total_Fit_norm': fit_values / max_amp_exp
            }
            for peak in peaks_sorted_exp:
                y_comp_norm = (peak.amplitude * GaussianModelDeconv.gaussian(
                    x_eval, 1.0, peak.center_log, peak.sigma_log
                )) / max_amp_exp
                norm_dict[f'Peak_{peak.id}_norm'] = y_comp_norm
            
            csv_norm = pd.DataFrame(norm_dict).to_csv(index=False)
            st.download_button(
                "📥 Export Normalized Deconvolution as CSV",
                data=csv_norm,
                file_name=f"deconvolution_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # --- Export 2.2 Gaussian Deconvolution vs Frequency ---
            st.markdown("**Export 2.2 Deconvolution vs Frequency** *(Spectrum + Total Fit + Peaks)*")
            
            freqs_deconv = 1.0 / (2.0 * np.pi * deconv_result.x_linear)
            sort_idx_f = np.argsort(freqs_deconv)[::-1]  # high to low, as on Graph 2.2
            log_tau_f = np.log10(np.maximum(deconv_result.x_linear[sort_idx_f], 1e-15))
            
            freq_deconv_dict = {
                'Frequency_Hz': freqs_deconv[sort_idx_f],
                'gamma_tau_Ohm': deconv_result.y_original[sort_idx_f],
                'Total_Fit_Ohm': fit_values[sort_idx_f]
            }
            for peak in peaks_sorted_exp:
                y_comp_f = peak.amplitude * GaussianModelDeconv.gaussian(
                    log_tau_f, 1.0, peak.center_log, peak.sigma_log
                )
                freq_deconv_dict[f'Peak_{peak.id}_Ohm'] = y_comp_f
            
            csv_freq_deconv = pd.DataFrame(freq_deconv_dict).to_csv(index=False)
            st.download_button(
                "📥 Export 2.2 Deconvolution vs Frequency as CSV",
                data=csv_freq_deconv,
                file_name=f"deconvolution_vs_freq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # --- Export 2.4 Experimental vs Sequential RQ (ZARC) Model ---
            if data is not None and drt_result is not None:
                st.markdown("**Export 2.4 Sequential RQ (ZARC) Model** *(Experimental + Total Model + Processes)*")
                
                sort_idx_exp = np.argsort(data.freq)
                freq_exp_s = data.freq[sort_idx_exp]
                re_exp_s = data.re_z[sort_idx_exp]
                im_exp_s = data.im_z[sort_idx_exp]
                omega_exp_s = 2.0 * np.pi * freq_exp_s
                
                R_inf_start = drt_result.R_inf
                for i in range(len(im_exp_s) - 1):
                    if im_exp_s[i] >= 0 and im_exp_s[i + 1] < 0:
                        t = -im_exp_s[i] / (im_exp_s[i + 1] - im_exp_s[i])
                        R_inf_start = float(re_exp_s[i] + t * (re_exp_s[i + 1] - re_exp_s[i]))
                        break
                
                Z_total_exp = np.full_like(omega_exp_s, R_inf_start, dtype=complex)
                if drt_result.L > 0:
                    Z_total_exp += 1j * omega_exp_s * drt_result.L
                
                process_cols = {}
                cumulative_R = R_inf_start
                for peak in peaks_sorted_exp:
                    sigma = max(float(peak.sigma_log), 1e-6)
                    center_log = float(peak.center_log)
                    if sigma < 1e-4:
                        Z_i = peak.area / (1.0 + 1j * omega_exp_s * peak.center)
                    else:
                        u_grid = np.linspace(center_log - 5.0 * sigma, center_log + 5.0 * sigma, 401)
                        tau_sub = 10.0 ** u_grid
                        gauss_profile = np.exp(-0.5 * ((u_grid - center_log) / sigma) ** 2)
                        profile_area = np.trapezoid(gauss_profile, u_grid)
                        g_weights = (peak.area / profile_area) * gauss_profile if profile_area > 0 else np.zeros_like(u_grid)
                        denom = 1.0 + 1j * np.outer(omega_exp_s, tau_sub)
                        Z_i = np.trapezoid(g_weights[np.newaxis, :] / denom, u_grid, axis=1)
                    
                    Z_total_exp += Z_i
                    Z_indiv = cumulative_R + Z_i
                    process_cols[f'Re_Process_{peak.id}_Ohm'] = np.real(Z_indiv)
                    process_cols[f'NegIm_Process_{peak.id}_Ohm'] = -np.imag(Z_indiv)
                    cumulative_R += peak.area
                
                rq_export_dict = {
                    'Frequency_Hz': freq_exp_s,
                    'Re_Exp_Ohm': re_exp_s,
                    'NegIm_Exp_Ohm': im_exp_s,
                    'Re_Total_Model_Ohm': np.real(Z_total_exp),
                    'NegIm_Total_Model_Ohm': -np.imag(Z_total_exp)
                }
                rq_export_dict.update(process_cols)
                
                csv_rq_model = pd.DataFrame(rq_export_dict).to_csv(index=False)
                st.download_button(
                    "📥 Export 2.4 Sequential RQ Model as CSV",
                    data=csv_rq_model,
                    file_name=f"sequential_rq_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# =============================================================================
# 6. SIDEBAR BULK EXPORT (ALL FIGURES + CSV IN ONE ZIP)
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Bulk Export for Publication")
if EXPORT_REGISTRY:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, data_bytes in EXPORT_REGISTRY.items():
            zf.writestr(file_path, data_bytes)
    st.sidebar.download_button(
        label=f"⬇️ Download All ({len(EXPORT_REGISTRY)} files, ZIP)",
        data=zip_buf.getvalue(),
        file_name="pcec_pcfc_publication_bundle.zip",
        mime="application/zip",
    )
