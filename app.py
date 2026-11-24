import io
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_bvp, trapezoid
import streamlit as st

# =============================================================================
# 1. PAGE CONFIGURATION & UNIVERSAL PHYSICAL CONSTANTS
# =============================================================================
st.set_page_config(
    page_title="PCEC / PCFC Multiphysics Membrane Model",
    page_icon="⚡",
    layout="wide"
)

R_GAS = 8.314462618      # J / (mol * K)
F_CONST = 96485.33212    # C / mol
# Gas-phase steam formation (H2 + 0.5 O2 <-> H2O(g)) at ~600 C (Thatte 2024; Moranti 2026)
DH_W = -248110.0         # J / mol
DS_W = -55.48            # J / (mol * K)
LHV_H2 = 241800.0        # J / mol (at 25 C, for LHV efficiency definition)
V_MOLAR_STP = 22414.0    # Nml / mol (0 C, 1 atm)


@dataclass(frozen=True)
class MaterialParams:
    name: str
    Vm: float
    dH_H2O: float
    dS_H2O: float
    dH_O2: float
    dS_O2: float
    D0_p: float
    E_p: float
    D0_v: float
    E_v: float
    D0_h: float
    E_h: float
    default_cR: float
    reference: str


MATERIAL_PRESETS = {
    "BCZYYb7111 (Thatte 2024 / Restrepo 2026)": MaterialParams(
        name="BCZYYb7111", Vm=5.13e-5,
        dH_H2O=-93300.0, dS_H2O=-100.0,
        dH_O2=114880.0, dS_O2=-45.89,
        D0_p=1.02e-7, E_p=42650.0,
        D0_v=1.73e-7, E_v=59700.0,
        D0_h=2.41e-5, E_h=8400.0,
        default_cR=0.20,
        reference="J. Power Sources 611 (2024) 234706; Appl. Energy 420 (2026) 128115"
    ),
    "BZY20 (Moranti 2026)": MaterialParams(
        name="BZY20", Vm=4.57e-5,
        dH_H2O=-93300.0, dS_H2O=-100.0,
        dH_O2=115310.0, dS_O2=-45.89,
        D0_p=5.18e-7, E_p=60660.0,
        D0_v=2.03e-7, E_v=85190.0,
        D0_h=1.38e-5, E_h=7180.0,
        default_cR=0.20,
        reference="J. Power Sources 663 (2026) 238898"
    ),
    "BCZY652015 (Moranti 2026)": MaterialParams(
        name="BCZY652015", Vm=4.57e-5,
        dH_H2O=-123000.0, dS_H2O=-130.95,
        dH_O2=132570.0, dS_O2=-119.93,
        D0_p=1.63e-7, E_p=47190.0,
        D0_v=1.36e-7, E_v=68060.0,
        D0_h=2.50e-3, E_h=9730.0,
        default_cR=0.15,
        reference="J. Power Sources 663 (2026) 238898"
    ),
}


# =============================================================================
# 2. PHYSICS ENGINE (3 MOBILE CARRIERS: p=OH_O*, v=V_O**, h=O_O*)
# =============================================================================
def get_material_props(T_K: float, mat: MaterialParams):
    """Returns equilibrium constants and defect diffusivities at T_K."""
    K_W = np.exp(DS_W / R_GAS - DH_W / (R_GAS * T_K))
    K_H2O = np.exp(mat.dS_H2O / R_GAS - mat.dH_H2O / (R_GAS * T_K))
    K_O2 = np.exp(mat.dS_O2 / R_GAS - mat.dH_O2 / (R_GAS * T_K))
    D_p = mat.D0_p * np.exp(-mat.E_p / (R_GAS * T_K))
    D_v = mat.D0_v * np.exp(-mat.E_v / (R_GAS * T_K))
    D_h = mat.D0_h * np.exp(-mat.E_h / (R_GAS * T_K))
    return K_W, K_H2O, K_O2, D_p, D_v, D_h


def solve_boundary_defects(p_H2O: float, p_O2: float, c_R: float, K_H2O: float, K_O2: float):
    """
    Exact analytical solution of defect equilibrium at a boundary
    (3 mobile defects + O_O^x = 3, electroneutrality p + 2v + h = c_R).
    Reduces strictly to a quadratic equation in v in (0, c_R/2).
    """
    p_H2O = max(p_H2O, 1e-18)
    p_O2 = max(p_O2, 1e-35)

    alpha = np.sqrt((K_O2 * np.sqrt(p_O2)) / (K_H2O * p_H2O))
    denom = 1.0 + alpha
    K_eff = K_H2O * p_H2O * (denom ** 2)

    # Equation: (c_R - 2v)^2 = K_eff * v * (3 - c_R + v)
    # => (4 - K_eff) * v^2 - (4*c_R + K_eff*(3 - c_R)) * v + c_R^2 = 0
    A = 4.0 - K_eff
    B = -(4.0 * c_R + K_eff * (3.0 - c_R))
    C = c_R ** 2

    v_max = 0.5 * c_R
    if abs(A) < 1e-12:
        v_sol = -C / B
    else:
        disc = max(B * B - 4.0 * A * C, 0.0)
        sqrt_disc = np.sqrt(disc)
        r1 = (-B - sqrt_disc) / (2.0 * A)
        r2 = (-B + sqrt_disc) / (2.0 * A)
        candidates = [r for r in (r1, r2) if 0.0 <= r <= v_max * (1.0 + 1e-12)]
        v_sol = min(candidates, key=lambda x: abs(x - 0.5 * v_max)) if candidates else np.clip(r1, 1e-18, v_max * (1.0 - 1e-15))

    v_sol = float(np.clip(v_sol, 1e-18 * v_max, v_max * (1.0 - 1e-15)))
    p_sol = (c_R - 2.0 * v_sol) / denom
    h_sol = alpha * p_sol
    o_sol = 3.0 - c_R + v_sol
    return p_sol, v_sol, h_sol, o_sol


def bernoulli_flux(u: float, c_f: float, c_o: float) -> float:
    """
    Numerically stable evaluation of u * (c_o - c_f * exp(-u)) / (1 - exp(-u))
    written via dc = c_o - c_f and c_bar = 0.5*(c_o + c_f) to avoid cancellation.
    """
    dc = c_o - c_f
    c_bar = 0.5 * (c_o + c_f)
    if abs(u) < 1e-5:
        u2 = u * u
        return dc * (1.0 + u2 / 12.0 - (u2 * u2) / 720.0) + u * c_bar
    elif u > 0:
        eu = np.exp(-u)
        return u * (dc + c_f * (1.0 - eu)) / (1.0 - eu)
    else:
        eu = np.exp(u)
        return u * (dc * eu - c_f * (1.0 - eu)) / (eu - 1.0)


def profile_exp(x_norm: np.ndarray, u: float, c_f: float, c_o: float) -> np.ndarray:
    """Numerically stable analytical concentration profile c(x/L) under constant field."""
    if abs(u) < 1e-6:
        return c_f + (c_o - c_f) * x_norm
    elif u > 0:
        return c_f + (c_o - c_f) * (1.0 - np.exp(-u * x_norm)) / (1.0 - np.exp(-u))
    else:
        return c_o + (c_f - c_o) * (1.0 - np.exp(u * (1.0 - x_norm))) / (1.0 - np.exp(u))


def compute_gas_fluxes(J_p, J_v, p_H2_f, p_H2O_f, p_O2_f):
    """Gas species fluxes at the fuel side (mol m^-2 s^-1) per Thatte (2024) / Restrepo (2026)."""
    p_H2O_f = max(p_H2O_f, 1e-12)
    p_O2_f = max(p_O2_f, 1e-35)
    p_H2_f = max(p_H2_f, 1e-12)

    term_p = 2.0 + 2.0 * (p_H2_f / p_H2O_f) + 0.5 * (p_H2_f / p_O2_f)
    term_v = 1.0 + 4.0 * (p_O2_f / p_H2O_f) + 4.0 * (p_O2_f / p_H2_f)

    J_H2O = J_p / term_p - J_v / term_v
    J_H2 = 0.5 * J_p - J_H2O
    J_O2 = -0.5 * J_v - 0.5 * J_H2O
    return J_H2, J_H2O, J_O2


@st.cache_data(show_spinner=False)
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
    mat: MaterialParams,
    solver_mode: str = "fast",
    use_arcsinh_shift: bool = True,
    n_grid: int = 80,
):
    """
    Unified 1D solver for PCFC (i > 0), OCV (i = 0), and PCEC (i < 0).
    Cached via @st.cache_data for fast interactive exploration.
    """
    T_K = T_C + 273.15
    L_m = L_um * 1e-6
    Vm = mat.Vm
    i_A_m2 = i_A_cm2 * 1e4  # A/m^2

    K_W, K_H2O, K_O2, D_p, D_v, D_h = get_material_props(T_K, mat)

    # Gas-phase equilibria at both boundaries
    p_O2_o = max(0.21 * (1.0 - p_H2O_o), 1e-12)
    p_O2_f = (p_H2O_f / (K_W * max(p_H2_f, 1e-12))) ** 2
    p_H2_o = p_H2O_o / (K_W * np.sqrt(p_O2_o))

    # Equilibrium boundary concentrations (i = 0)
    pf_eq, vf_eq, hf_eq, of_eq = solve_boundary_defects(p_H2O_f, p_O2_f, c_R, K_H2O, K_O2)
    po_eq, vo_eq, ho_eq, oo_eq = solve_boundary_defects(p_H2O_o, p_O2_o, c_R, K_H2O, K_O2)

    # Equilibrium (Nernst) potential E_eq (V)
    E_eq = (R_GAS * T_K / F_CONST) * np.log((ho_eq / oo_eq) / (hf_eq / of_eq))

    # Electrode activation overpotentials (Butler-Volmer arcsinh)
    eta_H = (R_GAS * T_K / F_CONST) * np.arcsinh((i_A_cm2 * Rp_H * F_CONST) / (R_GAS * T_K)) if Rp_H > 0 else 0.0
    eta_O = (R_GAS * T_K / F_CONST) * np.arcsinh((i_A_cm2 * Rp_O * F_CONST) / (R_GAS * T_K)) if Rp_O > 0 else 0.0

    # Phenomenological boundary concentration shift via effective p_O2
    if use_arcsinh_shift and (Rp_H > 0 or Rp_O > 0):
        shift_f = np.clip(4.0 * F_CONST * eta_H / (R_GAS * T_K), -25.0, 25.0)
        shift_o = np.clip(-4.0 * F_CONST * eta_O / (R_GAS * T_K), -25.0, 25.0)
        pf, vf, hf, of = solve_boundary_defects(p_H2O_f, p_O2_f * np.exp(shift_f), c_R, K_H2O, K_O2)
        po, vo, ho, oo = solve_boundary_defects(p_H2O_o, p_O2_o * np.exp(shift_o), c_R, K_H2O, K_O2)
    else:
        pf, vf, hf, of = pf_eq, vf_eq, hf_eq, of_eq
        po, vo, ho, oo = po_eq, vo_eq, ho_eq, oo_eq

    # Molar concentrations at boundaries (mol/m^3)
    cf_p, cf_v, cf_h = pf / Vm, vf / Vm, hf / Vm
    co_p, co_v, co_h = po / Vm, vo / Vm, ho / Vm

    # --- STEP 1: FAST ALGEBRAIC SOLVER (CONSTANT ELECTRIC FIELD) ---
    def total_current_alg(dPhi_V):
        u1 = F_CONST * dPhi_V / (R_GAS * T_K)
        u2 = 2.0 * u1
        Jp = -(D_p / L_m) * bernoulli_flux(u1, cf_p, co_p)
        Jv = -(D_v / L_m) * bernoulli_flux(u2, cf_v, co_v)
        Jh = -(D_h / L_m) * bernoulli_flux(u1, cf_h, co_h)
        return F_CONST * (Jp + 2.0 * Jv + Jh) - i_A_m2

    lo, hi = -3.0, 3.0
    f_lo, f_hi = total_current_alg(lo), total_current_alg(hi)
    for _ in range(10):
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
    ch_prof = np.maximum(c_R - cp_prof - 2.0 * cv_prof, 1e-18)
    Phi_prof = dPhi_alg * x_norm
    dPhi_dx_prof = np.full_like(x_norm, dPhi_alg / L_m)
    E_field_prof = -dPhi_dx_prof

    J_p = -(D_p / L_m) * bernoulli_flux(u1_alg, cf_p, co_p)
    J_v = -(D_v / L_m) * bernoulli_flux(u2_alg, cf_v, co_v)
    J_h = -(D_h / L_m) * bernoulli_flux(u1_alg, cf_h, co_h)
    bvp_converged = True

    # --- STEP 2: EXACT 1D BVP SOLVER (NEWMAN ELECTRONEUTRALITY + VARIABLE FIELD) ---
    # Eliminates artificial eps_reg; enforces exact local electroneutrality c_h = c_R - c_p - 2*c_v
    # and computes variable dPhi/dx from total current conservation sum(z_k J_k) = i_nd.
    if solver_mode == "bvp":
        Dp_nd = 1.0
        Dv_nd = D_v / D_p
        Dh_nd = D_h / D_p
        i_nd = i_A_m2 * L_m * Vm / (D_p * F_CONST)

        Jp_nd0 = J_p * L_m * Vm / D_p
        Jv_nd0 = J_v * L_m * Vm / D_p

        def ode_rhs(x, y):
            # y = [cp, cv, Jp_nd, Jv_nd, Phi_nd]
            cp, cv, Jp_n, Jv_n, Phi_n = y
            ch = np.maximum(c_R - cp - 2.0 * cv, 1e-18)
            # From electroneutrality: dch/dx = -dcp/dx - 2*dcv/dx
            # Total current conservation: Jp_n + 2*Jv_n + Jh_n = i_nd => Jh_n = i_nd - Jp_n - 2*Jv_n
            Jh_n = i_nd - Jp_n - 2.0 * Jv_n
            # Diffusive + migration balance yields exact variable field dPhi_n/dx:
            # -(Dp*dcp + 2*Dv*dcv + Dh*dch) = i_nd + (Dp*cp + 4*Dv*cv + Dh*ch)*dPhi_n
            # Substituting dch = -dcp - 2*dcv:
            # (Dh - Dp)*dcp + 2*(Dh - Dv)*dcv = i_nd + sigma_nd * dPhi_n
            # Using Nernst-Planck for dcp and dcv:
            # dcp = -Jp_n/Dp - cp*dPhi_n ; dcv = -Jv_n/Dv - 2*cv*dPhi_n
            sigma_nd = Dp_nd * cp + 4.0 * Dv_nd * cv + Dh_nd * ch
            diff_term = (
                (Dp_nd - Dh_nd) * (Jp_n / Dp_nd)
                + 2.0 * (Dv_nd - Dh_nd) * (Jv_n / Dv_nd)
                + Jh_n
            )
            # Equivalent direct identity:
            # sum z_k J_k = -sum z_k D_k dc_k/dx - sigma_nd * dPhi_n = i_nd
            # => dPhi_n = -(i_nd + (Dp - Dh)*(-Jp_n/Dp) + 2*(Dv - Dh)*(-Jv_n/Dv)) / (sigma_nd - (Dp-Dh)*cp - 4*(Dv-Dh)*cv)
            denom_phi = Dh_nd * (cp + 4.0 * cv + ch)
            num_phi = -(
                i_nd
                - (Dp_nd - Dh_nd) * (Jp_n / Dp_nd)
                - 2.0 * (Dv_nd - Dh_nd) * (Jv_n / Dv_nd)
            )
            dPhi_n = num_phi / np.maximum(denom_phi, 1e-18)

            dcp = -Jp_n / Dp_nd - 1.0 * cp * dPhi_n
            dcv = -Jv_n / Dv_nd - 2.0 * cv * dPhi_n
            zeros = np.zeros_like(x)
            return np.vstack([dcp, dcv, zeros, zeros, dPhi_n])

        def bc_func(ya, yb):
            return np.array([
                ya[0] - pf, ya[1] - vf,
                yb[0] - po, yb[1] - vo,
                ya[4] - 0.0,
            ])

        y_init = np.vstack([
            cp_prof, cv_prof,
            np.full_like(x_norm, Jp_nd0),
            np.full_like(x_norm, Jv_nd0),
            u1_alg * x_norm,
        ])

        sol = solve_bvp(ode_rhs, bc_func, x_norm, y_init, tol=1e-5, max_nodes=2000)
        if sol.success:
            y_sol = sol.sol(x_norm)
            cp_prof = np.maximum(y_sol[0], 1e-18)
            cv_prof = np.maximum(y_sol[1], 1e-18)
            ch_prof = np.maximum(c_R - cp_prof - 2.0 * cv_prof, 1e-18)
            Jp_nd_sol = float(np.mean(y_sol[2]))
            Jv_nd_sol = float(np.mean(y_sol[3]))
            Jh_nd_sol = i_nd - Jp_nd_sol - 2.0 * Jv_nd_sol

            J_p = Jp_nd_sol * D_p / (L_m * Vm)
            J_v = Jv_nd_sol * D_p / (L_m * Vm)
            J_h = Jh_nd_sol * D_p / (L_m * Vm)

            Phi_prof = y_sol[4] * (R_GAS * T_K / F_CONST)
            dPhi_n_prof = ode_rhs(x_norm, y_sol)[4]
            dPhi_dx_prof = dPhi_n_prof * (R_GAS * T_K / (F_CONST * L_m))
            E_field_prof = -dPhi_dx_prof
            dPhi_final = float(Phi_prof[-1] - Phi_prof[0])
        else:
            bvp_converged = False
            dPhi_final = dPhi_alg
    else:
        dPhi_final = dPhi_alg

    # --- DERIVED QUANTITIES ---
    co_prof = np.maximum(3.0 - cp_prof - cv_prof - ch_prof, 1e-12)

    # Partial and total conductivities (S/cm)
    pref = (F_CONST ** 2 / (R_GAS * T_K * Vm)) * 1e-2  # S/m -> S/cm
    sig_p_prof = pref * (1.0 ** 2) * cp_prof * D_p
    sig_v_prof = pref * (2.0 ** 2) * cv_prof * D_v
    sig_h_prof = pref * (1.0 ** 2) * ch_prof * D_h
    sig_tot_prof = sig_p_prof + sig_v_prof + sig_h_prof

    # Membrane Ohmic ASR (Ohm cm^2) integrated across thickness
    L_cm = L_um * 1e-4
    ASR_ohm = float(trapezoid(1.0 / np.maximum(sig_tot_prof, 1e-15), x_norm) * L_cm)

    # Spatial polaron leakage current components (A/cm^2)
    x_m = x_norm * L_m
    dch_dx = np.gradient(ch_prof / Vm, x_m)
    i_h_diff_prof = (-F_CONST * D_h * dch_dx) * 1e-4
    i_h_mig_prof = (-F_CONST * D_h * (ch_prof / Vm) * (F_CONST / (R_GAS * T_K)) * dPhi_dx_prof) * 1e-4

    # Internal gas partial pressures (atm)
    p_H2O_int = (cp_prof ** 2) / (co_prof * cv_prof * K_H2O)
    p_O2_int = ((ch_prof ** 2) / (co_prof * cv_prof * K_O2)) ** 2
    p_H2_int = p_H2O_int / (K_W * np.sqrt(np.maximum(p_O2_int, 1e-35)))

    # Partial current densities (A/cm^2)
    i_p = (1.0 * F_CONST * J_p) * 1e-4
    i_v = (2.0 * F_CONST * J_v) * 1e-4
    i_h = (1.0 * F_CONST * J_h) * 1e-4
    i_tot = i_p + i_v + i_h

    # Leakage fraction r_leak (%)
    sum_abs_i = abs(i_p) + abs(i_v) + abs(i_h)
    r_leak = (abs(i_h) / sum_abs_i * 100.0) if sum_abs_i > 1e-15 else 0.0

    # Cell voltage (V)
    if use_arcsinh_shift and (Rp_H > 0 or Rp_O > 0):
        E_cell_shifted = (R_GAS * T_K / F_CONST) * np.log((ho / oo) / (hf / of))
        V_cell = E_cell_shifted + dPhi_final
    else:
        V_cell = E_eq + dPhi_final - (eta_H + eta_O)

    # Gas fluxes and Faradaic efficiency
    J_H2, J_H2O, J_O2 = compute_gas_fluxes(J_p, J_v, p_H2_f, p_H2O_f, p_O2_f)
    i_H2_equiv = (2.0 * F_CONST * J_H2) * 1e-4  # A/cm^2

    if i_A_cm2 > 1e-6:  # PCFC
        FE = (i_tot / i_H2_equiv) if abs(i_H2_equiv) > 1e-12 else 0.0
        mode_str = "PCFC (Fuel Cell)"
    elif i_A_cm2 < -1e-6:  # PCEC
        FE = (i_H2_equiv / i_tot) if abs(i_tot) > 1e-12 else 0.0
        mode_str = "PCEC (Electrolyzer)"
    else:  # OCV
        FE = 0.0
        mode_str = "OCV (Open Circuit)"

    # Mean transport numbers
    t_p_cond = float(np.mean(sig_p_prof / sig_tot_prof))
    t_v_cond = float(np.mean(sig_v_prof / sig_tot_prof))
    t_h_cond = float(np.mean(sig_h_prof / sig_tot_prof))

    # Thermal balance (W/cm^2) and LHV energy efficiency (%)
    J_H2_cm2 = J_H2 * 1e-4  # mol cm^-2 s^-1
    J_H2_Nml_min = J_H2_cm2 * V_MOLAR_STP * 60.0  # Nml cm^-2 min^-1
    q_dot = -i_tot * V_cell + J_H2_cm2 * abs(DH_W)
    V_TN_theo = abs(DH_W) / (2.0 * F_CONST)
    V_TN_act = max(FE, 0.0) * V_TN_theo if i_A_cm2 < -1e-6 else V_TN_theo

    P_elec = i_tot * V_cell  # W/cm^2
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
        "bvp_converged": bvp_converged,
        "p_O2_f": p_O2_f,
        "T_C": T_C, "i": i_tot, "V_cell": V_cell, "E_eq": E_eq,
        "dPhi": dPhi_final, "eta_H": eta_H, "eta_O": eta_O,
        "i_p": i_p, "i_v": i_v, "i_h": i_h, "r_leak": r_leak,
        "t_p": t_p_cond, "t_v": t_v_cond, "t_h": t_h_cond,
        "ASR_ohm": ASR_ohm,
        "FE": FE * 100.0, "eff_LHV": eff_LHV * 100.0,
        "P_elec": P_elec, "q_dot": q_dot,
        "V_TN_theo": V_TN_theo, "V_TN_act": V_TN_act,
        "J_H2": J_H2_cm2, "J_H2_Nml_min": J_H2_Nml_min,
        "J_H2O": J_H2O * 1e-4, "J_O2": J_O2 * 1e-4,
        # 1D spatial profiles
        "x_norm": x_norm, "cp": cp_prof, "cv": cv_prof, "ch": ch_prof, "co": co_prof,
        "Phi": Phi_prof, "E_field": E_field_prof,
        "i_h_diff_prof": i_h_diff_prof, "i_h_mig_prof": i_h_mig_prof,
        "sig_p": sig_p_prof, "sig_v": sig_v_prof, "sig_h": sig_h_prof, "sig_tot": sig_tot_prof,
        "p_H2O_int": p_H2O_int, "p_O2_int": p_O2_int, "p_H2_int": p_H2_int,
    }


@st.cache_data(show_spinner=False)
def find_special_points(
    T_C: float, L_um: float, Rp_H: float, Rp_O: float,
    p_H2_f: float, p_H2O_f: float, p_H2O_o: float, c_R: float,
    mat: MaterialParams, use_arcsinh_shift: bool
):
    """Finds P_max (PCFC), i_TN & V_TN_act (PCEC), and i_PCFC_to_PCEC without redundant calls."""
    # 1. Transition current density i_PCFC->PCEC (where i_p = 0)
    def ip_func(iv):
        return solve_single_point(float(iv), T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat, "fast", use_arcsinh_shift, 15)["i_p"]

    try:
        i_trans = brentq(ip_func, -0.8, 0.2, xtol=1e-4)
    except ValueError:
        i_trans = np.nan

    # 2. Thermoneutral current density in PCEC (where q_dot = 0, i < 0)
    last_tn_res = {}
    def q_func(iv):
        r = solve_single_point(float(iv), T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat, "fast", use_arcsinh_shift, 15)
        last_tn_res["r"] = r
        return r["q_dot"]

    i_TN, V_TN_pt = np.nan, np.nan
    try:
        i_hi_tn = -0.05
        q_hi = q_func(i_hi_tn)
        for i_lo_tn in [-1.5, -3.5, -7.0, -12.0, -20.0]:
            q_lo = q_func(i_lo_tn)
            if q_lo * q_hi < 0:
                i_TN = brentq(q_func, i_lo_tn, i_hi_tn, xtol=1e-4)
                V_TN_pt = last_tn_res["r"]["V_cell"]
                break
    except ValueError:
        pass

    # 3. Maximum power density in PCFC (i > 0)
    i_fc_scan = np.linspace(0.05, 4.0, 35)
    p_vals = []
    for iv in i_fc_scan:
        r = solve_single_point(float(iv), T_C, L_um, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R, mat, "fast", use_arcsinh_shift, 15)
        p_vals.append(r["P_elec"] if r["V_cell"] > 0 else 0.0)
    idx_max = int(np.argmax(p_vals))
    P_max = float(p_vals[idx_max])
    i_Pmax = float(i_fc_scan[idx_max])

    return P_max, i_Pmax, i_TN, V_TN_pt, i_trans


# =============================================================================
# 3. SIDEBAR: INPUT PARAMETERS & PUBLICATION PLOT SETTINGS
# =============================================================================
st.sidebar.title("⚙️ Model Parameters")

preset_name = st.sidebar.selectbox("Electrolyte Material", list(MATERIAL_PRESETS.keys()))
base_mat = MATERIAL_PRESETS[preset_name]

with st.sidebar.expander("🔬 Solver & Boundary Conditions", expanded=True):
    solver_label = st.radio(
        "NPP Solver Mode",
        ["Fast (Algebraic, Constant Field)", "Exact (1D BVP, Newman Electroneutrality)"],
        index=0,
    )
    solver_mode = "fast" if "Fast" in solver_label else "bvp"
    use_arcsinh_shift = st.checkbox(
        "Shift boundary defects via Rp (arcsinh)",
        value=True,
        help="Couples electrode polarization overpotential with boundary defect concentrations and electronic leakage."
    )

with st.sidebar.expander("⚡ Operating & Geometric Parameters", expanded=True):
    i_work = st.number_input("Current density i (A cm⁻²) [<0 PCEC, >0 PCFC]", -2.5, 2.5, -0.8, 0.05, format="%.2f")
    T_work = st.slider("Temperature T (°C)", 450, 750, 600, 5)
    L_work = st.number_input("Electrolyte thickness L (µm)", 1.0, 500.0, 10.0, 1.0)
    c_R_work = st.slider("Acceptor dopant concentration c_R (-)", 0.10, 0.30, float(base_mat.default_cR), 0.01, key=f"cR_{preset_name}")
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
        p_H2O_f = st.slider("Fuel electrode p(H₂O) (atm)", 0.005, 0.50, 0.03, 0.005)
    p_H2O_o = st.slider("Air/steam electrode p(H₂O) (atm)", 0.01, 0.95, 0.50, 0.01)
    st.caption(f"Air/steam electrode p(O₂) = {0.21 * (1.0 - p_H2O_o):.4f} atm")

with st.sidebar.expander("🧪 Custom Material Overrides", expanded=False):
    st.caption(f"Ref: {base_mat.reference}")
    ov_dH_H2O = st.number_input("ΔH°(H₂O) (J mol⁻¹)", value=float(base_mat.dH_H2O), step=1000.0, key=f"dH_H2O_{preset_name}")
    ov_dS_H2O = st.number_input("ΔS°(H₂O) (J mol⁻¹ K⁻¹)", value=float(base_mat.dS_H2O), step=1.0, key=f"dS_H2O_{preset_name}")
    ov_dH_O2 = st.number_input("ΔH°(O₂) (J mol⁻¹)", value=float(base_mat.dH_O2), step=1000.0, key=f"dH_O2_{preset_name}")
    ov_dS_O2 = st.number_input("ΔS°(O₂) (J mol⁻¹ K⁻¹)", value=float(base_mat.dS_O2), step=1.0, key=f"dS_O2_{preset_name}")
    ov_E_p = st.number_input("E_act(OH•) (J mol⁻¹)", value=float(base_mat.E_p), step=500.0, key=f"E_p_{preset_name}")
    ov_E_h = st.number_input("E_act(O•) (J mol⁻¹)", value=float(base_mat.E_h), step=500.0, key=f"E_h_{preset_name}")

mat_params = MaterialParams(
    name=base_mat.name, Vm=base_mat.Vm,
    dH_H2O=ov_dH_H2O, dS_H2O=ov_dS_H2O,
    dH_O2=ov_dH_O2, dS_O2=ov_dS_O2,
    D0_p=base_mat.D0_p, E_p=ov_E_p,
    D0_v=base_mat.D0_v, E_v=base_mat.E_v,
    D0_h=base_mat.D0_h, E_h=ov_E_h,
    default_cR=base_mat.default_cR,
    reference=base_mat.reference
)

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
    cmap_multi = st.selectbox("Colormap (multi-curve series)", ["coolwarm", "viridis", "plasma", "turbo", "tab10"])
    cmap_heat = st.selectbox("Colormap (2D heatmaps)", ["inferno", "viridis", "magma", "RdYlBu_r", "plasma"])
    custom_legend_title = st.text_input("Custom legend title (multiparametric)", "")

# Apply publication Matplotlib style
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


def style_legend(ax, title=None, loc="best", handles=None, labels=None):
    """Creates a legend where each text label matches the color of its line/marker."""
    if handles is not None and labels is not None:
        leg = ax.legend(handles, labels, title=title, loc=loc)
    else:
        leg = ax.legend(title=title, loc=loc)
    if leg is not None:
        leg_handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
        for text, handle in zip(leg.get_texts(), leg_handles):
            if hasattr(handle, "get_color"):
                text.set_color(handle.get_color())
            elif hasattr(handle, "get_facecolor"):
                fc = handle.get_facecolor()
                if len(fc) > 0:
                    text.set_color(fc[0])
    return leg


def style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)


def render_plot_with_download(fig, filename_base: str):
    """Displays a Matplotlib figure with 600 DPI PNG and vector SVG download buttons."""
    st.pyplot(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=600)
    c1.download_button("⬇️ PNG (600 DPI)", buf_png.getvalue(), f"{filename_base}.png", "image/png", key=f"png_{filename_base}")
    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg")
    c2.download_button("⬇️ SVG (Vector)", buf_svg.getvalue(), f"{filename_base}.svg", "image/svg+xml", key=f"svg_{filename_base}")
    plt.close(fig)


# =============================================================================
# 4. HEADER, MODEL ASSUMPTIONS & WORKING POINT METRICS
# =============================================================================
st.title("⚡ Unified PCEC / PCFC Electro-, Mass-, and Heat-Transfer Model")
st.caption(f"Electrolyte: **{preset_name}** | Solver: **{solver_label}** | Carriers: OH•, V••, O•")

with st.expander("ℹ️ Model Assumptions, Governing Physics & Limitations", expanded=False):
    st.markdown("""
    * **Charge Carriers:** Three mobile point defects in Kröger–Vink notation: protons ($\\mathrm{OH_O^\\bullet}, z=+1$), oxygen vacancies ($\\mathrm{V_O^{\\bullet\\bullet}}, z=+2$), and small O-site polarons ($\\mathrm{O_O^\\bullet}, z=+1$), coupled with lattice oxygen conservation ($=3$) and local electroneutrality ($c_p + 2c_v + c_h = c_R$).
    * **Solvers:**
      * *Fast (Algebraic):* Assumes constant electric field across the dense electrolyte (Putilov et al., 2020; Restrepo & Banerjee, 2026).
      * *Exact (1D BVP):* Solves coupled Nernst–Planck ODEs with variable electric field $\\nabla\\Phi(x)$ derived from total current conservation and strict electroneutrality (Newman formulation).
    * **Limitations:** 1D isothermal membrane-level model; gas-channel convective depletion and concentration overpotentials in thick porous electrodes are neglected; electrode activation losses are modeled via macroscopic $R_p$ using Butler–Volmer $\\operatorname{arcsinh}$ kinetics; LHV efficiency in endothermic PCEC mode accounts for external heat input $Q_{\\mathrm{in}} = \\max(0, -\\dot{q})$.
    """)

res_pt = solve_single_point(
    i_work, T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
    mat_params, solver_mode=solver_mode, use_arcsinh_shift=use_arcsinh_shift
)

if solver_mode == "bvp" and not res_pt["bvp_converged"]:
    st.warning("1D BVP solver did not converge at this operating point (falling back to algebraic solution).")
if res_pt["FE"] > 100.5:
    st.warning(f"Calculated Faradaic Efficiency ({res_pt['FE']:.1f}%) exceeds 100% — check extreme gas composition or Rp settings.")
if res_pt["p_O2_f"] < 1e-32:
    st.warning("Fuel-side equilibrium p(O₂) is below 10⁻³² atm; numerical precision limits may apply at very low T.")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Mode", res_pt["mode"].split()[0])
m2.metric("Cell Voltage (V)", f"{res_pt['V_cell']:.4f}", delta=f"E_eq = {res_pt['E_eq']:.3f} V")
m3.metric("Faradaic Eff. (%)", f"{res_pt['FE']:.2f}%")
m4.metric("LHV Efficiency (%)", f"{res_pt['eff_LHV']:.2f}%")
m5.metric("Heat Flux q̇ (W cm⁻²)", f"{res_pt['q_dot']:.3f}", delta="Exothermic" if res_pt["q_dot"] > 0 else "Endothermic")
m6.metric("Leakage i_h (A cm⁻²)", f"{res_pt['i_h']:.4f}", delta=f"r_leak = {res_pt['r_leak']:.1f}%")

# =============================================================================
# 5. MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Spatial Profiles (x/L)",
    "2. I–V, Leakage & Thermal",
    "3. Multiparametric Analysis",
    "4. 2D Heatmaps"
])

mk = "o" if show_markers else None

# -----------------------------------------------------------------------------
# TAB 1: SPATIAL PROFILES ACROSS MEMBRANE THICKNESS
# -----------------------------------------------------------------------------
with tab1:
    st.subheader(f"Spatial Distributions Across Membrane Thickness (T = {T_work} °C, i = {i_work:.2f} A cm⁻²)")
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
        ax.axhline(0.0, color="gray", ls=":", lw=1.0, label="1 atm threshold")
        style_ax(ax, r"Normalized Coordinate ($x / L$)", r"$\log_{10}(p_i\ /\ \mathrm{atm})$", "Internal Gas Partial Pressures")
        style_legend(ax)
        render_plot_with_download(fig, "spatial_internal_pressures")

# -----------------------------------------------------------------------------
# TAB 2: I-V, LEAKAGE, AND THERMAL CHARACTERISTICS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader(f"Electrochemical, Leakage, and Thermal Characteristics (T = {T_work} °C)")
    col_i1, col_i2, col_i3 = st.columns(3)
    i_min = col_i1.number_input("Min Current Density (PCEC) (A cm⁻²)", -3.0, -0.1, -1.5, 0.1)
    i_max = col_i2.number_input("Max Current Density (PCFC) (A cm⁻²)", 0.1, 3.0, 1.0, 0.1)
    n_pts = col_i3.slider("Sweep Points", 20, 80, 41)

    if not (i_min <= i_work <= i_max):
        st.info(f"Note: Working current density i = {i_work:.2f} A cm⁻² lies outside the sweep range [{i_min:.2f}, {i_max:.2f}].")

    i_grid = np.linspace(i_min, i_max, n_pts)
    sweep_rows = []
    for iv in i_grid:
        r = solve_single_point(
            float(iv), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
            mat_params, solver_mode=solver_mode, use_arcsinh_shift=use_arcsinh_shift, n_grid=40
        )
        sweep_rows.append({k: r[k] for k in [
            "i", "V_cell", "E_eq", "dPhi", "eta_H", "eta_O",
            "i_p", "i_v", "i_h", "r_leak", "t_p", "t_v", "t_h",
            "FE", "eff_LHV", "P_elec", "q_dot", "V_TN_theo", "V_TN_act"
        ]})
    df_i = pd.DataFrame(sweep_rows)

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
        df_ec = df_i[df_i["i"] < -0.05]
        df_fc_p = df_i[df_i["i"] > 0.05]
        ax.plot(df_ec["i"], np.clip(df_ec["FE"], 0, 105), lw=lw, color="tab:blue", label="FE (PCEC)")
        ax.plot(df_fc_p["i"], np.clip(df_fc_p["FE"], 0, 105), lw=lw, color="tab:green", label="FE (PCFC)")
        ax.plot(df_i["i"], df_i["t_p"] * 100.0, lw=lw, ls="--", color="black", label=r"Mean $t_p$")
        ax.set_ylim(0, 105)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", "Efficiency / Transport Number (%)", "Faradaic Efficiency and Proton Transport Number")
        style_legend(ax)
        render_plot_with_download(fig, "iv_FE_transport")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_ec["i"], np.clip(df_ec["eff_LHV"], 0, 105), lw=lw, color="teal", label="PCEC Efficiency (LHV)")
        ax.plot(df_fc_p["i"], np.clip(df_fc_p["eff_LHV"], 0, 105), lw=lw, color="purple", label="PCFC Efficiency (LHV)")
        ax.set_ylim(0, 105)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", "LHV Energy Efficiency (%)", "Energy Conversion Efficiency (LHV)")
        style_legend(ax)
        render_plot_with_download(fig, "iv_energy_efficiency")

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["dPhi"], lw=lw, label=r"Membrane ($\Delta\Phi$)")
        ax.plot(df_i["i"], -df_i["eta_H"], lw=lw, label=r"Fuel electrode ($-\eta_{\mathrm{H_2}}$)")
        ax.plot(df_i["i"], -df_i["eta_O"], lw=lw, ls="--", label=r"Air electrode ($-\eta_{\mathrm{O_2}}$)")
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
        ax.plot(df_i["i"], df_i["q_dot"], lw=lw, color="darkorange", label=r"Heat flux ($\dot{q}$)")
        ax.axhline(0.0, color="black", ls="--", lw=1.0, label=r"Thermoneutral ($\dot{q}=0$)")
        ax.fill_between(df_i["i"], df_i["q_dot"], 0, where=(df_i["q_dot"] > 0), color="red", alpha=0.12)
        ax.fill_between(df_i["i"], df_i["q_dot"], 0, where=(df_i["q_dot"] < 0), color="blue", alpha=0.12)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", r"Heat Flux ($\mathrm{W\ cm^{-2}}$)", "Thermal Balance (Endothermic / Exothermic)")
        style_legend(ax)
        render_plot_with_download(fig, "iv_heat_balance")

    st.download_button("📥 Download I–V Sweep Data (CSV)", df_i.to_csv(index=False).encode("utf-8"), "iv_sweep_data.csv", "text/csv")


# -----------------------------------------------------------------------------
# TAB 3: MULTIPARAMETRIC ANALYSIS (ISOLATED VIA @st.fragment)
# -----------------------------------------------------------------------------
@st.fragment
def render_multiparametric_tab():
    st.subheader("Multiparametric Sensitivity Analysis")

    PARAM_OPTIONS = {
        "Temperature (°C)": {"key": "T_C", "min": 450.0, "max": 750.0, "def_min": 500.0, "def_max": 700.0, "fmt": "{:.0f} °C"},
        "Electrolyte thickness (µm)": {"key": "L_um", "min": 2.0, "max": 200.0, "def_min": 5.0, "def_max": 50.0, "fmt": "{:.1f} µm"},
        "Acceptor dopant concentration (-)": {"key": "c_R", "min": 0.10, "max": 0.30, "def_min": 0.10, "def_max": 0.30, "fmt": "{:.2f}"},
        "Fuel electrode p(H₂) (atm)": {"key": "p_H2_f", "min": 0.10, "max": 0.99, "def_min": 0.50, "def_max": 0.98, "fmt": "{:.2f} atm"},
        "Air electrode p(H₂O) (atm)": {"key": "p_H2O_o", "min": 0.02, "max": 0.95, "def_min": 0.10, "def_max": 0.90, "fmt": "{:.2f} atm"},
        "Electrode Rp (Ω cm²)": {"key": "Rp", "min": 0.0, "max": 1.0, "def_min": 0.02, "def_max": 0.20, "fmt": "{:.2f} Ω cm²"},
    }

    p_col1, p_col2, p_col3, p_col4 = st.columns([2, 1, 1, 1])
    var_label = p_col1.selectbox("Select Parameter to Vary", list(PARAM_OPTIONS.keys()), key="mp_var_select")
    p_cfg = PARAM_OPTIONS[var_label]
    v_min = p_col2.number_input("Min", p_cfg["min"], p_cfg["max"], p_cfg["def_min"], key=f"mp_min_{p_cfg['key']}")
    v_max = p_col3.number_input("Max", p_cfg["min"], p_cfg["max"], p_cfg["def_max"], key=f"mp_max_{p_cfg['key']}")
    n_vals = p_col4.slider("Number of Steps", 3, 9, 5, key="mp_steps")

    if v_max <= v_min:
        st.error("Max value must be greater than Min value.")
        return

    var_values = np.linspace(v_min, v_max, n_vals)
    cmap = plt.get_cmap(cmap_multi)
    colors = [cmap(k / max(len(var_values) - 1, 1)) for k in range(len(var_values))]
    leg_title_use = custom_legend_title.strip() if custom_legend_title.strip() else var_label

    i_multi = np.linspace(i_min, i_max, 33)
    multi_iv = {}
    summary_rows = []

    for val in var_values:
        T_v = float(val) if p_cfg["key"] == "T_C" else T_work
        L_v = float(val) if p_cfg["key"] == "L_um" else L_work
        cR_v = float(val) if p_cfg["key"] == "c_R" else c_R_work
        pH2_v = float(val) if p_cfg["key"] == "p_H2_f" else p_H2_f
        pH2Of_v = max(1.0 - pH2_v, 1e-4) if (p_cfg["key"] == "p_H2_f" and binary_fuel) else p_H2O_f
        pH2Oo_v = float(val) if p_cfg["key"] == "p_H2O_o" else p_H2O_o
        RpH_v = float(val) if p_cfg["key"] == "Rp" else Rp_H
        RpO_v = float(val) if p_cfg["key"] == "Rp" else Rp_O

        rows_v = []
        for iv in i_multi:
            r = solve_single_point(
                float(iv), T_v, L_v, RpH_v, RpO_v, pH2_v, pH2Of_v, pH2Oo_v, cR_v,
                mat_params, solver_mode="fast", use_arcsinh_shift=use_arcsinh_shift, n_grid=25
            )
            rows_v.append(r)
        multi_iv[val] = pd.DataFrame(rows_v)

        r_ocv = solve_single_point(0.0, T_v, L_v, RpH_v, RpO_v, pH2_v, pH2Of_v, pH2Oo_v, cR_v, mat_params, "fast", use_arcsinh_shift, 25)
        r_wk = solve_single_point(i_work, T_v, L_v, RpH_v, RpO_v, pH2_v, pH2Of_v, pH2Oo_v, cR_v, mat_params, "fast", use_arcsinh_shift, 25)
        P_max, i_Pmax, i_TN, V_TN_pt, i_trans = find_special_points(
            T_v, L_v, RpH_v, RpO_v, pH2_v, pH2Of_v, pH2Oo_v, cR_v, mat_params, use_arcsinh_shift
        )

        summary_rows.append({
            "param": val,
            "P_max": P_max, "i_Pmax": i_Pmax,
            "i_TN": i_TN, "V_TN_pt": V_TN_pt,
            "J_H2": abs(r_wk["J_H2"]), "J_H2_Nml_min": abs(r_wk["J_H2_Nml_min"]),
            "FE_wk": r_wk["FE"], "eff_wk": r_wk["eff_LHV"],
            "E_eq": r_ocv["E_eq"], "OCV": r_ocv["V_cell"],
            "RdOCV_mV": (r_ocv["E_eq"] - r_ocv["V_cell"]) * 1e3,
            "i_leak_OCV": abs(r_ocv["i_h"]), "i_trans": i_trans,
            "ASR_ohm": r_wk["ASR_ohm"], "t_p_wk": r_wk["t_p"] * 100.0,
            "q_dot_wk": r_wk["q_dot"], "V_cell_wk": r_wk["V_cell"],
        })

    df_sum = pd.DataFrame(summary_rows)

    st.markdown("### Families of Curves vs. Current Density")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for idx, val in enumerate(var_values):
            df_v = multi_iv[val]
            ax.plot(df_v["i"], df_v["V_cell"], lw=lw, marker=mk, ms=ms, color=colors[idx], label=p_cfg["fmt"].format(val))
        ax.axvline(0, color="black", ls=":", lw=0.8)
        style_ax(ax, r"Current Density (A cm$^{-2}$)", "Cell Voltage (V)", f"Polarization Curves vs. {var_label}")
        style_legend(ax, title=leg_title_use)
        render_plot_with_download(fig, "mp_family_IV")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for idx, val in enumerate(var_values):
            df_v = multi_iv[val]
            ax.plot(df_v["i"], df_v["i_h"], lw=lw, marker=mk, ms=ms, color=colors[idx], label=p_cfg["fmt"].format(val))
        ax.axhline(0, color="black", ls=":", lw=0.8)
        ax.axvline(0, color="black", ls=":", lw=0.8)
        style_ax(ax, r"External Current Density (A cm$^{-2}$)", r"Leakage Current Density (A cm$^{-2}$)", f"Electronic Leakage vs. {var_label}")
        style_legend(ax, title=leg_title_use)
        render_plot_with_download(fig, "mp_family_leakage")

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for idx, val in enumerate(var_values):
            df_v = multi_iv[val]
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
            df_v = multi_iv[val]
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
        ax2.plot(df_sum["param"], df_sum["t_p_wk"], lw=lw, ls="--", marker="s", ms=ms, color="tab:blue", label=r"Mean $t_p$")
        ax2.set_ylabel("Mean Proton Transport Number (%)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
        render_plot_with_download(fig, "mp_ASR_tp")

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.0))
        if df_sum["i_TN"].notna().any():
            ax.plot(df_sum["param"], df_sum["i_TN"], lw=lw, marker="o", ms=ms, color="darkorange", label=r"$i_{\mathrm{TN}}$ (PCEC)")
        else:
            ax.set_xlim(v_min, v_max)
        style_ax(ax, var_label, r"Thermoneutral Current Density (A cm$^{-2}$)", "Thermoneutral Operating Point (PCEC)")
        ax2 = ax.twinx()
        if df_sum["V_TN_pt"].notna().any():
            ax2.plot(df_sum["param"], df_sum["V_TN_pt"], lw=lw, ls="--", marker="s", ms=ms, color="black", label=r"$V_{\mathrm{TN}}^{\mathrm{act}}$")
        ax2.set_ylabel("Actual Thermoneutral Voltage (V)", color="black")
        ax2.tick_params(axis="y", labelcolor="black", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        if l1 or l2:
            style_legend(ax, handles=l1 + l2, labels=lb1 + lb2)
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

    st.download_button("📥 Download Multiparametric Summary (CSV)", df_sum.to_csv(index=False).encode("utf-8"), "multiparametric_summary.csv", "text/csv")


with tab3:
    render_multiparametric_tab()


# -----------------------------------------------------------------------------
# TAB 4: 2D HEATMAPS (ISOLATED VIA @st.fragment)
# -----------------------------------------------------------------------------
@st.fragment
def render_heatmaps_tab():
    st.subheader("2D Parametric Heatmaps")
    res_hm = st.slider("Heatmap Grid Resolution (N × N)", 12, 35, 20, key="hm_res")

    T_hm = np.linspace(480, 700, res_hm)
    i_hm_ec = np.linspace(-1.8, -0.1, res_hm)
    L_hm = np.linspace(3.0, 50.0, res_hm)

    FE_map = np.zeros((res_hm, res_hm))
    Q_map = np.zeros((res_hm, res_hm))
    Eff_L_map = np.zeros((res_hm, res_hm))
    pO2_map = np.zeros((res_hm, res_hm))

    for idx_t, tv in enumerate(T_hm):
        for idx_i, iv in enumerate(i_hm_ec):
            r = solve_single_point(float(iv), float(tv), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat_params, "fast", use_arcsinh_shift, 15)
            FE_map[idx_t, idx_i] = np.clip(r["FE"], 0, 100)
            Q_map[idx_t, idx_i] = r["q_dot"]

    for idx_l, lv in enumerate(L_hm):
        for idx_i, iv in enumerate(i_hm_ec):
            r = solve_single_point(float(iv), T_work, float(lv), Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat_params, "fast", use_arcsinh_shift, 15)
            Eff_L_map[idx_l, idx_i] = np.clip(r["eff_LHV"], 0, 100)

    x_hm = np.linspace(0.0, 1.0, res_hm)
    for idx_i, iv in enumerate(i_hm_ec):
        r = solve_single_point(float(iv), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat_params, "fast", use_arcsinh_shift, res_hm)
        pO2_map[:, idx_i] = np.log10(np.maximum(r["p_O2_int"], 1e-30))

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.4))
        cf = ax.contourf(i_hm_ec, T_hm, FE_map, levels=20, cmap=cmap_heat)
        if show_contours:
            cs = ax.contour(i_hm_ec, T_hm, FE_map, levels=8, colors="white", linewidths=0.7, alpha=0.85)
            ax.clabel(cs, inline=True, fontsize=fs_tick - 1, fmt="%.0f%%")
        fig.colorbar(cf, ax=ax, label="Faradaic Efficiency (%)")
        style_ax(ax, r"PCEC Current Density (A cm$^{-2}$)", "Temperature (°C)", "Faradaic Efficiency Map")
        render_plot_with_download(fig, "heatmap_FE_Ti")

        fig, ax = plt.subplots(figsize=(6, 4.4))
        cf = ax.contourf(i_hm_ec, L_hm, Eff_L_map, levels=20, cmap=cmap_heat)
        if show_contours:
            cs = ax.contour(i_hm_ec, L_hm, Eff_L_map, levels=8, colors="white", linewidths=0.7, alpha=0.85)
            ax.clabel(cs, inline=True, fontsize=fs_tick - 1, fmt="%.0f%%")
        fig.colorbar(cf, ax=ax, label="LHV Energy Efficiency (%)")
        style_ax(ax, r"PCEC Current Density (A cm$^{-2}$)", "Electrolyte Thickness (µm)", f"Optimum Thickness Map ({T_work} °C)")
        render_plot_with_download(fig, "heatmap_Eff_Li")

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.4))
        vmax = max(abs(Q_map.min()), abs(Q_map.max()), 0.1)
        cf = ax.contourf(i_hm_ec, T_hm, Q_map, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        if show_contours:
            cs0 = ax.contour(i_hm_ec, T_hm, Q_map, levels=[0.0], colors="black", linewidths=2.0)
            ax.clabel(cs0, inline=True, fontsize=fs_tick, fmt="q̇=0 (TN)")
        fig.colorbar(cf, ax=ax, label=r"Heat Flux (W cm$^{-2}$)")
        style_ax(ax, r"PCEC Current Density (A cm$^{-2}$)", "Temperature (°C)", "Thermal Balance and Thermoneutral Boundary")
        render_plot_with_download(fig, "heatmap_Q_Ti")

        fig, ax = plt.subplots(figsize=(6, 4.4))
        cf = ax.contourf(i_hm_ec, x_hm, pO2_map, levels=20, cmap=cmap_heat)
        if show_contours:
            cs = ax.contour(i_hm_ec, x_hm, pO2_map, levels=6, colors="white", linewidths=0.7, alpha=0.85)
            ax.clabel(cs, inline=True, fontsize=fs_tick - 1, fmt="%.1f")
        fig.colorbar(cf, ax=ax, label=r"$\log_{10}(p_{\mathrm{O_2}}\ /\ \mathrm{atm})$")
        style_ax(ax, r"PCEC Current Density (A cm$^{-2}$)", r"Normalized Coordinate ($x / L$)", f"Internal Oxygen Pressure Map ({T_work} °C)")
        render_plot_with_download(fig, "heatmap_pO2_xi")


with tab4:
    render_heatmaps_tab()
