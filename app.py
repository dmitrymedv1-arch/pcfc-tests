import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import brentq
from scipy.integrate import solve_bvp
import streamlit as st

# =============================================================================
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ И ФИЗИЧЕСКИЕ КОНСТАНТЫ
# =============================================================================
st.set_page_config(
    page_title="PCEC / PCFC Multiphysics Membrane Model",
    page_icon="⚡",
    layout="wide"
)

R_GAS = 8.314462618      # Дж / (моль * К)
F_CONST = 96485.33212    # Кл / моль
DH_W = -248110.0         # Дж / моль (H2 + 0.5 O2 <-> H2O, 500-700 C)
DS_W = -55.48            # Дж / (моль * К)
LHV_H2 = 241800.0        # Дж / моль (при 25 C)

# Пресеты материалов из статей (Thatte 2024, Restrepo 2026, Moranti 2026)
MATERIAL_PRESETS = {
    "BCZYYb7111 (Thatte 2024 / Restrepo 2026)": {
        "Vm": 5.13e-5,
        "dH_H2O": -93300.0,  "dS_H2O": -100.0,
        "dH_O2":  114880.0,  "dS_O2":  -45.89,
        "D0_p": 1.02e-7,     "E_p": 42650.0,
        "D0_v": 1.73e-7,     "E_v": 59700.0,
        "D0_h": 2.41e-5,     "E_h": 8400.0,
        "default_cR": 0.20,
    },
    "BZY20 (Moranti 2026)": {
        "Vm": 4.57e-5,
        "dH_H2O": -93300.0,  "dS_H2O": -100.0,
        "dH_O2":  115310.0,  "dS_O2":  -45.89,
        "D0_p": 5.18e-7,     "E_p": 60660.0,
        "D0_v": 2.03e-7,     "E_v": 85190.0,
        "D0_h": 1.38e-5,     "E_h": 7180.0,
        "default_cR": 0.20,
    },
    "BCZY652015 (Moranti 2026)": {
        "Vm": 4.57e-5,
        "dH_H2O": -123000.0, "dS_H2O": -130.95,
        "dH_O2":  132570.0,  "dS_O2":  -119.93,
        "D0_p": 1.63e-7,     "E_p": 47190.0,
        "D0_v": 1.36e-7,     "E_v": 68060.0,
        "D0_h": 2.50e-3,     "E_h": 9730.0,
        "default_cR": 0.15,
    },
}


# =============================================================================
# 2. ФИЗИКО-МАТЕМАТИЧЕСКОЕ ЯДРО (3 НОСИТЕЛЯ: p=OH_O*, v=V_O**, h=O_O*)
# =============================================================================
def get_material_props(T_K: float, mat: dict):
    """Возвращает константы равновесия и коэффициенты диффузии при T_K."""
    K_W = np.exp(DS_W / R_GAS - DH_W / (R_GAS * T_K))
    K_H2O = np.exp(mat["dS_H2O"] / R_GAS - mat["dH_H2O"] / (R_GAS * T_K))
    K_O2 = np.exp(mat["dS_O2"] / R_GAS - mat["dH_O2"] / (R_GAS * T_K))
    D_p = mat["D0_p"] * np.exp(-mat["E_p"] / (R_GAS * T_K))
    D_v = mat["D0_v"] * np.exp(-mat["E_v"] / (R_GAS * T_K))
    D_h = mat["D0_h"] * np.exp(-mat["E_h"] / (R_GAS * T_K))
    return K_W, K_H2O, K_O2, D_p, D_v, D_h


def solve_boundary_defects(p_H2O: float, p_O2: float, c_R: float, K_H2O: float, K_O2: float):
    """
    Точное решение дефектного равновесия на границе (3 подвижных дефекта + O_O^x = 3).
    Возвращает безразмерные концентрации [OH_O*]_L, [V_O**]_L, [O_O*]_L, [O_O^x]_L.
    """
    p_H2O = max(p_H2O, 1e-18)
    p_O2 = max(p_O2, 1e-35)

    alpha = np.sqrt((K_O2 * np.sqrt(p_O2)) / (K_H2O * p_H2O))
    denom = 1.0 + alpha

    def residual(v):
        p = (c_R - 2.0 * v) / denom
        o = 3.0 - c_R + v
        return p * p - K_H2O * p_H2O * v * o

    v_max = 0.5 * c_R
    # Гарантированный единственный корень в (0, c_R/2)
    v_sol = brentq(residual, 1e-18 * v_max, v_max * (1.0 - 1e-15), xtol=1e-14, rtol=1e-13)
    p_sol = (c_R - 2.0 * v_sol) / denom
    h_sol = alpha * p_sol
    o_sol = 3.0 - c_R + v_sol
    return p_sol, v_sol, h_sol, o_sol


def bernoulli_flux(u: float, c_f: float, c_o: float) -> float:
    """
    Численно устойчивое вычисление u * (c_o - c_f * exp(-u)) / (1 - exp(-u))
    без переполнения экспоненты для любого знака и величины u.
    """
    if abs(u) < 1e-6:
        # Разложение Тейлора вокруг u = 0
        return (c_o - c_f) + 0.5 * u * (c_o + c_f) + (u * u / 12.0) * (c_o - c_f)
    elif u > 0:
        eu = np.exp(-u)
        return u * (c_o - c_f * eu) / (1.0 - eu)
    else:
        eu = np.exp(u)
        return u * (c_o * eu - c_f) / (eu - 1.0)


def profile_exp(x_norm: np.ndarray, u: float, c_f: float, c_o: float) -> np.ndarray:
    """Устойчивый аналитический профиль концентрации c(x/L) при постоянном поле."""
    if abs(u) < 1e-6:
        return c_f + (c_o - c_f) * x_norm
    elif u > 0:
        return c_f + (c_o - c_f) * (1.0 - np.exp(-u * x_norm)) / (1.0 - np.exp(-u))
    else:
        return c_o + (c_f - c_o) * (1.0 - np.exp(u * (1.0 - x_norm))) / (1.0 - np.exp(u))


def compute_gas_fluxes(J_p, J_v, p_H2_f, p_H2O_f, p_O2_f):
    """
    Газовые потоки на водородной стороне (моль/(м^2*с)) по Thatte (2024) / Restrepo (2026).
    """
    p_H2O_f = max(p_H2O_f, 1e-12)
    p_O2_f = max(p_O2_f, 1e-35)
    p_H2_f = max(p_H2_f, 1e-12)

    term_p = 2.0 + 2.0 * (p_H2_f / p_H2O_f) + 0.5 * (p_H2_f / p_O2_f)
    term_v = 1.0 + 4.0 * (p_O2_f / p_H2O_f) + 4.0 * (p_O2_f / p_H2_f)

    J_H2O = J_p / term_p - J_v / term_v
    J_H2 = 0.5 * J_p - J_H2O
    J_O2 = -0.5 * J_v - 0.5 * J_H2O
    return J_H2, J_H2O, J_O2


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
):
    """
    Единый решатель для PCFC (i>0), OCV (i=0) и PCEC (i<0).
    Возвращает словарь со всеми скалярными метриками и 1D-профилями по x/L.
    """
    T_K = T_C + 273.15
    L_m = L_um * 1e-6
    Vm = mat["Vm"]
    i_A_m2 = i_A_cm2 * 1e4  # А/м^2

    K_W, K_H2O, K_O2, D_p, D_v, D_h = get_material_props(T_K, mat)

    # Газофазное равновесие на границах
    p_O2_o = max(0.21 * (1.0 - p_H2O_o), 1e-12)
    p_O2_f = (p_H2O_f / (K_W * max(p_H2_f, 1e-12))) ** 2
    p_H2_o = p_H2O_o / (K_W * np.sqrt(p_O2_o))

    # Равновесные концентрации (i=0)
    pf_eq, vf_eq, hf_eq, of_eq = solve_boundary_defects(p_H2O_f, p_O2_f, c_R, K_H2O, K_O2)
    po_eq, vo_eq, ho_eq, oo_eq = solve_boundary_defects(p_H2O_o, p_O2_o, c_R, K_H2O, K_O2)

    # Равновесный (нернстовский) потенциал E_eq [В]
    E_eq = (R_GAS * T_K / F_CONST) * np.log((ho_eq / oo_eq) / (hf_eq / of_eq))

    # Перенапряжение электродов (Батлер-Фольмер arcsinh)
    eta_H = (R_GAS * T_K / F_CONST) * np.arcsinh((i_A_cm2 * Rp_H * F_CONST) / (R_GAS * T_K)) if Rp_H > 0 else 0.0
    eta_O = (R_GAS * T_K / F_CONST) * np.arcsinh((i_A_cm2 * Rp_O * F_CONST) / (R_GAS * T_K)) if Rp_O > 0 else 0.0

    # Сдвиг граничных концентраций (если включен) через эффективное p_O2
    if use_arcsinh_shift and (Rp_H > 0 or Rp_O > 0):
        shift_f = np.clip(4.0 * F_CONST * eta_H / (R_GAS * T_K), -25.0, 25.0)
        shift_o = np.clip(-4.0 * F_CONST * eta_O / (R_GAS * T_K), -25.0, 25.0)
        pf, vf, hf, of = solve_boundary_defects(p_H2O_f, p_O2_f * np.exp(shift_f), c_R, K_H2O, K_O2)
        po, vo, ho, oo = solve_boundary_defects(p_H2O_o, p_O2_o * np.exp(shift_o), c_R, K_H2O, K_O2)
    else:
        pf, vf, hf, of = pf_eq, vf_eq, hf_eq, of_eq
        po, vo, ho, oo = po_eq, vo_eq, ho_eq, oo_eq

    # Молярные концентрации на границах [моль/м^3]
    cf_p, cf_v, cf_h = pf / Vm, vf / Vm, hf / Vm
    co_p, co_v, co_h = po / Vm, vo / Vm, ho / Vm

    # --- ШАГ 1: БЫСТРЫЙ АЛГЕБРАИЧЕСКИЙ РЕШАТЕЛЬ (ПОСТОЯННОЕ ПОЛЕ) ---
    def total_current_alg(dPhi_V):
        u1 = F_CONST * dPhi_V / (R_GAS * T_K)
        u2 = 2.0 * u1
        Jp = -(D_p / L_m) * bernoulli_flux(u1, cf_p, co_p)
        Jv = -(D_v / L_m) * bernoulli_flux(u2, cf_v, co_v)
        Jh = -(D_h / L_m) * bernoulli_flux(u1, cf_h, co_h)
        return F_CONST * (Jp + 2.0 * Jv + Jh) - i_A_m2

    # Поиск границ корня для dPhi
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
    ch_prof = profile_exp(x_norm, u1_alg, hf, ho)
    Phi_prof = dPhi_alg * x_norm
    E_field_prof = np.full_like(x_norm, -dPhi_alg / L_m)

    J_p = -(D_p / L_m) * bernoulli_flux(u1_alg, cf_p, co_p)
    J_v = -(D_v / L_m) * bernoulli_flux(u2_alg, cf_v, co_v)
    J_h = -(D_h / L_m) * bernoulli_flux(u1_alg, cf_h, co_h)
    bvp_converged = True

    # --- ШАГ 2: ТОЧНЫЙ 1D BVP РЕШАТЕЛЬ (ПЕРЕМЕННОЕ ПОЛЕ + ПУАССОН) ---
    if solver_mode == "bvp":
        Dp_nd = 1.0
        Dv_nd = D_v / D_p
        Dh_nd = D_h / D_p
        i_nd = i_A_m2 * L_m * Vm / (D_p * F_CONST)
        # Регуляризованный параметр Пуассона (как в MATLAB-коде)
        eps_reg = max(100.0 * Vm * R_GAS * T_K / (F_CONST ** 2 * L_m ** 2), 1e-4)

        Jp_nd0 = J_p * L_m * Vm / D_p
        Jv_nd0 = J_v * L_m * Vm / D_p
        Jh_nd0 = J_h * L_m * Vm / D_p

        def ode_rhs(x, y):
            # y = [cp, cv, ch, Jp, Jv, Jh, Phi_nd, dPhi_dx_nd]
            cp, cv, ch, Jp_n, Jv_n, Jh_n, Phi_n, dPhi_n = y
            dcp = -Jp_n / Dp_nd - 1.0 * cp * dPhi_n
            dcv = -Jv_n / Dv_nd - 2.0 * cv * dPhi_n
            dch = -Jh_n / Dh_nd - 1.0 * ch * dPhi_n
            zeros = np.zeros_like(x)
            d2Phi = -(cp + 2.0 * cv + ch - c_R) / eps_reg
            return np.vstack([dcp, dcv, dch, zeros, zeros, zeros, dPhi_n, d2Phi])

        def bc_func(ya, yb):
            return np.array([
                ya[0] - pf, ya[1] - vf, ya[2] - hf,
                yb[0] - po, yb[1] - vo, yb[2] - ho,
                ya[6] - 0.0,
                (yb[3] + 2.0 * yb[4] + yb[5]) - i_nd,
            ])

        y_init = np.vstack([
            cp_prof, cv_prof, ch_prof,
            np.full_like(x_norm, Jp_nd0),
            np.full_like(x_norm, Jv_nd0),
            np.full_like(x_norm, Jh_nd0),
            u1_alg * x_norm,
            np.full_like(x_norm, u1_alg),
        ])

        sol = solve_bvp(ode_rhs, bc_func, x_norm, y_init, tol=1e-4, max_nodes=1500)
        if sol.success:
            y_sol = sol.sol(x_norm)
            cp_prof = np.maximum(y_sol[0], 1e-18)
            cv_prof = np.maximum(y_sol[1], 1e-18)
            ch_prof = np.maximum(y_sol[2], 1e-18)
            J_p = float(np.mean(y_sol[3])) * D_p / (L_m * Vm)
            J_v = float(np.mean(y_sol[4])) * D_p / (L_m * Vm)
            J_h = float(np.mean(y_sol[5])) * D_p / (L_m * Vm)
            Phi_prof = y_sol[6] * (R_GAS * T_K / F_CONST)
            E_field_prof = -y_sol[7] * (R_GAS * T_K / (F_CONST * L_m))
            dPhi_final = float(Phi_prof[-1] - Phi_prof[0])
        else:
            bvp_converged = False
            dPhi_final = dPhi_alg
    else:
        dPhi_final = dPhi_alg

    # --- ВЫХОДНЫЕ ВЕЛИЧИНЫ ---
    co_prof = np.maximum(3.0 - cp_prof - cv_prof - ch_prof, 1e-12)

    # Парциальные проводимости по толщине [См/см]
    pref = (F_CONST ** 2 / (R_GAS * T_K * Vm)) * 1e-2  # См/м -> См/см
    sig_p_prof = pref * (1.0 ** 2) * cp_prof * D_p
    sig_v_prof = pref * (2.0 ** 2) * cv_prof * D_v
    sig_h_prof = pref * (1.0 ** 2) * ch_prof * D_h
    sig_tot_prof = sig_p_prof + sig_v_prof + sig_h_prof

    # Внутренние парциальные давления газов по толщине [атм]
    p_H2O_int = (cp_prof ** 2) / (co_prof * cv_prof * K_H2O)
    p_O2_int = ((ch_prof ** 2) / (co_prof * cv_prof * K_O2)) ** 2
    p_H2_int = p_H2O_int / (K_W * np.sqrt(np.maximum(p_O2_int, 1e-35)))

    # Парциальные плотности токов [А/см^2]
    i_p = (1.0 * F_CONST * J_p) * 1e-4
    i_v = (2.0 * F_CONST * J_v) * 1e-4
    i_h = (1.0 * F_CONST * J_h) * 1e-4
    i_tot = i_p + i_v + i_h

    # Напряжение на ячейке [В]
    # Если сдвиг границ включен, перенапряжение уже учтено в разности химических потенциалов h на границах;
    # иначе вычитаем eta_H + eta_O явно.
    if use_arcsinh_shift and (Rp_H > 0 or Rp_O > 0):
        E_cell_shifted = (R_GAS * T_K / F_CONST) * np.log((ho / oo) / (hf / of))
        V_cell = E_cell_shifted + dPhi_final
    else:
        V_cell = E_eq + dPhi_final - (eta_H + eta_O)

    # Газовые потоки и Фарадеевская эффективность (единая для PCFC и PCEC)
    J_H2, J_H2O, J_O2 = compute_gas_fluxes(J_p, J_v, p_H2_f, p_H2O_f, p_O2_f)
    i_H2_equiv = (2.0 * F_CONST * J_H2) * 1e-4  # А/см^2

    if i_A_cm2 > 1e-6:  # PCFC
        FE = (i_tot / i_H2_equiv) if abs(i_H2_equiv) > 1e-12 else 0.0
        mode_str = "PCFC (Fuel Cell)"
    elif i_A_cm2 < -1e-6:  # PCEC
        FE = (i_H2_equiv / i_tot) if abs(i_tot) > 1e-12 else 0.0
        mode_str = "PCEC (Electrolyzer)"
    else:  # OCV
        FE = 0.0
        mode_str = "OCV (Open Circuit)"

    # Числа переноса (по проводимости, осредненные по толщине, и по току)
    t_p_cond = float(np.mean(sig_p_prof / sig_tot_prof))
    t_v_cond = float(np.mean(sig_v_prof / sig_tot_prof))
    t_h_cond = float(np.mean(sig_h_prof / sig_tot_prof))

    # Тепловой баланс [Вт/см^2] и КПД по LHV
    # q_dot > 0: экзотермический (выделение тепла), q_dot < 0: эндотермический (поглощение)
    J_H2_cm2 = J_H2 * 1e-4
    q_dot = -i_tot * V_cell + J_H2_cm2 * abs(DH_W)
    V_TN_theo = abs(DH_W) / (2.0 * F_CONST)
    V_TN_act = max(FE, 0.0) * V_TN_theo if i_A_cm2 < -1e-6 else V_TN_theo

    P_elec = i_tot * V_cell  # Вт/см^2 (>0 в PCFC, <0 в PCEC)
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
        "T_C": T_C, "i": i_tot, "V_cell": V_cell, "E_eq": E_eq,
        "dPhi": dPhi_final, "eta_H": eta_H, "eta_O": eta_O,
        "i_p": i_p, "i_v": i_v, "i_h": i_h,
        "t_p": t_p_cond, "t_v": t_v_cond, "t_h": t_h_cond,
        "FE": FE * 100.0, "eff_LHV": eff_LHV * 100.0,
        "P_elec": P_elec, "q_dot": q_dot,
        "V_TN_theo": V_TN_theo, "V_TN_act": V_TN_act,
        "J_H2": J_H2_cm2, "J_H2O": J_H2O * 1e-4, "J_O2": J_O2 * 1e-4,
        # Профили по x/L
        "x_norm": x_norm, "cp": cp_prof, "cv": cv_prof, "ch": ch_prof, "co": co_prof,
        "Phi": Phi_prof, "E_field": E_field_prof,
        "sig_p": sig_p_prof, "sig_v": sig_v_prof, "sig_h": sig_h_prof, "sig_tot": sig_tot_prof,
        "p_H2O_int": p_H2O_int, "p_O2_int": p_O2_int, "p_H2_int": p_H2_int,
    }


# =============================================================================
# 3. БОКОВАЯ ПАНЕЛЬ (ВХОДНЫЕ ПАРАМЕТРЫ + НАСТРОЙКА ГРАФИКОВ)
# =============================================================================
st.sidebar.title("⚙️ Параметры модели")

preset_name = st.sidebar.selectbox("Материал электролита", list(MATERIAL_PRESETS.keys()))
mat_params = dict(MATERIAL_PRESETS[preset_name])

with st.sidebar.expander("🔬 Режим решателя и границы", expanded=True):
    solver_label = st.radio(
        "Метод решения NPP",
        ["Быстрый (алгебраический, постоянное поле)", "Точный (1D BVP, переменное поле)"],
        index=0,
    )
    solver_mode = "fast" if "Быстрый" in solver_label else "bvp"
    use_arcsinh_shift = st.checkbox(
        "Сдвиг граничных концентраций через Rp (arcsinh)",
        value=True,
        help="Учитывает влияние поляризации электродов на ток электронной утечки."
    )

with st.sidebar.expander("⚡ Режимные и геометрические параметры", expanded=True):
    i_work = st.number_input("Плотность тока i [А/см²] (<0 PCEC, >0 PCFC)", -2.5, 2.5, -0.8, 0.05, format="%.2f")
    T_work = st.slider("Температура T [°C]", 450, 750, 600, 5)
    L_work = st.number_input("Толщина электролита L [мкм]", 1.0, 500.0, 10.0, 1.0)
    c_R_work = st.slider("Концентрация допанта c_R [отн. ед.]", 0.10, 0.30, float(mat_params["default_cR"]), 0.01)
    col_rp1, col_rp2 = st.columns(2)
    Rp_H = col_rp1.number_input("Rp(H₂) [Ом·см²]", 0.0, 2.0, 0.05, 0.01, format="%.3f")
    Rp_O = col_rp2.number_input("Rp(O₂) [Ом·см²]", 0.0, 2.0, 0.05, 0.01, format="%.3f")

with st.sidebar.expander("💨 Состав газов на электродах [атм]", expanded=True):
    binary_fuel = st.checkbox("Двухкомпонентная смесь H₂/H₂O (p_H₂O = 1 - p_H₂)", value=True)
    p_H2_f = st.slider("p(H₂) водородный электрод", 0.05, 0.995, 0.97, 0.01)
    if binary_fuel:
        p_H2O_f = max(1.0 - p_H2_f, 1e-4)
        st.caption(f"p(H₂O) водородный = {p_H2O_f:.3f} атм")
    else:
        p_H2O_f = st.slider("p(H₂O) водородный электрод", 0.005, 0.50, 0.03, 0.005)
    p_H2O_o = st.slider("p(H₂O) кислородный электрод (увлажнение воздуха)", 0.01, 0.95, 0.50, 0.01)
    st.caption(f"p(O₂) кислородный = {0.21 * (1.0 - p_H2O_o):.4f} атм")

with st.sidebar.expander("🧪 Ручная правка параметров материала", expanded=False):
    mat_params["dH_H2O"] = st.number_input("ΔH°(H₂O) [Дж/моль]", value=float(mat_params["dH_H2O"]), step=1000.0)
    mat_params["dS_H2O"] = st.number_input("ΔS°(H₂O) [Дж/моль·К]", value=float(mat_params["dS_H2O"]), step=1.0)
    mat_params["dH_O2"] = st.number_input("ΔH°(O₂) [Дж/моль]", value=float(mat_params["dH_O2"]), step=1000.0)
    mat_params["dS_O2"] = st.number_input("ΔS°(O₂) [Дж/моль·К]", value=float(mat_params["dS_O2"]), step=1.0)
    mat_params["E_p"] = st.number_input("E_act(OH•) [Дж/моль]", value=float(mat_params["E_p"]), step=500.0)
    mat_params["E_h"] = st.number_input("E_act(O•) [Дж/моль]", value=float(mat_params["E_h"]), step=500.0)

with st.sidebar.expander("🎨 Настройка графиков (для статьи)", expanded=False):
    fs_label = st.slider("Размер подписей осей", 8, 18, 11)
    fs_tick = st.slider("Размер чисел на осях", 7, 16, 10)
    fs_title = st.slider("Размер заголовка", 9, 20, 12)
    fs_legend = st.slider("Размер легенды", 7, 16, 9)
    lw = st.slider("Толщина линий", 1.0, 4.0, 2.0, 0.25)
    show_markers = st.checkbox("Показывать маркеры на линиях", value=False)
    ms = st.slider("Размер маркеров", 2, 10, 5)
    show_grid = st.checkbox("Тонкая сетка", value=True)
    cmap_multi = st.selectbox("Палитра для серий T", ["coolwarm", "viridis", "plasma", "turbo", "tab10"])
    cmap_heat = st.selectbox("Палитра для тепловых карт", ["inferno", "viridis", "magma", "RdYlBu_r", "plasma"])
    legend_title = st.text_input("Заголовок легенды (для серий T)", "Temperature")
    legend_prefix = st.text_input("Формат подписи в легенде", "T = {T} °C")

# Применение стиля Matplotlib для научных публикаций
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


def render_plot_with_download(fig, filename_base: str):
    """Отображает график и кнопки скачивания PNG (600 DPI) и SVG."""
    st.pyplot(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=600)
    c1.download_button("⬇️ PNG (600 DPI)", buf_png.getvalue(), f"{filename_base}.png", "image/png", key=f"png_{filename_base}")
    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg")
    c2.download_button("⬇️ SVG (вектор)", buf_svg.getvalue(), f"{filename_base}.svg", "image/svg+xml", key=f"svg_{filename_base}")
    plt.close(fig)


def style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)


# =============================================================================
# 4. ЗАГОЛОВОК И КЛЮЧЕВЫЕ МЕТРИКИ В РАБОЧЕЙ ТОЧКЕ
# =============================================================================
st.title("⚡ Единая модель электро-, массо- и теплопереноса в мембране PCEC / PCFC")
st.caption(f"Материал: **{preset_name}** | Решатель: **{solver_label}** | Носители: OH•, V••, O•")

res_pt = solve_single_point(
    i_work, T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
    mat_params, solver_mode=solver_mode, use_arcsinh_shift=use_arcsinh_shift
)

if solver_mode == "bvp" and not res_pt["bvp_converged"]:
    st.warning("1D BVP решатель не сошёлся в данной точке (использовано алгебраическое приближение).")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Режим", res_pt["mode"].split()[0])
m2.metric("V_cell [В]", f"{res_pt['V_cell']:.4f}", delta=f"E_eq = {res_pt['E_eq']:.3f} В")
m3.metric("FE [%]", f"{res_pt['FE']:.2f}%")
m4.metric("КПД (LHV) [%]", f"{res_pt['eff_LHV']:.2f}%")
m5.metric("Тепло q̇ [Вт/см²]", f"{res_pt['q_dot']:.3f}", delta="Экзо" if res_pt["q_dot"] > 0 else "Эндо")
m6.metric("Ток утечки i_h [А/см²]", f"{res_pt['i_h']:.4f}")

# =============================================================================
# 5. ВКЛАДКИ С ГРАФИКАМИ
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Профили по толщине (x/L)",
    "2. ВАХ, токи и тепло (при T)",
    "3. Мультитемпературный анализ",
    "4. Тепловые карты (2D)"
])

mk = "o" if show_markers else None

# -----------------------------------------------------------------------------
# ВКЛАДКА 1: 4 ПРОФИЛЯ ПО ТОЛЩИНЕ МЕМБРАНЫ
# -----------------------------------------------------------------------------
with tab1:
    st.subheader(f"Пространственные распределения по толщине мембраны (T = {T_work} °C, i = {i_work:.2f} А/см²)")
    c1, c2 = st.columns(2)
    x = res_pt["x_norm"]

    with c1:
        # График 1: Концентрации дефектов
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, res_pt["cp"], lw=lw, marker=mk, ms=ms, label=r"$[\mathrm{OH_O^\bullet}]_L$ (протоны)")
        ax.plot(x, res_pt["cv"], lw=lw, marker=mk, ms=ms, label=r"$[\mathrm{V_O^{\bullet\bullet}}]_L$ (вакансии)")
        ax.plot(x, res_pt["ch"], lw=lw, marker=mk, ms=ms, label=r"$[\mathrm{O_O^\bullet}]_L$ (поляроны)")
        ax.set_yscale("log")
        style_ax(ax, r"Безразмерная координата $x / L$", "Концентрация на форм. ед. (log)", "1. Профили концентраций дефектов")
        ax.legend()
        render_plot_with_download(fig, "01_defect_profiles")

        # График 3: Парциальные проводимости
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, res_pt["sig_p"] * 1e3, lw=lw, label=r"$\sigma_p$ (протоны)")
        ax.plot(x, res_pt["sig_h"] * 1e3, lw=lw, label=r"$\sigma_h$ (поляроны)")
        ax.plot(x, res_pt["sig_v"] * 1e3, lw=lw, label=r"$\sigma_v$ (вакансии)")
        ax.plot(x, res_pt["sig_tot"] * 1e3, lw=lw, ls="--", color="black", label=r"$\sigma_{\mathrm{tot}}$")
        style_ax(ax, r"Безразмерная координата $x / L$", r"Проводимость [мСм/см]", "3. Парциальные проводимости")
        ax.legend()
        render_plot_with_download(fig, "03_conductivities")

    with c2:
        # График 2: Потенциал и электрическое поле
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, res_pt["Phi"], lw=lw, color="tab:blue", label=r"Потенциал $\Phi(x)$")
        style_ax(ax, r"Безразмерная координата $x / L$", r"Потенциал $\Phi$ [В]", "2. Электростатический потенциал и поле")
        ax2 = ax.twinx()
        ax2.plot(x, res_pt["E_field"] * 1e-5, lw=lw, ls="--", color="tab:red", label=r"Поле $E(x)$")
        ax2.set_ylabel(r"Напряжённость поля $E$ [$10^5$ В/м]", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red", direction="in")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        render_plot_with_download(fig, "02_potential_field")

        # График 4: Внутренние парциальные давления газов
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(x, np.log10(np.maximum(res_pt["p_O2_int"], 1e-35)), lw=lw, label=r"$\log_{10} p_{\mathrm{O_2}}$")
        ax.plot(x, np.log10(np.maximum(res_pt["p_H2O_int"], 1e-35)), lw=lw, label=r"$\log_{10} p_{\mathrm{H_2O}}$")
        ax.plot(x, np.log10(np.maximum(res_pt["p_H2_int"], 1e-35)), lw=lw, label=r"$\log_{10} p_{\mathrm{H_2}}$")
        ax.axhline(0.0, color="gray", ls=":", lw=1.0, label="1 атм (порог)")
        style_ax(ax, r"Безразмерная координата $x / L$", r"$\log_{10}(p_i / \mathrm{атм})$", "4. Внутренние парциальные давления")
        ax.legend()
        render_plot_with_download(fig, "04_internal_pressures")

# -----------------------------------------------------------------------------
# ВКЛАДКА 2: РАЗВЁРТКА ПО ТОКУ (PCEC -> OCV -> PCFC) ПРИ T_work
# -----------------------------------------------------------------------------
with tab2:
    st.subheader(f"Вольт-амперные и энергетические характеристики при T = {T_work} °C")
    col_i1, col_i2, col_i3 = st.columns(3)
    i_min = col_i1.number_input("Мин. ток (PCEC) [А/см²]", -3.0, -0.1, -1.5, 0.1)
    i_max = col_i2.number_input("Макс. ток (PCFC) [А/см²]", 0.1, 3.0, 1.0, 0.1)
    n_pts = col_i3.slider("Число точек развёртки", 20, 80, 41)

    i_grid = np.linspace(i_min, i_max, n_pts)
    sweep_rows = []
    for iv in i_grid:
        # Для быстрой развёртки используем выбранный решатель
        r = solve_single_point(
            float(iv), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
            mat_params, solver_mode=solver_mode, use_arcsinh_shift=use_arcsinh_shift, n_grid=40
        )
        sweep_rows.append({k: r[k] for k in [
            "i", "V_cell", "E_eq", "dPhi", "eta_H", "eta_O",
            "i_p", "i_v", "i_h", "t_p", "t_v", "t_h",
            "FE", "eff_LHV", "P_elec", "q_dot", "V_TN_theo", "V_TN_act"
        ]})
    df_i = pd.DataFrame(sweep_rows)

    c1, c2 = st.columns(2)
    with c1:
        # График 5: ВАХ и мощность PCFC
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["V_cell"], lw=lw, marker=mk, ms=ms, color="navy", label=r"$V_{\mathrm{cell}}$")
        ax.axvline(0.0, color="black", ls=":", lw=1.0)
        style_ax(ax, r"Плотность тока $i$ [А/см²]", r"Напряжение $V_{\mathrm{cell}}$ [В]", "5. ВАХ (PCEC / PCFC) и мощность")
        ax2 = ax.twinx()
        df_fc = df_i[df_i["i"] >= 0]
        ax2.plot(df_fc["i"], df_fc["P_elec"], lw=lw, ls="--", color="crimson", label="Мощность PCFC")
        ax2.set_ylabel(r"Удельная мощность $P$ [Вт/см²]", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson", direction="in")
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, loc="best")
        render_plot_with_download(fig, "05_IV_power")

        # График 7: Парциальные плотности токов
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["i_p"], lw=lw, label=r"$i_p$ (протоны)")
        ax.plot(df_i["i"], df_i["i_h"], lw=lw, label=r"$i_h$ (поляроны / утечка)")
        ax.plot(df_i["i"], df_i["i_v"], lw=lw, label=r"$i_v$ (вакансии)")
        ax.axhline(0, color="black", ls=":", lw=0.8)
        ax.axvline(0, color="black", ls=":", lw=0.8)
        style_ax(ax, r"Полный ток $i$ [А/см²]", r"Парциальный ток $i_k$ [А/см²]", "7. Парциальные плотности токов")
        ax.legend()
        render_plot_with_download(fig, "07_partial_currents")

        # График 9: Тепловой баланс и термонейтральное напряжение
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["q_dot"], lw=lw, color="darkorange", label=r"Тепловыделение $\dot{q}$")
        ax.axhline(0.0, color="black", ls="--", lw=1.0, label="Термонейтраль ($\dot{q}=0$)")
        ax.fill_between(df_i["i"], df_i["q_dot"], 0, where=(df_i["q_dot"] > 0), color="red", alpha=0.12)
        ax.fill_between(df_i["i"], df_i["q_dot"], 0, where=(df_i["q_dot"] < 0), color="blue", alpha=0.12)
        style_ax(ax, r"Плотность тока $i$ [А/см²]", r"Удельный тепловой поток $\dot{q}$ [Вт/см²]", "9. Тепловой баланс (эндо / экзо)")
        ax.legend()
        render_plot_with_download(fig, "09_heat_balance")

    with c2:
        # График 6: Разложение потерь напряжения
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_i["i"], df_i["dPhi"], lw=lw, label=r"Мембрана $\Delta\Phi$")
        ax.plot(df_i["i"], -df_i["eta_H"], lw=lw, label=r"Поляризация $-\eta_{\mathrm{H_2}}$")
        ax.plot(df_i["i"], -df_i["eta_O"], lw=lw, ls="--", label=r"Поляризация $-\eta_{\mathrm{O_2}}$")
        ax.axhline(0, color="black", ls=":", lw=0.8)
        style_ax(ax, r"Плотность тока $i$ [А/см²]", r"Составляющие напряжения [В]", "6. Разложение омических и электродных потерь")
        ax.legend()
        render_plot_with_download(fig, "06_voltage_losses")

        # График 8: Числа переноса и Фарадеевская эффективность
        fig, ax = plt.subplots(figsize=(6, 4.2))
        df_ec = df_i[df_i["i"] < -0.05]
        df_fc = df_i[df_i["i"] > 0.05]
        ax.plot(df_ec["i"], np.clip(df_ec["FE"], 0, 105), lw=lw, color="tab:blue", label="FE (PCEC)")
        ax.plot(df_fc["i"], np.clip(df_fc["FE"], 0, 105), lw=lw, color="tab:green", label="FE (PCFC)")
        ax.plot(df_i["i"], df_i["t_p"] * 100.0, lw=lw, ls="--", color="black", label=r"$t_p$ (среднее)")
        ax.set_ylim(0, 105)
        style_ax(ax, r"Плотность тока $i$ [А/см²]", "Эффективность / число переноса [%]", "8. Фарадеевская эффективность и t_p")
        ax.legend()
        render_plot_with_download(fig, "08_FE_transport")

        # График 10: Энергетический КПД по LHV
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(df_ec["i"], np.clip(df_ec["eff_LHV"], 0, 105), lw=lw, color="teal", label="КПД PCEC (LHV)")
        ax.plot(df_fc["i"], np.clip(df_fc["eff_LHV"], 0, 105), lw=lw, color="purple", label="КПД PCFC (LHV)")
        ax.set_ylim(0, 105)
        style_ax(ax, r"Плотность тока $i$ [А/см²]", "Энергетический КПД (LHV) [%]", "10. Энергетический КПД по LHV")
        ax.legend()
        render_plot_with_download(fig, "10_energy_efficiency")

    st.download_button("📥 Скачать таблицу развёртки по току (CSV)", df_i.to_csv(index=False).encode("utf-8"), "iv_sweep.csv", "text/csv")

# -----------------------------------------------------------------------------
# ВКЛАДКА 3: МУЛЬТИТЕМПЕРАТУРНЫЙ АНАЛИЗ
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Динамика характеристик при различных температурах")
    default_temps = [500, 550, 600, 650, 700]
    selected_temps = st.multiselect(
        "Выберите температуры для анализа [°C]",
        options=list(range(450, 755, 25)),
        default=default_temps,
    )
    selected_temps = sorted(selected_temps)

    if len(selected_temps) < 2:
        st.info("Выберите хотя бы 2 температуры для построения сравнительных графиков.")
    else:
        cmap = plt.get_cmap(cmap_multi)
        colors = [cmap(k / max(len(selected_temps) - 1, 1)) for k in range(len(selected_temps))]

        i_multi = np.linspace(i_min, i_max, 35)
        multi_iv = {}
        t_summary = []

        for T_val in selected_temps:
            rows_T = []
            for iv in i_multi:
                r = solve_single_point(
                    float(iv), T_val, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work,
                    mat_params, solver_mode="fast", use_arcsinh_shift=use_arcsinh_shift, n_grid=25
                )
                rows_T.append(r)
            multi_iv[T_val] = pd.DataFrame(rows_T)

            # Точка OCV и рабочая точка i_work
            r_ocv = solve_single_point(0.0, T_val, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat_params, solver_mode="fast", use_arcsinh_shift=use_arcsinh_shift, n_grid=25)
            r_wk = solve_single_point(i_work, T_val, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat_params, solver_mode="fast", use_arcsinh_shift=use_arcsinh_shift, n_grid=25)
            t_summary.append({
                "T_C": T_val, "E_eq": r_ocv["E_eq"], "OCV": r_ocv["V_cell"],
                "RdOCV_mV": (r_ocv["E_eq"] - r_ocv["V_cell"]) * 1e3,
                "i_leak_OCV": abs(r_ocv["i_h"]), "t_p_wk": r_wk["t_p"] * 100.0,
                "FE_wk": r_wk["FE"], "eff_wk": r_wk["eff_LHV"],
                "q_dot_wk": r_wk["q_dot"], "V_TN_act": r_wk["V_TN_act"], "V_TN_theo": r_wk["V_TN_theo"]
            })
        df_T = pd.DataFrame(t_summary)

        c1, c2 = st.columns(2)
        with c1:
            # График 11: Семейство ВАХ
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for idx, T_val in enumerate(selected_temps):
                df_t = multi_iv[T_val]
                lbl = legend_prefix.replace("{T}", str(T_val))
                ax.plot(df_t["i"], df_t["V_cell"], lw=lw, marker=mk, ms=ms, color=colors[idx], label=lbl)
            ax.axvline(0, color="black", ls=":", lw=0.8)
            style_ax(ax, r"Плотность тока $i$ [А/см²]", r"Напряжение $V_{\mathrm{cell}}$ [В]", "11. Семейство ВАХ при разных T")
            ax.legend(title=legend_title)
            render_plot_with_download(fig, "11_multiT_IV")

            # График 13: OCV, E_eq и потери RdOCV(T)
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.plot(df_T["T_C"], df_T["E_eq"], lw=lw, marker="o", ms=ms, label=r"$E_{\mathrm{eq}}$")
            ax.plot(df_T["T_C"], df_T["OCV"], lw=lw, marker="s", ms=ms, label="OCV")
            style_ax(ax, "Температура T [°C]", "Потенциал [В]", "13. E_eq, OCV и потери RdOCV от T")
            ax2 = ax.twinx()
            ax2.plot(df_T["T_C"], df_T["RdOCV_mV"], lw=lw, ls="--", marker="^", ms=ms, color="crimson", label="RdOCV [мВ]")
            ax2.set_ylabel("Снижение OCV (RdOCV) [мВ]", color="crimson")
            ax2.tick_params(axis="y", labelcolor="crimson", direction="in")
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, loc="best")
            render_plot_with_download(fig, "13_multiT_OCV")

        with c2:
            # График 12: Семейство FE(i, T)
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for idx, T_val in enumerate(selected_temps):
                df_t = multi_iv[T_val]
                lbl = legend_prefix.replace("{T}", str(T_val))
                df_ec_t = df_t[df_t["i"] < -0.05]
                df_fc_t = df_t[df_t["i"] > 0.05]
                ax.plot(df_ec_t["i"], np.clip(df_ec_t["FE"], 0, 105), lw=lw, color=colors[idx], label=lbl)
                ax.plot(df_fc_t["i"], np.clip(df_fc_t["FE"], 0, 105), lw=lw, ls="--", color=colors[idx])
            ax.set_ylim(0, 105)
            style_ax(ax, r"Плотность тока $i$ [А/см²]", "Фарадеевская эффективность FE [%]", "12. Семейство FE(i) при разных T")
            ax.legend(title=legend_title)
            render_plot_with_download(fig, "12_multiT_FE")

            # График 14: Тепловыделение, V_TN_act и КПД от T
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.plot(df_T["T_C"], df_T["FE_wk"], lw=lw, marker="o", ms=ms, label="FE [%]")
            ax.plot(df_T["T_C"], df_T["eff_wk"], lw=lw, marker="s", ms=ms, label="КПД LHV [%]")
            style_ax(ax, "Температура T [°C]", "Эффективность [%]", f"14. Динамика FE, КПД и q̇ (i={i_work:.2f} А/см²)")
            ax2 = ax.twinx()
            ax2.plot(df_T["T_C"], df_T["q_dot_wk"], lw=lw, ls="--", marker="^", ms=ms, color="darkorange", label=r"$\dot{q}$ [Вт/см²]")
            ax2.set_ylabel(r"Тепловыделение $\dot{q}$ [Вт/см²]", color="darkorange")
            ax2.tick_params(axis="y", labelcolor="darkorange", direction="in")
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, loc="best")
            render_plot_with_download(fig, "14_multiT_thermal_eff")

        st.download_button("📥 Скачать сводную таблицу по температурам (CSV)", df_T.to_csv(index=False).encode("utf-8"), "multiT_summary.csv", "text/csv")

# -----------------------------------------------------------------------------
# ВКЛАДКА 4: ТЕПЛОВЫЕ КАРТЫ (2D CONTOURF / HEATMAPS)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("2D Тепловые карты (параметрические карты с изолиниями)")
    res_hm = st.slider("Разрешение сетки тепловых карт (N × N)", 12, 35, 20)

    T_hm = np.linspace(480, 700, res_hm)
    i_hm_ec = np.linspace(-1.8, -0.1, res_hm)
    L_hm = np.linspace(3.0, 50.0, res_hm)

    FE_map = np.zeros((res_hm, res_hm))
    Q_map = np.zeros((res_hm, res_hm))
    Eff_L_map = np.zeros((res_hm, res_hm))
    pO2_map = np.zeros((res_hm, res_hm))

    for idx_t, tv in enumerate(T_hm):
        for idx_i, iv in enumerate(i_hm_ec):
            r = solve_single_point(float(iv), float(tv), L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat_params, solver_mode="fast", use_arcsinh_shift=use_arcsinh_shift, n_grid=15)
            FE_map[idx_t, idx_i] = np.clip(r["FE"], 0, 100)
            Q_map[idx_t, idx_i] = r["q_dot"]

    for idx_l, lv in enumerate(L_hm):
        for idx_i, iv in enumerate(i_hm_ec):
            r = solve_single_point(float(iv), T_work, float(lv), Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat_params, solver_mode="fast", use_arcsinh_shift=use_arcsinh_shift, n_grid=15)
            Eff_L_map[idx_l, idx_i] = np.clip(r["eff_LHV"], 0, 100)

    x_hm = np.linspace(0.0, 1.0, res_hm)
    for idx_i, iv in enumerate(i_hm_ec):
        r = solve_single_point(float(iv), T_work, L_work, Rp_H, Rp_O, p_H2_f, p_H2O_f, p_H2O_o, c_R_work, mat_params, solver_mode="fast", use_arcsinh_shift=use_arcsinh_shift, n_grid=res_hm)
        pO2_map[:, idx_i] = np.log10(np.maximum(r["p_O2_int"], 1e-30))

    c1, c2 = st.columns(2)
    with c1:
        # График 15: Тепловая карта FE(T, i)
        fig, ax = plt.subplots(figsize=(6, 4.4))
        cf = ax.contourf(i_hm_ec, T_hm, FE_map, levels=20, cmap=cmap_heat)
        cs = ax.contour(i_hm_ec, T_hm, FE_map, levels=8, colors="white", linewidths=0.7, alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=fs_tick - 1, fmt="%.0f%%")
        fig.colorbar(cf, ax=ax, label="FE [%]")
        style_ax(ax, r"Плотность тока PCEC $i$ [А/см²]", "Температура T [°C]", "15. Карта Фарадеевской эффективности FE(T, i)")
        render_plot_with_download(fig, "15_heatmap_FE_Ti")

        # График 17: Тепловая карта КПД(L, i) — выбор оптимальной толщины
        fig, ax = plt.subplots(figsize=(6, 4.4))
        cf = ax.contourf(i_hm_ec, L_hm, Eff_L_map, levels=20, cmap=cmap_heat)
        cs = ax.contour(i_hm_ec, L_hm, Eff_L_map, levels=8, colors="white", linewidths=0.7, alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=fs_tick - 1, fmt="%.0f%%")
        fig.colorbar(cf, ax=ax, label="КПД LHV [%]")
        style_ax(ax, r"Плотность тока PCEC $i$ [А/см²]", r"Толщина электролита $L$ [мкм]", f"17. Оптимум толщины: КПД(L, i) при {T_work} °C")
        render_plot_with_download(fig, "17_heatmap_Eff_Li")

    with c2:
        # График 16: Тепловая карта теплового баланса q_dot(T, i)
        fig, ax = plt.subplots(figsize=(6, 4.4))
        vmax = max(abs(Q_map.min()), abs(Q_map.max()), 0.1)
        cf = ax.contourf(i_hm_ec, T_hm, Q_map, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        cs0 = ax.contour(i_hm_ec, T_hm, Q_map, levels=[0.0], colors="black", linewidths=2.0)
        ax.clabel(cs0, inline=True, fontsize=fs_tick, fmt="q̇=0 (TN)")
        fig.colorbar(cf, ax=ax, label=r"Тепловой поток $\dot{q}$ [Вт/см²]")
        style_ax(ax, r"Плотность тока PCEC $i$ [А/см²]", "Температура T [°C]", "16. Тепловая карта q̇(T, i) и линия термонейтрали")
        render_plot_with_download(fig, "16_heatmap_Q_Ti")

        # График 18: Тепловая карта внутреннего давления O2(x/L, i)
        fig, ax = plt.subplots(figsize=(6, 4.4))
        cf = ax.contourf(i_hm_ec, x_hm, pO2_map, levels=20, cmap=cmap_heat)
        cs = ax.contour(i_hm_ec, x_hm, pO2_map, levels=6, colors="white", linewidths=0.7, alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=fs_tick - 1, fmt="%.1f")
        fig.colorbar(cf, ax=ax, label=r"$\log_{10}(p_{\mathrm{O_2}} / \mathrm{атм})$")
        style_ax(ax, r"Плотность тока PCEC $i$ [А/см²]", r"Координата в мембране $x / L$", f"18. Карта внутреннего давления O₂ ({T_work} °C)")
        render_plot_with_download(fig, "18_heatmap_pO2_xi")
