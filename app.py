import os
import io
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import brentq, least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import streamlit as st

# ==============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="PCEC Electrochemistry Simulator (BZCYYb4411)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Глобальные физические константы
R_GAS = 8.314        # Дж / (моль * К)
F_CONST = 96485.0    # Кл / моль
VM_4411 = 5.1323e-5  # Молярный объем BZCYYb4411, м^3 / моль


# ==============================================================================
# 1. ГРАНИЧНЫЕ УСЛОВИЯ НА ЭЛЕКТРОДАХ (Водородный и Кислородный)
# ==============================================================================

def _charge_balance_residual(Cv, electrode, Ph2, Ph2o, Po2, Cd, params):
    """Невязка уравнения электронейтральности в равновесии (OCV) на границе."""
    Kph2o = params['Kph2o']
    Kpo2 = params['Kpo2']
    Kph2 = params['Kph2']
    Kpeh = params['Kpeh']

    a = 1.0 / (Kph2o * Ph2o * Cv)
    b = 1.0
    c = Cv - 3.0
    disc = max(b**2 - 4.0 * a * c, 0.0)
    Coh = (-b + np.sqrt(disc)) / (2.0 * a)
    Co = (Coh**2) / (Kph2o * Ph2o * Cv)

    if electrode == 'oxygen':
        Ch = np.sqrt(Cv * Kpo2 * np.sqrt(Po2) / Co)
        Ce = Kpeh / Ch
    elif electrode == 'hydrogen':
        Ce = Kph2 * np.sqrt(Ph2) * Co / Coh
        Ch = Kpeh / Ce
    else:
        raise ValueError("electrode must be 'oxygen' or 'hydrogen'")

    return 2.0 * Cv + Coh + Ch - Ce - Cd


def _find_bracket_and_solve(func, a_min=1e-12, a_max=1.5, n_scan=2000):
    """Надежный поиск корня на интервале (замена initial_guess_* + fzero)."""
    grid = np.linspace(a_min, a_max, n_scan)
    vals = np.array([func(x) for x in grid])
    sign_changes = np.where(vals[:-1] * vals[1:] <= 0)[0]
    if len(sign_changes) == 0:
        raise RuntimeError("Не удалось локализовать корень уравнения электронейтральности на границе.")
    idx = sign_changes[0]
    root_val, res = brentq(func, grid[idx], grid[idx + 1], xtol=1e-14, rtol=1e-14, full_output=True)
    if not res.converged:
        raise RuntimeError("brentq не сошелся при расчете граничных условий.")
    return root_val


def compute_boundary_concentrations(electrode, ic, TC, Rp, Ph2, Ph2o, Po2, Cd, params):
    """
    Расчет безразмерных граничных концентраций (C * Vm) с учетом
    поляризационного сопротивления электрода Rp при плотности тока ic (А/см^2).
    """
    TK = TC + 273.15
    Kph2o = params['Kph2o']
    Kpo2 = params['Kpo2']
    Kph2 = params['Kph2']
    Kpeh = params['Kpeh']

    Cv = _find_bracket_and_solve(
        lambda x: _charge_balance_residual(x, electrode, Ph2, Ph2o, Po2, Cd, params)
    )

    a = 1.0 / (Kph2o * Ph2o * Cv)
    Coh = (-1.0 + np.sqrt(max(1.0 - 4.0 * a * (Cv - 3.0), 0.0))) / (2.0 * a)
    Co = (Coh**2) / (Kph2o * Ph2o * Cv)

    if electrode == 'oxygen':
        Ch = np.sqrt(Cv * Kpo2 * np.sqrt(Po2) / Co)
        Ce = Kpeh / Ch
        Ce_b = Ce * np.exp(+np.arcsinh(ic * Rp * F_CONST / (R_GAS * TK)))
    else:
        Ce = Kph2 * np.sqrt(Ph2) * Co / Coh
        Ch = Kpeh / Ce
        Ce_b = Ce * np.exp(-np.arcsinh(ic * Rp * F_CONST / (R_GAS * TK)))

    Ch_b = Kpeh / Ce_b
    Cv_b = Ce_b + Cd - Ch_b - (3.0 - Co)
    Cp_b = 3.0 - Co - Cv_b
    PO2_b = (Co * (Ch_b**2) / (Kpo2 * Cv_b))**2

    return {
        'Ce': Ce_b, 'Ch': Ch_b, 'Cp': Cp_b, 'Cv': Cv_b, 'Co': Co, 'PO2': PO2_b
    }


# ==============================================================================
# 2. СТАЦИОНАРНЫЙ РЕШАТЕЛЬ НЕРНСТА-ПЛАНКА-ПУАССОНА
# ==============================================================================

def build_jacobian_sparsity(num_grid):
    """
    Строит разреженную матрицу связности якобиана (блочно-трехдиагональная +
    связь правой границы через PhiOH).
    """
    n_vars = 4 * num_grid
    sparsity = lil_matrix((n_vars, n_vars), dtype=int)
    for i in range(num_grid):
        for j in range(max(0, i - 1), min(num_grid, i + 2)):
            sparsity[4*i : 4*(i+1), 4*j : 4*(j+1)] = 1
    # Правый граничный узел зависит от N-2 при вычислении PhiOH
    sparsity[4*(num_grid-1):4*num_grid, 4*(num_grid-2):4*num_grid] = 1
    return sparsity.tocsc()


def compute_right_boundary_potential(y_last, y_prev, icg, bc_O, params):
    """
    Вычисление безразмерного потенциала на правой границе (PhiOH)
    из условия равенства полного электрического тока заданному току icg.
    """
    Dh, De, Dv, Dp = params['Dh'], params['De'], params['Dv'], params['Dp']
    zh, ze, zv, zp = 1.0, -1.0, 2.0, 1.0
    dx = params['dx']
    Kpeh = params['Kpeh']

    Ch_N = y_last[0]**2
    Ce_N = Kpeh / Ch_N
    Cv_N = y_last[1]**2
    Cp_N = y_last[2]**2

    Ch_prev = y_prev[0]**2
    Ce_prev = Kpeh / Ch_prev
    Cv_prev = y_prev[1]**2
    Cp_prev = y_prev[2]**2

    sum_CF = (
        Dh * zh * (bc_O['Ch'] - Ch_prev)
        + De * ze * (bc_O['Ce'] - Ce_prev)
        + Dv * zv * (bc_O['Cv'] - Cv_prev)
        + Dp * zp * (bc_O['Cp'] - Cp_prev)
    )
    sigma_N = (
        Dh * (zh**2) * Ch_N
        + De * (ze**2) * Ce_N
        + Dv * (zv**2) * Cv_N
        + Dp * (zp**2) * Cp_N
    )
    PhiOH = y_prev[3] - 2.0 * dx * (icg + sum_CF / (2.0 * dx)) / sigma_N
    return PhiOH


def steady_residual(y_flat, icg, bc_H, bc_O, params):
    """Вектор невязок стационарной системы Нернста-Планка-Пуассона."""
    N = params['NumGrid']
    dx = params['dx']
    Dh, De, Dv, Dp = params['Dh'], params['De'], params['Dv'], params['Dp']
    zh, ze, zv, zp = 1.0, -1.0, 2.0, 1.0
    Kpeh = params['Kpeh']
    eps_term = params['epsilon0'] * params['epsilonr']
    Cd = params['Cd']

    y = y_flat.reshape((N, 4))
    res = np.zeros_like(y)

    # Переход от корней к безразмерным концентрациям (гарантия C >= 0)
    Ch = y[:, 0]**2
    Ce = Kpeh / Ch
    Cv = y[:, 1]**2
    Cp = y[:, 2]**2
    Phi = y[:, 3]

    # Граничные значения концентраций и потенциалов
    PhiH = 0.0
    PhiOH = compute_right_boundary_potential(y[-1], y[-2], icg, bc_O, params)

    Ch_ext = np.concatenate(([bc_H['Ch']], Ch, [bc_O['Ch']]))
    Ce_ext = np.concatenate(([bc_H['Ce']], Ce, [bc_O['Ce']]))
    Cv_ext = np.concatenate(([bc_H['Cv']], Cv, [bc_O['Cv']]))
    Cp_ext = np.concatenate(([bc_H['Cp']], Cp, [bc_O['Cp']]))
    Phi_ext = np.concatenate(([PhiH], Phi, [PhiOH]))

    def flux_div(C_ext, D_val, z_val):
        diff_right = 2.0 * D_val * (C_ext[2:] - C_ext[1:-1])
        diff_left = 2.0 * D_val * (C_ext[1:-1] - C_ext[:-2])

        migr_right = z_val * D_val * (C_ext[2:] + C_ext[1:-1]) * (Phi_ext[2:] - Phi_ext[1:-1])
        migr_left = z_val * D_val * (C_ext[:-2] + C_ext[1:-1]) * (Phi_ext[1:-1] - Phi_ext[:-2])

        return (diff_right - diff_left) + (migr_right - migr_left)

    # 1. Сохранение электронного заряда (дырки + электроны)
    res[:, 0] = zh * flux_div(Ch_ext, Dh, zh) + ze * flux_div(Ce_ext, De, ze)

    # 2. Сохранение кислородных вакансий
    res[:, 1] = flux_div(Cv_ext, Dv, zv)

    # 3. Сохранение протонных дефектов
    res[:, 2] = flux_div(Cp_ext, Dp, zp)

    # 4. Регуляризованное уравнение Пуассона
    laplacian_phi = (Phi_ext[2:] - 2.0 * Phi_ext[1:-1] + Phi_ext[:-2]) / (dx * dx)
    charge_density = zh * Ch + ze * Ce + zv * Cv + zp * Cp - Cd
    scale_poisson = 1.0 / max(Dh, De, 1.0)
    res[:, 3] = (eps_term * laplacian_phi + charge_density) * Ch * scale_poisson

    return res.ravel()


@st.cache_data(show_spinner=False)
def solve_transport_steady(
    icu, Temp, Thick, RpH, RpH2O, Phh2o, Phh2, Psh2o, Pso2, Cd, P_tuple,
    num_grid=200, n_continuation_steps=5
):
    """
    Решает стационарную 1D задачу электрохимического переноса в электролите.
    Кэшируется средствами Streamlit для мгновенного отклика интерфейса.
    """
    P_vec = np.array(P_tuple, dtype=float)
    TK = Temp + 273.15
    Dpd, Dvd, Dhd, Ded = P_vec[0], P_vec[1], P_vec[2], P_vec[3]
    Kph2o, Kpo2, Kph2, Kpeh = P_vec[4], P_vec[5], P_vec[6], P_vec[7]

    dx = 1.0 / num_grid
    epsilon0 = 100.0
    epsilonr = 100.0 * VM_4411 * R_GAS * TK / (F_CONST**2 * Thick**2)

    params = {
        'NumGrid': num_grid, 'dx': dx, 'Cd': Cd,
        'Dh': Dhd / Dpd, 'De': Ded / Dpd, 'Dv': Dvd / Dpd, 'Dp': 1.0,
        'Kph2o': Kph2o, 'Kpo2': Kpo2, 'Kph2': Kph2, 'Kpeh': Kpeh,
        'epsilon0': epsilon0, 'epsilonr': epsilonr
    }

    # Шаг 1: Начальное приближение при разомкнутой цепи (OCV, i = 0)
    bc_H_ocv = compute_boundary_concentrations('hydrogen', 0.0, Temp, RpH, Phh2, Phh2o, 0.0, Cd, params)
    bc_O_ocv = compute_boundary_concentrations('oxygen', 0.0, Temp, RpH2O, 0.0, Psh2o, Pso2, Cd, params)

    s = np.linspace(0.0, 1.0, num_grid)
    y0 = np.zeros((num_grid, 4))
    y0[:, 0] = np.sqrt(bc_H_ocv['Ch']) + s * (np.sqrt(bc_O_ocv['Ch']) - np.sqrt(bc_H_ocv['Ch']))
    y0[:, 1] = np.sqrt(bc_H_ocv['Cv']) + s * (np.sqrt(bc_O_ocv['Cv']) - np.sqrt(bc_H_ocv['Cv']))
    y0[:, 2] = np.sqrt(bc_H_ocv['Cp']) + s * (np.sqrt(bc_O_ocv['Cp']) - np.sqrt(bc_H_ocv['Cp']))
    y0[:, 3] = s * np.log(bc_O_ocv['Ce'] / bc_H_ocv['Ce'])

    sparsity = build_jacobian_sparsity(num_grid)

    sol_ocv = least_squares(
        steady_residual, y0.ravel(), args=(0.0, bc_H_ocv, bc_O_ocv, params),
        method='trf', jac_sparsity=sparsity, ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000
    )
    if np.max(np.abs(sol_ocv.fun)) > 1e-5:
        raise RuntimeError(f"Решатель не сошелся при OCV (max_res={np.max(np.abs(sol_ocv.fun)):.2e}): {sol_ocv.message}")

    y_curr = sol_ocv.x.reshape((num_grid, 4))
    y_curr[:, :3] = np.abs(y_curr[:, :3])

    # Профили при OCV до нагружения током
    Cv_ocv = y_curr[:, 1]**2
    Cp_ocv = y_curr[:, 2]**2
    Co_ocv = 3.0 - Cv_ocv - Cp_ocv
    PO2_ocv = (Co_ocv * (y_curr[:, 0]**4) / (Cv_ocv * Kpo2))**2

    # Шаг 2: Параметрическое продолжение по току от 0 до icu
    if abs(icu) > 1e-12:
        currents_ramp = np.linspace(0.0, icu, n_continuation_steps + 1)[1:]
    else:
        currents_ramp = []

    bc_H, bc_O = bc_H_ocv, bc_O_ocv
    sol_final = sol_ocv

    for ic_step in currents_ramp:
        icg_step = ic_step * Thick * VM_4411 / (Dpd * F_CONST) * 1e4
        bc_H = compute_boundary_concentrations('hydrogen', ic_step, Temp, RpH, Phh2, Phh2o, 0.0, Cd, params)
        bc_O = compute_boundary_concentrations('oxygen', ic_step, Temp, RpH2O, 0.0, Psh2o, Pso2, Cd, params)

        sol_final = least_squares(
            steady_residual, y_curr.ravel(), args=(icg_step, bc_H, bc_O, params),
            method='trf', jac_sparsity=sparsity, ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000
        )
        if np.max(np.abs(sol_final.fun)) > 1e-5:
            raise RuntimeError(
                f"Решатель не сошелся при i = {ic_step:.3f} A/cm2 (max_res={np.max(np.abs(sol_final.fun)):.2e})"
            )

        y_curr = sol_final.x.reshape((num_grid, 4))
        y_curr[:, :3] = np.abs(y_curr[:, :3])

    max_residual = float(np.max(np.abs(sol_final.fun)))

    # Шаг 3: Сборка полных профилей (длина N + 2)
    ic_final_dimless = icu * Thick * VM_4411 / (Dpd * F_CONST) * 1e4
    PhiH = 0.0
    PhiOH = compute_right_boundary_potential(y_curr[-1], y_curr[-2], ic_final_dimless, bc_O, params)

    Ch_dimless = np.concatenate(([bc_H['Ch']], y_curr[:, 0]**2, [bc_O['Ch']]))
    Ce_dimless = np.concatenate(([bc_H['Ce']], Kpeh / (y_curr[:, 0]**2), [bc_O['Ce']]))
    Cv_dimless = np.concatenate(([bc_H['Cv']], y_curr[:, 1]**2, [bc_O['Cv']]))
    Cp_dimless = np.concatenate(([bc_H['Cp']], y_curr[:, 2]**2, [bc_O['Cp']]))
    Phi_dimless = np.concatenate(([PhiH], y_curr[:, 3], [PhiOH]))

    Co_dimless = 3.0 - Cv_dimless - Cp_dimless
    PO2S = (Co_dimless * (Ch_dimless**2) / (Cv_dimless * Kpo2))**2
    PH2S = (Cp_dimless * Ce_dimless / (Co_dimless * Kph2))**2
    PH2OS = (Cp_dimless**2) / (Co_dimless * Cv_dimless * Kph2o)

    # Шаг 4: Консервативный расчет парциальных токов на гранях ячеек (N + 1 грань)
    step_sizes = np.full(num_grid + 1, dx)
    step_sizes[0] = 0.5 * dx
    step_sizes[-1] = 0.5 * dx

    def edge_current(C_ext, D_val, z_val):
        dC = C_ext[1:] - C_ext[:-1]
        dPhi = Phi_dimless[1:] - Phi_dimless[:-1]
        C_mid = 0.5 * (C_ext[1:] + C_ext[:-1])
        j_dimless = -z_val * D_val * (dC / step_sizes) - (z_val**2) * D_val * C_mid * (dPhi / step_sizes)
        return j_dimless * Dpd * (1.0 / VM_4411) * F_CONST / Thick * 1e-4

    ih = edge_current(Ch_dimless, params['Dh'], 1.0)
    ie = edge_current(Ce_dimless, params['De'], -1.0)
    iv = edge_current(Cv_dimless, params['Dv'], 2.0)
    ip = edge_current(Cp_dimless, params['Dp'], 1.0)

    i_total = ih + ie + iv + ip
    if abs(icu) > 1e-12:
        current_conservation_err = float(np.max(np.abs(i_total - icu)) / abs(icu))
        re = ie / i_total
        rh = ih / i_total
        rv = iv / i_total
        rp = ip / i_total
        FE_mean = float(np.mean(rp[1:-1]))
        FE_std = float(np.std(rp[1:-1]))
    else:
        current_conservation_err = float(np.max(np.abs(i_total)))
        re = np.zeros_like(i_total)
        rh = np.zeros_like(i_total)
        rv = np.zeros_like(i_total)
        rp = np.zeros_like(i_total)
        FE_mean = 0.0
        FE_std = 0.0

    ChS = Ch_dimless / VM_4411
    CeS = Ce_dimless / VM_4411
    CvS = Cv_dimless / VM_4411
    CpS = Cp_dimless / VM_4411
    CoS = Co_dimless / VM_4411
    PhiS = Phi_dimless * (R_GAS * TK / F_CONST)

    V_cell = float((R_GAS * TK / F_CONST) * (np.log(CeS[0]) - np.log(CeS[-1])) + (PhiS[-1] - PhiS[0]))

    x_nodes = np.concatenate(([0.0], (np.arange(num_grid) + 0.5) * dx, [1.0]))
    x_edges = np.linspace(0.0, 1.0, num_grid + 1)

    return {
        'x_nodes': x_nodes, 'x_edges': x_edges,
        'Ce': CeS, 'Ch': ChS, 'Cp': CpS, 'Cv': CvS, 'Co': CoS,
        'Phi': PhiS, 'V': V_cell,
        'pO2': PO2S, 'pH2': PH2S, 'pH2O': PH2OS,
        'Co_OCV': Co_ocv / VM_4411, 'pO2_OCV': PO2_ocv,
        'electron_current': re, 'hole_current': rh,
        'proton_current': rp, 'vacancy_current': rv,
        'ie_abs': ie, 'ih_abs': ih, 'iv_abs': iv, 'ip_abs': ip,
        'i_total': i_total,
        'FE_mean': FE_mean, 'FE_std': FE_std,
        'current_conservation_err': current_conservation_err,
        'max_residual': max_residual
    }


# ==============================================================================
# 3. ЗАГРУЗКА ПАРАМЕТРОВ ИЗ РЕПОЗИТОРИЯ
# ==============================================================================

def load_mat_parameters(uploaded_file=None):
    """Ищет example_parameters_4411.mat в стандартных путях GitHub-репозитория или берет загруженный."""
    if uploaded_file is not None:
        mat_data = sio.loadmat(uploaded_file)
        return mat_data['P'][:, 0]

    candidate_paths = [
        'example_parameters_4411.mat',
        os.path.join('code', 'example_parameters_4411.mat'),
        os.path.join(os.path.dirname(__file__), 'example_parameters_4411.mat'),
        os.path.join(os.path.dirname(__file__), 'code', 'example_parameters_4411.mat'),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            mat_data = sio.loadmat(path)
            return mat_data['P'][:, 0]
    return None


def load_summary_data():
    """Ищет figure_summary_4411.mat в стандартных путях GitHub-репозитория."""
    candidate_paths = [
        'figure_summary_4411.mat',
        os.path.join('figure_data', 'figure_summary_4411.mat'),
        os.path.join(os.path.dirname(__file__), 'figure_summary_4411.mat'),
        os.path.join(os.path.dirname(__file__), 'figure_data', 'figure_summary_4411.mat'),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return sio.loadmat(path)
    return None


# ==============================================================================
# 4. ВИЗУАЛИЗАЦИЯ (6 ПРОФИЛЕЙ + СВОДНЫЕ BOXPLOTS АНСАМБЛЯ)
# ==============================================================================

def make_profiles_figure(results):
    styles = ['-', '--', ':']
    names = [f"{int(round(r['fraction']*100))}% H₂O" for r in results]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    # A: Электроны и дырки
    ax = axes[0, 0]
    for r, st, name in zip(results, styles, names):
        ax.semilogy(r['x_nodes'], r['Ce'], color='#1f77b4', linestyle=st, lw=1.8, label=f'e⁻ ({name})')
        ax.semilogy(r['x_nodes'], r['Ch'], color='#d62728', linestyle=st, lw=1.8, label=f'h• ({name})')
    ax.set_xlabel('Нормированная координата x / L')
    ax.set_ylabel('Концентрация C_e, C_h (моль/м³)')
    ax.set_title('A. Электронные носители заряда')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(ncol=2, fontsize=8.5)

    # B: Протоны и вакансии
    ax = axes[0, 1]
    for r, st, name in zip(results, styles, names):
        ax.plot(r['x_nodes'], r['Cp'], color='#1f77b4', linestyle=st, lw=1.8, label=f'OH• ({name})')
        ax.plot(r['x_nodes'], r['Cv'], color='#d62728', linestyle=st, lw=1.8, label=f'v•• ({name})')
    ax.set_xlabel('Нормированная координата x / L')
    ax.set_ylabel('Концентрация C_p, C_v (моль/м³)')
    ax.set_title('B. Ионные дефекты')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8.5)

    # C: Электронные парциальные токи (доли)
    ax = axes[1, 0]
    for r, st, name in zip(results, styles, names):
        ax.plot(r['x_edges'], r['electron_current'], color='#1f77b4', linestyle=st, lw=1.8, label=f't_e ({name})')
        ax.plot(r['x_edges'], r['hole_current'], color='#d62728', linestyle=st, lw=1.8, label=f't_h ({name})')
    ax.set_xlabel('Нормированная координата x / L')
    ax.set_ylabel('Число переноса t_e, t_h')
    ax.set_title('C. Доли электронных токов (утечка)')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8.5)

    # D: Ионные парциальные токи (доли)
    ax = axes[1, 1]
    for r, st, name in zip(results, styles, names):
        ax.plot(r['x_edges'], r['proton_current'], color='#1f77b4', linestyle=st, lw=1.8, label=f't_p = FE ({name})')
        ax.plot(r['x_edges'], r['vacancy_current'], color='#d62728', linestyle=st, lw=1.8, label=f't_v ({name})')
    ax.set_xlabel('Нормированная координата x / L')
    ax.set_ylabel('Число переноса t_p, t_v')
    ax.set_title('D. Доли ионных токов (Фарадеевская эффективность)')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8.5)

    # E: Электростатический потенциал
    ax = axes[2, 0]
    for r, st, name in zip(results, styles, names):
        ax.plot(r['x_nodes'], r['Phi'], color='#0b2545', linestyle=st, lw=1.8, label=name)
    ax.set_xlabel('Нормированная координата x / L')
    ax.set_ylabel('Электростатический потенциал Φ (В)')
    ax.set_title('E. Профиль потенциала (включая границу x = L)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # F: Внутренние парциальные давления газов
    ax = axes[2, 1]
    ax_right = ax.twinx()
    for r, st, name in zip(results, styles, names):
        ax.semilogy(r['x_nodes'], r['pO2'], color='#2ca02c', linestyle=st, lw=1.8, label=f'pO2 ({name})')
        ax.semilogy(r['x_nodes'], r['pH2'], color='#1f77b4', linestyle=st, lw=1.8, label=f'pH2 ({name})')
        ax_right.plot(r['x_nodes'], r['pH2O'], color='#d62728', linestyle=st, lw=1.8, label=f'pH2O ({name})')
    ax.set_xlabel('Нормированная координата x / L')
    ax.set_ylabel('pO2, pH2 (атм)', color='#1f77b4')
    ax.set_ylim([1e-30, 1e10])
    ax_right.set_ylabel('pH2O (атм)', color='#d62728')
    ax_right.set_ylim([0, 4])
    ax.set_title('F. Внутренние парциальные давления газов')
    ax.grid(True, which='both', alpha=0.3)

    return fig


def make_summary_figure(S):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    fe_groups = [S['FE_50'] * 100, S['FE_70'] * 100, S['FE_90'] * 100]
    fe_labels = ['50%', '70%', '90%']
    fe_colors = ['#1420EE', '#3879FF', '#85CBF5']
    currents_1 = [-0.5, -1.0, -1.5, -2.0]
    offsets_1 = [-0.21, 0.0, 0.21]

    for grp, lbl, col, off in zip(fe_groups, fe_labels, fe_colors, offsets_1):
        positions = np.arange(len(currents_1)) + off
        bp = ax1.boxplot(
            [grp[j, np.isfinite(grp[j, :])] for j in range(len(currents_1))],
            positions=positions, widths=0.16, patch_artist=True,
            boxprops=dict(facecolor=col, edgecolor='black'),
            medianprops=dict(color='black', lw=1.2),
            flierprops=dict(marker='+', markeredgecolor='red', markersize=5)
        )
        bp['boxes'][0].set_label(lbl)

    ax1.set_xticks(range(len(currents_1)))
    ax1.set_xticklabels([f'{c:.1f}' for c in currents_1])
    ax1.set_ylim([50, 100])
    ax1.set_xlabel('Плотность тока (А/см²)')
    ax1.set_ylabel('Фарадеевская эффективность FE (%)')
    ax1.set_title('FE при разной доле пара (50%, 70%, 90%)')
    ax1.legend(loc='lower left', title='Доля H₂O')
    ax1.grid(True, axis='y', alpha=0.3)

    rp_groups = [S['RP_70_0']*100, S['RP_70_005']*100, S['RP_70_01']*100, S['RP_70_02']*100]
    rp_labels = ['Rp = 0', 'Rp = 0.05', 'Rp = 0.1', 'Rp = 0.2']
    rp_colors = ['#160FED', '#3757F0', '#5DACEF', '#9BFBFA']
    currents_2 = [-1.0, -1.5, -2.0]
    offsets_2 = [-0.315, -0.105, 0.105, 0.315]

    for grp, lbl, col, off in zip(rp_groups, rp_labels, rp_colors, offsets_2):
        positions = np.arange(len(currents_2)) + off
        bp = ax2.boxplot(
            [grp[j, np.isfinite(grp[j, :])] for j in range(len(currents_2))],
            positions=positions, widths=0.16, patch_artist=True,
            boxprops=dict(facecolor=col, edgecolor='black'),
            medianprops=dict(color='black', lw=1.2),
            flierprops=dict(marker='+', markeredgecolor='red', markersize=5)
        )
        bp['boxes'][0].set_label(lbl)

    ax2.set_xticks(range(len(currents_2)))
    ax2.set_xticklabels([f'{c:.1f}' for c in currents_2])
    ax2.set_ylim([40, 100])
    ax2.set_xlabel('Плотность тока (А/см²)')
    ax2.set_ylabel('Фарадеевская эффективность FE (%)')
    ax2.set_title('Чувствительность FE к поляризационному сопротивлению Rp (70% H₂O)')
    ax2.legend(loc='lower left', title='Сопротивление Rp')
    ax2.grid(True, axis='y', alpha=0.3)

    return fig


# ==============================================================================
# 5. ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС STREAMLIT
# ==============================================================================

st.title("Моделирование электрохимического переноса в BZCYYb4411")
st.markdown(
    "Одномерная стационарная модель Нернста–Планка–Пуассона для протонного керамического "
    "электролизера ($\mathrm{BaZr_{0.4}Ce_{0.4}Y_{0.1}Yb_{0.1}O_{3-\delta}}$) с исправленной "
    "граничной индексацией и проверкой сохранения полного тока."
)

# Боковая панель параметров
st.sidebar.header("Параметры эксперимента")

uploaded_param = st.sidebar.file_uploader(
    "Файл параметров (.mat)", type=["mat"],
    help="Необязательно, если example_parameters_4411.mat лежит в папке репозитория или code/"
)
P_vec = load_mat_parameters(uploaded_param)

if P_vec is None:
    st.error(
        "Файл `example_parameters_4411.mat` не найден в корне репозитория или папке `code/`. "
        "Загрузите его через панель слева или добавьте в GitHub-репозиторий."
    )
    st.stop()

icu = st.sidebar.slider("Плотность тока i (А/см²)", min_value=-2.5, max_value=0.0, value=-1.0, step=0.1)
temp_c = st.sidebar.slider("Температура T (°C)", min_value=450, max_value=750, value=600, step=10)
thick_um = st.sidebar.number_input("Толщина электролита L (мкм)", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
thick_m = thick_um * 1e-6

st.sidebar.subheader("Электроды и газовая среда")
rp_h = st.sidebar.number_input("Rp водородного электрода (Ом·см²)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
rp_o = st.sidebar.number_input("Rp кислородного электрода (Ом·см²)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
ph_h2 = st.sidebar.slider("Доля H₂ (водородный электрод)", min_value=0.10, max_value=0.99, value=0.99, step=0.01)
ph_h2o = st.sidebar.number_input("Доля H₂O (водородный электрод)", min_value=0.001, max_value=0.50, value=0.005, step=0.005, format="%.3f")

st.sidebar.subheader("Настройки сетки и материала")
cd_val = st.sidebar.number_input("Концентрация акцептора Cd", min_value=0.05, max_value=0.30, value=0.19, step=0.01)
num_grid = st.sidebar.select_slider("Число узлов сетки", options=[100, 150, 200, 300], value=200)

# Показываем загруженный вектор P из .mat
with st.sidebar.expander("Константы материала из .mat (вектор P)"):
    param_names = ["Dp (м²/с)", "Dv (м²/с)", "Dh (м²/с)", "De (м²/с)", "Kp,H2O", "Kp,O2", "Kp,H2", "Keh"]
    df_p = pd.DataFrame({"Параметр": param_names, "Значение": [f"{v:.4e}" for v in P_vec]})
    st.dataframe(df_p, hide_index=True, use_container_width=True)

# Расчет трех режимов влажности (50%, 70%, 90% H2O)
fractions = [0.50, 0.70, 0.90]
oxygen_fractions = [0.2 * (1.0 - f) for f in fractions]
P_tuple = tuple(float(x) for x in P_vec)

results = []
try:
    with st.spinner("Выполняется расчет стационарного переноса..."):
        for frac, po2 in zip(fractions, oxygen_fractions):
            res = solve_transport_steady(
                icu=float(icu), Temp=float(temp_c), Thick=float(thick_m),
                RpH=float(rp_h), RpH2O=float(rp_o),
                Phh2o=float(ph_h2o), Phh2=float(ph_h2),
                Psh2o=float(frac), Pso2=float(po2),
                Cd=float(cd_val), P_tuple=P_tuple, num_grid=int(num_grid)
            )
            res['fraction'] = frac
            results.append(res)
except Exception as exc:
    st.error(f"Ошибка численного решателя: {exc}")
    st.stop()

# Вкладки интерфейса
tab1, tab2, tab3 = st.tabs([
    "Сводка и верификация модели",
    "Пространственные профили (6 графиков)",
    "Данные ансамбля статьи (Boxplots)"
])

with tab1:
    st.subheader("Ключевые показатели и диагностика численной схемы")
    cols = st.columns(3)
    for col, r in zip(cols, results):
        pct = int(round(r['fraction'] * 100))
        with col:
            st.metric(
                label=f"Режим {pct}% H₂O | Фарадеевская эффективность",
                value=f"{r['FE_mean']*100:.2f} %",
                delta=f"Напряжение V = {r['V']:.3f} В"
            )

    summary_rows = []
    for r in results:
        summary_rows.append({
            "Доля пара H₂O (%)": int(round(r['fraction'] * 100)),
            "FE среднее (%)": round(r['FE_mean'] * 100, 4),
            "Разброс FE std (%)": f"{r['FE_std'] * 100:.2e}",
            "Напряжение ячейки V (В)": round(r['V'], 4),
            "Ошибка сохранения тока ε_i": f"{r['current_conservation_err']:.2e}",
            "Макс. невязка решателя": f"{r['max_residual']:.2e}"
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    st.info(
        "**Проверка физической состоятельности:** Относительная ошибка сохранения полного тока "
        "$\varepsilon_i = \max |i_{\text{total}}(x) - i_{\text{target}}| / |i_{\text{target}}|$ и разброс FE "
        "по толщине вычисляются автоматически по всем внутренним граням сетки."
    )

    # Экспорт результатов в CSV
    export_dfs = []
    for r in results:
        pct = int(round(r['fraction'] * 100))
        df_case = pd.DataFrame({
            'H2O_pct': pct,
            'x_L': r['x_nodes'],
            'Ce_mol_m3': r['Ce'],
            'Ch_mol_m3': r['Ch'],
            'Cp_mol_m3': r['Cp'],
            'Cv_mol_m3': r['Cv'],
            'Co_mol_m3': r['Co'],
            'Phi_V': r['Phi'],
            'pO2_atm': r['pO2'],
            'pH2_atm': r['pH2'],
            'pH2O_atm': r['pH2O']
        })
        export_dfs.append(df_case)
    csv_buffer = pd.concat(export_dfs, ignore_index=True).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать рассчитанные профили (CSV)",
        data=csv_buffer,
        file_name=f"BZCYYb4411_T{temp_c}_i{abs(icu):.1f}.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Пространственные распределения носителей, токов, потенциала и давлений")
    fig_profiles = make_profiles_figure(results)
    st.pyplot(fig_profiles)

    img_buf = io.BytesIO()
    fig_profiles.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
    st.download_button(
        label="Скачать графики профилей (PNG)",
        data=img_buf.getvalue(),
        file_name=f"profiles_BZCYYb4411_T{temp_c}.png",
        mime="image/png"
    )

with tab3:
    st.subheader("Распределения FE по ансамблю параметров (figure_summary_4411.mat)")
    summary_mat = load_summary_data()
    if summary_mat is not None:
        fig_sum = make_summary_figure(summary_mat)
        st.pyplot(fig_sum)
    else:
        st.warning(
            "Файл `figure_summary_4411.mat` не найден в репозитории (ожидается в корне или в папке `figure_data/`)."
        )
