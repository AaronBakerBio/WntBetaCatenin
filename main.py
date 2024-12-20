"""Final model

"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def k_mat(t):
    return 1 / (1 + np.exp(t - 10))

def k_zyg(t):
    return 1 / (1 + np.exp(-(t - 10)))

def full_model(y, t, ks, gammas, h):
    NO, S, F, CH, GSC, L, A, BC1, BM, WN, FF, M, CDX = y
    dNO = (ks[0] * k_mat(t) * NO**h + ks[1] * k_mat(t) * BC1**h) / (NO**h + BC1**h + L**h + 1) - gammas[0] * NO
    dS = (ks[2] * k_mat(t) * NO**h + ks[3] * k_mat(t) * BC1**h) / (NO**h + BC1**h + L**h + 1) - gammas[1] * S
    dF = (ks[4] * BC1**h + ks[5] * NO**h + ks[6] * GSC**h) / (BC1**h + NO**h + GSC**h + 1) - gammas[2] * F
    dCH = (ks[7] * BC1**h + ks[8] * NO**h + ks[9] * GSC**h) / (BC1**h + NO**h + GSC**h + 1) - gammas[3] * CH
    dGSC = (ks[10] * BC1**h + ks[11] * NO**h + ks[12] * GSC**h) / (BC1**h + NO**h + GSC**h + 1) - gammas[4] * GSC
    dL = (ks[13] * BC1**h + ks[14] * NO**h + ks[15] * GSC**h) / (BC1**h + NO**h + GSC**h + 1) - gammas[5] * L
    dA = (ks[16] * BC1**h) / (BC1**h + 1) - gammas[6] * A
    dBC1 = (ks[17] * WN**h) / (A**h + WN**h + 1) - gammas[7] * BC1
    dBM = (ks[18] * k_zyg(t)) / (1 + CH**h) - gammas[8] * BM
    dWN = (ks[19] * k_zyg(t)) / (F**h + 1) - gammas[9] * WN
    dFF = ks[20] * k_zyg(t) - gammas[10] * FF
    dM = (ks[21] * BC1**h + ks[22] * BM**h) / (BC1**h + BM**h + 1) - gammas[11] * M
    dCDX = (ks[23] * BC1**h + ks[24] * FF**h) / (BC1**h + FF**h + 1) - gammas[12] * CDX
    return [dNO, dS, dF, dCH, dGSC, dL, dA, dBC1, dBM, dWN, dFF, dM, dCDX]

def simple_model(y, t, ks, gammas, h):
    NS, FCGL, A, BC1, BM, WN, FF, M, CDX = y
    dNS = (ks[0] * k_mat(t) * NS**h + ks[1] * k_mat(t) * BC1**h) / (NS**h + BC1**h + FCGL**h + 1) - gammas[0] * NS
    dFCGL = (ks[4] * BC1**h + ks[5] * NS**h + ks[6] * FCGL**h) / (BC1**h + NS**h + FCGL**h + 1) - gammas[2] * FCGL
    dA = (ks[16] * BC1**h) / (BC1**h + 1) - gammas[6] * A
    dBC1 = (ks[17] * WN**h) / (A**h + WN**h + 1) - gammas[7] * BC1
    dBM = (ks[18] * k_zyg(t)) / (1 + FCGL**h) - gammas[8] * BM
    dWN = (ks[19] * k_zyg(t)) / (FCGL**h + 1) - gammas[9] * WN
    dFF = ks[20] * k_zyg(t) - gammas[10] * FF
    dM = (ks[21] * BC1**h + ks[22] * BM**h) / (BC1**h + BM**h + 1) - gammas[11] * M
    dCDX = (ks[23] * BC1**h + ks[24] * FF**h) / (BC1**h + FF**h + 1) - gammas[12] * CDX
    return [dNS, dFCGL, dA, dBC1, dBM, dWN, dFF, dM, dCDX]

def plot_full_model(t, solution, title, colors):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, k_mat(t), color=colors['maternal'], label='Maternal')
    ax.plot(t, k_zyg(t), color=colors['zygotic'], label='Zygotic')
    ax.plot(t, solution[:, 0], color=colors['nodal'], label='Nodal')
    ax.plot(t, solution[:, 1], color=colors['siamois'], label='Siamois')
    ax.plot(t, solution[:, 2], color=colors['frzb'], label='FrzB')
    ax.plot(t, solution[:, 3], color=colors['chordin'], label='Chordin')
    ax.plot(t, solution[:, 4], color=colors['goosecoid'], label='Goosecoid')
    ax.plot(t, solution[:, 5], color=colors['lefty'], label='Lefty')
    ax.plot(t, solution[:, 6], color=colors['axin2'], label='Axin2')
    ax.plot(t, solution[:, 7], color=colors['bcatenin'], label='β-catenin')
    ax.plot(t, solution[:, 8], color=colors['bmp'], label='BMP')
    ax.plot(t, solution[:, 9], color=colors['wnt8a'], label='WNT8a')
    ax.plot(t, solution[:, 10], color=colors['fgf'], label='FGF')
    ax.plot(t, solution[:, 11], color=colors['msx1'], label='Msx1')
    ax.plot(t, solution[:, 12], color=colors['cdx'], label='CDX')
    ax.set_xlabel('Time')
    ax.set_ylabel('Protein Concentration')
    # for consistency’s sake to compare with the paper
    ax.set_ylim(0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_simple_model(t, solution, title, colors):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, k_mat(t), color=colors['maternal'], label='Maternal')
    ax.plot(t, k_zyg(t), color=colors['zygotic'], label='Zygotic')
    ax.plot(t, solution[:, 0], color=colors['nodal_siamois'], label='Nodal/Siamois')
    ax.plot(t, solution[:, 1], color=colors['frzb'], label='FrzB/Chordin/Goosecoid/Lefty')
    ax.plot(t, solution[:, 2], color=colors['axin2'], label='Axin2')
    ax.plot(t, solution[:, 3], color=colors['bcatenin'], label='β-catenin')
    ax.plot(t, solution[:, 4], color=colors['bmp'], label='BMP')
    ax.plot(t, solution[:, 5], color=colors['wnt8a'], label='WNT8a')
    ax.plot(t, solution[:, 6], color=colors['fgf'], label='FGF')
    ax.plot(t, solution[:, 7], color=colors['msx1'], label='Msx1')
    ax.plot(t, solution[:, 8], color=colors['cdx'], label='CDX')
    ax.set_xlabel('Time')
    ax.set_ylabel('Protein Concentration')
    ax.set_title(title)
    # for consistency’s sake to compare with the paper
    ax.set_ylim(0)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def main():
    full_colors = {
        'maternal': '#fafa43', 'zygotic': '#32f92c',
        'nodal': '#272727', 'siamois': '#f78c8f',
        'frzb': '#0708ba', 'chordin': '#e4edfc',
        'goosecoid': '#edfeed', 'lefty': '#e6e6e6',
        'axin2': '#7ff3ed', 'bcatenin': '#d521d8',
        'bmp': '#ff5053', 'wnt8a': '#8f6e52',
        'fgf': '#ef8b1d', 'msx1': '#838383',
        'cdx': '#6a216a'
    }
    simple_colors = {
        'maternal': '#fafa43', 'zygotic': '#32f92c',
        'nodal_siamois': '#272727', 'frzb': '#0708ba',
        'axin2': '#7ff3ed', 'bcatenin': '#d521d8',
        'bmp': '#ff5053', 'wnt8a': '#8f6e52',
        'fgf': '#ef8b1d', 'msx1': '#838383',
        'cdx': '#6a216a'
    }
    t = np.linspace(0, 20, 1000)
    h = 1.8
    ks = np.array([1.0, 0.1, 1.0, 0.1, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 0.1, 1.0])
    gammas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Full model simulations
    full_ventral = np.zeros(13)
    full_dorsal = np.zeros(13)
    full_dorsal[0] = 0.1
    full_dorsal[7] = 1.0
    full_vent_sol = odeint(full_model, full_ventral, t, args=(ks, gammas, h))
    full_dor_sol = odeint(full_model, full_dorsal, t, args=(ks, gammas, h))
    plot_full_model(t, full_vent_sol, 'Fig 2: Full Model - Ventral', full_colors)
    plot_full_model(t, full_dor_sol, 'Fig 3: Full Model - Dorsal', full_colors)

    # Simple model simulations
    simple_ventral = np.zeros(9)
    simple_dorsal = np.zeros(9)
    simple_dorsal[0] = 0.1
    simple_dorsal[3] = 1.0
    simple_vent_sol = odeint(simple_model, simple_ventral, t, args=(ks, gammas, h))
    simple_dor_sol = odeint(simple_model, simple_dorsal, t, args=(ks, gammas, h))
    plot_simple_model(t, simple_vent_sol, 'Fig 5: Simple Model - Ventral', simple_colors)
    plot_simple_model(t, simple_dor_sol, 'Fig 6: Simple Model - Dorsal', simple_colors)

    # Figure 7 - NS overexpression
    ns_init = np.zeros(9)
    ns_init[0] = 1.0  # Initial NS = 1
    sol_7 = odeint(simple_model, ns_init, t, args=(ks, gammas, h))
    plot_simple_model(t, sol_7, 'Figure 7: NS Over-expression', simple_colors)

    # Figure 8 - Class II genes knockdown
    ks_knockdown = ks.copy()
    ks_knockdown[4:7] = 0  # Set k5, k6, k7 = 0
    knockdown_init = np.zeros(9)
    knockdown_init[0] = 1.0  # Initial NS = 1
    knockdown_init[3] = 1.0  # Initial BC1 = 1
    sol_8 = odeint(simple_model, knockdown_init, t, args=(ks_knockdown, gammas, h))
    plot_simple_model(t, sol_8, 'Figure 8: Class II Genes Knock-down', simple_colors)

    # Figure 9 - FCGL overexpression
    fcgl_init = np.zeros(9)
    fcgl_init[1] = 1.0  # Initial FCGL = 1
    sol_9 = odeint(simple_model, fcgl_init, t, args=(ks, gammas, h))
    plot_simple_model(t, sol_9, 'Figure 9: FCGL Over-activation', simple_colors)

    # Figure 10 - BMP overexpression
    bmp_init = np.zeros(9)
    bmp_init[0] = 1.0  # Initial NS = 1
    bmp_init[3] = 1.0  # Initial BC1 = 1
    bmp_init[4] = 1.0  # Initial BMP = 1
    sol_10 = odeint(simple_model, bmp_init, t, args=(ks, gammas, h))
    plot_simple_model(t, sol_10, 'Figure 10: BMP Over-activation', simple_colors)

    # Figure 11 - FGF overexpression
    fgf_init = np.zeros(9)
    fgf_init[0] = 1.0  # Initial NS = 1
    fgf_init[3] = 1.0  # Initial BC1 = 1
    fgf_init[6] = 1.0  # Initial FF = 1
    sol_11 = odeint(simple_model, fgf_init, t, args=(ks, gammas, h))
    plot_simple_model(t, sol_11, 'Figure 11: FGF Over-activation', simple_colors)

    # custom perturbations
    # axin2
    axin_ventral = np.zeros(9)
    axin_dorsal = np.zeros(9)
    axin_dorsal[0] = 0.1
    axin_dorsal[3] = 1.0
    axin_ks = ks.copy()
    # Reduce Axin2 production rate to simulate repression
    axin_ks[16] = 0.01  # Reduce from original 0.1 to simulate repression
    axin_vent_sol = odeint(simple_model, axin_ventral, t, args=(axin_ks, gammas, h))
    axin_dor_sol = odeint(simple_model, axin_dorsal, t, args=(axin_ks, gammas, h))
    plot_simple_model(t, axin_dor_sol, 'Simple Model - Class3/Axin2 repression', simple_colors)
    plot_simple_model(t, axin_vent_sol, 'Simple Model - Class3/Axin2 repression', simple_colors)

    # Axin2 overexpression
    axin2_ventral = np.zeros(9)
    axin2_dorsal = np.zeros(9)
    axin2_dorsal[0] = 0.1
    axin2_dorsal[3] = 1.0
    # Add Axin2 overexpression in both conditions
    axin2_ventral[2] = 1.0  # Set initial Axin2 concentration to 1
    axin2_dorsal[2] = 1.0  # Set initial Axin2 concentration to 1

    axin2_vent_sol = odeint(simple_model, axin2_ventral, t, args=(ks, gammas, h))
    axin2_dor_sol = odeint(simple_model, axin2_dorsal, t, args=(ks, gammas, h))
    plot_simple_model(t, axin2_vent_sol, 'Simple Model - Axin2 Overexpression Ventral', simple_colors)
    plot_simple_model(t, axin2_dor_sol, 'Simple Model - Axin2 Overexpression Dorsal', simple_colors)
    # Combined BMP and WNT8a overexpression
    combo_ventral = np.zeros(9)
    combo_dorsal = np.zeros(9)
    combo_dorsal[0] = 0.1
    combo_dorsal[3] = 1.0
    # Set high initial BMP
    combo_ventral[4] = 1.0  # BMP
    combo_dorsal[4] = 1.0
    # Set high initial WNT8a
    combo_ventral[5] = 1.0  # WNT8a
    combo_dorsal[5] = 1.0

    combo_vent_sol = odeint(simple_model, combo_ventral, t, args=(ks, gammas, h))
    combo_dor_sol = odeint(simple_model, combo_dorsal, t, args=(ks, gammas, h))
    plot_simple_model(t, combo_vent_sol, 'Simple Model - BMP + WNT8a Overexpression Ventral', simple_colors)
    plot_simple_model(t, combo_dor_sol, 'Simple Model - BMP + WNT8a Overexpression Dorsal', simple_colors)


    # Full model perturbations
    # Axin2 repression
    axin_ks = ks.copy()
    axin_ks[16] = 0.01  # Reduce Axin2 production rate
    full_axin_ventral = np.zeros(13)
    full_axin_dorsal = np.zeros(13)
    full_axin_dorsal[0] = 0.1
    full_axin_dorsal[7] = 1.0
    full_axin_vent_sol = odeint(full_model, full_axin_ventral, t, args=(axin_ks, gammas, h))
    full_axin_dor_sol = odeint(full_model, full_axin_dorsal, t, args=(axin_ks, gammas, h))
    plot_full_model(t, full_axin_vent_sol, 'Full Model - Axin2 Repression Ventral', full_colors)
    plot_full_model(t, full_axin_dor_sol, 'Full Model - Axin2 Repression Dorsal', full_colors)

    # Axin2 overexpression
    full_axin2_ventral = np.zeros(13)
    full_axin2_dorsal = np.zeros(13)
    full_axin2_dorsal[0] = 0.1
    full_axin2_dorsal[7] = 1.0
    full_axin2_ventral[6] = 1.0  # Set initial Axin2 concentration to 1
    full_axin2_dorsal[6] = 1.0  # Set initial Axin2 concentration to 1
    full_axin2_vent_sol = odeint(full_model, full_axin2_ventral, t, args=(ks, gammas, h))
    full_axin2_dor_sol = odeint(full_model, full_axin2_dorsal, t, args=(ks, gammas, h))
    plot_full_model(t, full_axin2_vent_sol, 'Full Model - Axin2 Overexpression Ventral', full_colors)
    plot_full_model(t, full_axin2_dor_sol, 'Full Model - Axin2 Overexpression Dorsal', full_colors)

    # Combined BMP and WNT8a overexpression
    full_combo_ventral = np.zeros(13)
    full_combo_dorsal = np.zeros(13)
    full_combo_dorsal[0] = 0.1
    full_combo_dorsal[7] = 1.0
    full_combo_ventral[8] = 1.0  # BMP
    full_combo_dorsal[8] = 1.0
    full_combo_ventral[9] = 1.0  # WNT8a
    full_combo_dorsal[9] = 1.0
    full_combo_vent_sol = odeint(full_model, full_combo_ventral, t, args=(ks, gammas, h))
    full_combo_dor_sol = odeint(full_model, full_combo_dorsal, t, args=(ks, gammas, h))
    plot_full_model(t, full_combo_vent_sol, 'Full Model - BMP + WNT8a Overexpression Ventral', full_colors)
    plot_full_model(t, full_combo_dor_sol, 'Full Model - BMP + WNT8a Overexpression Dorsal', full_colors)

if __name__ == "__main__":
    main()