# -*- coding: utf-8 -*-
"""波能俘获 —— 归一化单位制下的教科书模型

单位制声明（极其重要）
----------------------
本模块**不使用任何 SI 数值**。取 rho = 1、g = 1、入射波高 H_s = 1、
特征长度 D = 1 作为归一化基准，于是：

    波能面密度      E*  = (1/8) * rho * g * H^2        = 1/8
    深水群速度      cg* = g * T / (4*pi)               = 1/(4*pi)
    单宽波能通量    Pw* = E* * cg*                     = 1/(32*pi)
    俘能器动质量    M*  = m + m_a                      = 1
    静水恢复刚度    K*  = omega_n^2 * M*               = 1
    辐射阻尼        B*  = 2 * zeta_b * omega_n * M*

换算到真实量纲只需把 rho、g、H_s、T、D 乘回去 —— 这些量随海域而变，
属于工程标定范畴，不在本仓库内给定。

物理依据
--------
1. 线性波理论：深水波能通量 Pw = (rho*g^2*H^2*T)/(32*pi)   （教科书式）
2. 单自由度垂荡：M*x'' + (B_rad + B_pto)*x' + K*x = F_exc
3. Falnes 最优 PTO 阻尼：B_opt = sqrt(B_rad^2 + (omega*M - K/omega)^2)
4. 俘获宽度比 CWR = P_abs / (Pw * D)，无量纲效率指标
"""

from __future__ import annotations

import math

# ── 归一化基准（全部取 1，输出因此是无量纲比值）──────────────────────
RHO_STAR = 1.0
G_STAR = 1.0
H_STAR = 1.0
D_STAR = 1.0


def wave_energy_density() -> float:
    """单位面积波能 E* = (1/8) * rho * g * H^2（归一化后为 1/8）。"""
    return 0.125 * RHO_STAR * G_STAR * H_STAR ** 2


def group_velocity_deep(t_star: float) -> float:
    """深水群速度 cg* = g*T/(4*pi)。浅水需另换色散关系，本骨架不覆盖。"""
    if t_star <= 0.0:
        raise ValueError("t_star 必须为正的无量纲周期")
    return G_STAR * t_star / (4.0 * math.pi)


def wave_power_flux(t_star: float) -> float:
    """单宽波能通量 Pw* = E* * cg*（单位：归一化功率 / 单位波峰线长）。"""
    return wave_energy_density() * group_velocity_deep(t_star)


def excitation_force_amplitude() -> float:
    """入射激励力幅值 F0*（归一化取 1，仅用于演示传递关系）。"""
    return 1.0


def heave_amplitude(omega_rel: float, zeta_rad: float, zeta_pto: float) -> float:
    """单自由度垂荡位移幅值 |X|*。

        M*x'' + (B_rad + B_pto)*x' + K*x = F0*sin(omega*t)

    以 omega_n = 1、M = 1、K = 1 归一后：
        |X|* = F0 / sqrt((1 - omega_rel^2)^2 + (2*(zeta_rad+zeta_pto)*omega_rel)^2)
    """
    if omega_rel <= 0.0:
        raise ValueError("omega_rel 必须为正")
    if zeta_rad < 0.0 or zeta_pto < 0.0:
        raise ValueError("阻尼比不可为负")
    den = math.hypot(1.0 - omega_rel ** 2,
                     2.0 * (zeta_rad + zeta_pto) * omega_rel)
    return excitation_force_amplitude() / den if den > 0.0 else float("inf")


def absorbed_power(omega_rel: float, zeta_rad: float, zeta_pto: float) -> float:
    """俘获功率 P_abs* = (1/2) * B_pto * omega^2 * |X|^2。

    其中 B_pto = 2 * zeta_pto * omega_n * M，归一后 B_pto = 2 * zeta_pto。
    """
    x = heave_amplitude(omega_rel, zeta_rad, zeta_pto)
    b_pto = 2.0 * zeta_pto * 1.0 * 1.0
    return 0.5 * b_pto * omega_rel ** 2 * x ** 2


def capture_width_ratio(omega_rel: float, zeta_rad: float,
                        zeta_pto: float, t_star: float) -> float:
    """俘获宽度比 CWR = P_abs / (Pw * D)。无量纲效率指标。

    点吸收式俘能器在二维波场中 CWR 可以大于 1，这不是效率超过 100%，
    而是「捕获宽度大于自身宽度」——这正是点吸收路线的理论优势所在。
    """
    pw = wave_power_flux(t_star) * D_STAR
    return absorbed_power(omega_rel, zeta_rad, zeta_pto) / pw if pw > 0 else float("inf")


def optimal_pto_damping(omega_rel: float, zeta_rad: float) -> float:
    """Falnes 最优 PTO 阻尼比（解析式）。

    最优条件要求 PTO 提供的阻尼与「辐射阻尼 + 失谐电抗」的合模相等：
        B_opt = sqrt(B_rad^2 + (omega*M - K/omega)^2)
    除以 2*omega_n*M 后写成阻尼比形式：
        zeta_opt = sqrt(zeta_rad^2 + (omega_rel/2 - 1/(2*omega_rel))^2)
    """
    if omega_rel <= 0.0:
        raise ValueError("omega_rel 必须为正")
    react = 0.5 * (omega_rel - 1.0 / omega_rel)
    return math.sqrt(zeta_rad ** 2 + react ** 2)


def optimal_pto_damping_stroke_limited(omega_rel: float, zeta_rad: float,
                                       x_max: float, zeta_hi: float = 50.0) -> float:
    """行程受限时的最优 PTO 阻尼比。

    无约束最优解会让位移幅值趋于无穷（阻尼趋于零）。真实装置有行程上限，
    因此工程上的最优是「在 |X|* <= x_max 的边界上取最小的阻尼」，
    即把阻尼刚刚调到不撞限位为止。用二分法求该边界点。
    """
    if x_max <= 0.0:
        raise ValueError("x_max 必须为正")
    lo = optimal_pto_damping(omega_rel, zeta_rad)
    if heave_amplitude(omega_rel, zeta_rad, lo) <= x_max:
        return lo  # 行程不起控制作用
    hi = max(lo * 2.0, 1e-6)
    while heave_amplitude(omega_rel, zeta_rad, hi) > x_max and hi < zeta_hi:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if heave_amplitude(omega_rel, zeta_rad, mid) > x_max:
            lo = mid
        else:
            hi = mid
    return hi


def power_sweep(omega_rel_list, zeta_rad: float, zeta_pto: float, t_star: float):
    """在一串频率比上扫描俘获功率与 CWR，用于画频响曲线。"""
    rows = []
    for w in omega_rel_list:
        rows.append((w,
                     heave_amplitude(w, zeta_rad, zeta_pto),
                     absorbed_power(w, zeta_rad, zeta_pto),
                     capture_width_ratio(w, zeta_rad, zeta_pto, t_star)))
    return rows


def displacement_transmissibility(r: float, zeta: float) -> float:
    """基底激励位移传递率（隔振教科书公式）。

        T = sqrt((1 + (2*zeta*r)^2) / ((1 - r^2)^2 + (2*zeta*r)^2))
    r > sqrt(2) 时进入隔振区（T < 1），这是「频率隔离」主张的数学依据。
    """
    if r <= 0.0:
        raise ValueError("r 必须为正")
    num = 1.0 + (2.0 * zeta * r) ** 2
    den = (1.0 - r * r) ** 2 + (2.0 * zeta * r) ** 2
    return math.sqrt(num / den) if den > 0 else float("inf")


def isolation_threshold() -> float:
    """隔振区起点 r = sqrt(2)，与阻尼无关。"""
    return math.sqrt(2.0)
