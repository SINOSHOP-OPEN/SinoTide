# -*- coding: utf-8 -*-
"""静力平衡与浮力补偿 —— 全部无量纲

记号约定（不使用任何工程单位）
------------------------------
rho_rel : 浮体平均相对密度 = rho_body / rho_water，落在 (0, 1] 表示可自浮
s       : 浸没体积占比，落在 [0, 1]
xi      : 外部水位相对变化 = Δh / h_ref（相对参考吃水归一）
lam     : 舱内液位相对变化 = Δl / h_ref（同口径归一）
kappa   : 浮力补偿率 = lam / xi

物理依据只有三条教科书关系：
  1. 阿基米德原理     F_b = rho_water * g * V_sub
  2. 静力平衡         F_b = W   ⇒   s_eq = rho_rel
  3. 水面线斜率       dV/dh = A_w（水线面面积），用于舱容换算
"""

from __future__ import annotations

import math


# ── 1. 静力平衡 ────────────────────────────────────────────────────────
def equilibrium_submerged_fraction(rho_rel: float) -> float:
    """由静力平衡直接解出浸没占比（解析解）。

    平衡时 F_b = W：
        rho_water * g * s * V = rho_body * g * V
      ⇒ s = rho_rel
    这里刻意保留解析解，说明结论可以手算复核。
    """
    if not 0.0 < rho_rel <= 1.0:
        raise ValueError("rho_rel 必须落在 (0, 1]；本函数只接受无量纲相对量")
    return rho_rel


def buoyancy_residual(rho_rel: float, s: float) -> float:
    """相对形式的静力平衡残差。

    返回 s - rho_rel：正值偏浮、负值偏沉、零为平衡。
    仅用于说明残差随浸没占比单调变化，不代表任何真实受力数值。
    """
    return s - rho_rel


def solve_equilibrium_bisection(rho_rel: float, tol: float = 1e-12) -> float:
    """用二分法独立复核解析解 —— 两条路径给同一个数，才算自洽。"""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if buoyancy_residual(rho_rel, mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def heave_natural_period_rel(s: float, waterplane_rel: float) -> float:
    """无量纲垂荡固有周期。

    单自由度垂荡：omega_n = sqrt(rho*g*A_w / (m + m_a))
    取 m = rho * s * V（排水质量）、忽略附连水（m_a = 0）后
        omega_n^2 = g * A_w / (s * V)
    再令 L_ref 为特征长度、V ~ L_ref^3、A_w ~ waterplane_rel * L_ref^2，
    得无量纲形式
        T* = 2*pi*sqrt(s / waterplane_rel)
    该式与尺度无关，只用来演示「频率由浸没深度与水线面共同决定」这一条逻辑。
    """
    if s <= 0.0 or waterplane_rel <= 0.0:
        raise ValueError("s 与 waterplane_rel 必须为正的无量纲量")
    return 2.0 * math.pi * math.sqrt(s / waterplane_rel)


# ── 2. 浮力补偿 ────────────────────────────────────────────────────────
def ballast_first_order(xi, tau_star: float, dt_star: float):
    """舱内液位对水位变化的一阶滞后响应（对线性输入精确的积分格式）。

        tau* * dlam/dt* + lam = xi(t*)

    在一个步长内把 xi 视为**线性变化**，则该方程有解析解：
        lam[i] = xi[i] - k*D + (lam[i-1] - xi[i-1] + k*D) * exp(-dt*/tau*)
    其中 D = (xi[i] - xi[i-1]) / dt*，k = tau*。

    相比「把 xi 当分段常数」的显式格式，本格式没有 dt*/tau* 的稳定性上限，
    且误差为 O(dt*^2)：步长减半时结果几乎不变（见 verify_tide.py 的收敛判据）。

    参数
    ----
    xi       : 外部水位相对变化序列（无量纲）
    tau_star : 响应时间常数，以「一个潮周期」为 1
    dt_star  : 相对步长

    说明：这里只出现时间常数与步长的比值，不含任何真实秒数。
    """
    if tau_star <= 0.0 or dt_star <= 0.0:
        raise ValueError("tau_star 与 dt_star 必须为正")
    n = len(xi)
    if n == 0:
        return []
    out = [0.0] * n
    out[0] = xi[0]
    e = math.exp(-dt_star / tau_star)
    k = tau_star
    for i in range(1, n):
        d = (xi[i] - xi[i - 1]) / dt_star
        out[i] = xi[i] - k * d + (out[i - 1] - xi[i - 1] + k * d) * e
    return out


def compensation_gain(xi, lam) -> float:
    """浮力补偿率的**回归**定义（推荐用这个）。

        kappa = Σ(xi*lam) / Σ(xi^2)

    这是一阶滞后系统在正弦输入下的最小二乘增益，等价于幅频特性 |H|：
        kappa = 1 / sqrt(1 + (omega*tau)^2)

    之所以不用逐点 lam/xi：水位过零点时 lam/xi 会趋于无穷或跳号，
    逐点比值在数学上是不适定的，用它做判据只会得到噪声。
    """
    num = sum(a * b for a, b in zip(xi, lam))
    den = sum(a * a for a in xi)
    return num / den if den > 0 else float("nan")


def compensation_ratio(xi, lam, floor: float = 0.5):
    """逐点补偿率 lam/xi，仅保留 |xi| >= floor 的样本（避开过零点）。

    floor 以「相对幅值」计，默认 0.5 表示只统计水位变化超过半幅的区段。
    逐点值用于观察相位滞后，**不用于判定**——判定请用 compensation_gain。
    """
    out = []
    for a, b in zip(xi, lam):
        if abs(a) >= floor:
            out.append(b / a)
    return out


def vertical_residual(xi, lam):
    """残余竖向位移（相对量）：zeta = xi - lam。

    zeta 恒为零即完美定深。该量是「补偿是否有效」的直接度量，
    比 kappa 更稳健（在转折点不出现 0/0）。
    """
    return [a - b for a, b in zip(xi, lam)]


def ballast_volume_fraction(xi_max: float, waterplane_rel: float) -> float:
    """补偿最大水位变化所需的舱容占总体积之比（无量纲）。

        dV / V = A_w * Δh / V = waterplane_rel * xi_max
    """
    return waterplane_rel * xi_max


def natural_flow_rate(zeta, waterplane_rel: float, dt_star: float):
    """零耗能通道的自然流量（相对量）。

    流量只由**舱内外水头差** zeta 驱动，无泵、无外力：
        Q* ~ A_w * zeta / dt*
    这里给出它，是为了让「补/放水本身即水流」这句话可被逐点核对，
    而不是一句口号。
    """
    return [waterplane_rel * z / dt_star for z in zeta]
