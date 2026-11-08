# -*- coding: utf-8 -*-
"""虚拟测试序列 —— 固定种子，可复现，非任何真实海域实测

★ 数据性质声明
--------------
本模块生成的全部序列都是**人为构造的虚拟数据**，用途只有一个：
让 `verify_tide.py` 里的判据有输入可跑、可被第三方重复。

它 **不是** 任何真实海域的潮位记录、波浪谱或实测标定结果；
不得用于工程计算、不得作为设计输入、不得引用为实测证据。

一切都以「相对量」表达：
    xi     : 外部水位相对变化，幅值归一为 1
    t_star : 相对时间，一个潮周期记为 1.0
因此本文件不含任何长度、时间、质量的真实数值。
"""

from __future__ import annotations

import math

SEED = 20261108          # 构造编号（本序列为完全确定性合成，不依赖随机数）
DECLARATION = (
    "虚拟测试数据（virtual test data）：由固定随机种子合成，"
    "非任何真实海域实测记录，不得用于工程计算或作为设计输入。"
)


def tide_series(n: int = 4001, cycles: float = 2.0, asym: float = 0.05,
                ripple: float = 0.02) -> list:
    """合成一条无量纲「不对称潮位」序列。

    参数
    ----
    n      : 采样点数
    cycles : 覆盖多少个潮周期（无量纲）
    asym   : 二次谐波占比，用来制造「涨潮快、退潮慢」的不对称性
    ripple : 叠加的高次谐波幅值（模拟量测抖动，见下方说明）

    返回
    ----
    (t_star[], xi[]) 两条等长序列，幅值已归一为 max|xi| = 1
    """
    if n < 3:
        raise ValueError("n 至少为 3")
    w = 2.0 * math.pi * cycles

    t, xi = [], []
    for i in range(n):
        u = i / (n - 1)
        base = math.sin(w * u) + asym * math.sin(2.0 * w * u + 0.6)
        # 抖动刻意用**确定性高次谐波**，而不是白噪声：
        # 白噪声在相邻样本间跳变，任何对时间求导的格式都会把它放大成毛刺，
        # 图上看到的就不是物理残差而是数值噪声。高次谐波既保留了抖动形态，
        # 又能被采样加密逐步解析，收敛性判据才有意义。
        jitter = ripple * math.sin(2.0 * math.pi * 13.0 * u + 0.9) \
            + 0.5 * ripple * math.sin(2.0 * math.pi * 29.0 * u + 2.3)
        t.append(u * cycles)          # t* 以潮周期为单位：1.0 = 一个周期
        xi.append(base + jitter)

    peak = max(abs(v) for v in xi)
    return t, [v / peak for v in xi]


def reference_case() -> dict:
    """基准算例的全部无量纲输入 —— 与任何真实工程参数无关。

    这些数字是**为了方便复核而选的整数量级**，不是标定结果：
        rho_rel     0.60   浮体平均相对密度（可自浮，等体积排水 60%）
        waterplane  0.55   水线面面积 / 特征面积
        tau_star    0.002  舱内响应时间常数（以潮周期为单位，即千分之二个周期）
        zeta_rad    0.20   辐射阻尼比
        t_star      1.00   入射波相对周期
        x_max       0.80   允许的垂荡位移幅值（相对量）

    这些数字是**为了复核方便而取的名义值**，不是标定结果；换一组同样成立，
    判据考核的是关系是否自洽，不是某个数好不好看。
    """
    return {
        "rho_rel": 0.60,
        "waterplane_rel": 0.55,
        "tau_star": 0.002,
        "zeta_rad": 0.20,
        "t_star": 1.00,
        "x_max": 0.80,
        "omega_rel": 1.00,
    }
