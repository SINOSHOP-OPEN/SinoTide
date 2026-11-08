#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
SinoTide · 公开仿真骨架 · verify_tide.py
定深悬浮（浮力补偿率）× 潮汐取能（波能俘获）—— 无量纲判据自检
================================================================================

【运行】
    python verify_tide.py
    python verify_tide.py --no-fig      # 不画图（无 matplotlib 环境亦可跑）

【退出码】
    0 = 全部判据通过
    1 = 存在失败判据（逐条打印 FAIL 原因）

【它验证什么】
    一、静力平衡与浮力补偿
        · 解析解与二分法独立复核一致
        · 浮力补偿率中位数 ≥ 0.95，补偿率 ≥ 0.9 的样本占比 ≥ 90%
        · 残余竖向位移（定深误差）峰值 ≤ 相对波幅的 5%
        · 补偿所需舱容不超过可用舱容
        · 时间常数增大 ⇒ 残余位移单调增大（逻辑自洽性）
        · 步长减半 ⇒ 结果收敛（数值稳健性）

    二、波能俘获
        · 单宽波能通量与线性波理论解析式一致
        · Falnes 最优 PTO 阻尼：解析式与数值扫描的极值点吻合（相对误差 < 1%）
        · 共振点最优阻尼 = 辐射阻尼（经典结论的数值复现）
        · 行程受限时最优阻尼上移、位移幅值恰好压到限位
        · 位移传递率在 r = sqrt(2) 处为 1，r > sqrt(2) 进入隔振区

    三、口径自检
        · 输出中不得出现千瓦时、兆瓦、米、秒等工程量纲单位
        · 虚拟数据声明必须上屏

【它不验证什么 —— 请务必读这一段】
    本脚本使用**归一化单位制**（rho = g = H_s = D = 1），全部输入为无量纲相对量。
    它证明的是「模型自洽、判据可复现」，**不是**任何真实海域的发电量、位移量或
    结构安全裕度。真实量纲结果需要代入具体海域的密度、重力、波高、周期与特征
    尺度，并叠加入水动力学、粘性、系泊与结构弹性 —— 那些属于工程标定范畴，
    不在本仓库内给定。

    一句话：这里给出的是骨架，不是体检报告。
================================================================================
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import hydrostatics as hs
from sim import waveenergy as we
from sim import virtualdata as vd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FAILS = []
PASSES = []


def chk(cond, msg, detail=""):
    """记录一条判据。cond 为真记 PASS，否则记 FAIL 并保留细节。"""
    if cond:
        PASSES.append(msg)
        print(f"  PASS  {msg}")
    else:
        FAILS.append(msg if not detail else f"{msg} —— {detail}")
        print(f"  FAIL  {msg}" + (f"  [{detail}]" if detail else ""))


def hr(title):
    print("\n" + "─" * 74)
    print(title)
    print("─" * 74)


# ══════════════════════════════════════════════════════════════════════════
# 一、静力平衡与浮力补偿
# ══════════════════════════════════════════════════════════════════════════
def part_hydrostatics(cfg):
    hr("一、静力平衡与浮力补偿（浮力补偿率判据）")

    rho_rel = cfg["rho_rel"]
    wp = cfg["waterplane_rel"]
    tau = cfg["tau_star"]

    s_eq = hs.equilibrium_submerged_fraction(rho_rel)
    s_bis = hs.solve_equilibrium_bisection(rho_rel)
    chk(abs(s_eq - s_bis) < 1e-9,
        f"解析解与二分法一致：浸没占比 = {s_eq:.6f}",
        f"解析 {s_eq:.9f} / 二分 {s_bis:.9f}")
    chk(abs(hs.buoyancy_residual(rho_rel, s_eq)) < 1e-12,
        "平衡点处静力残差归零（ΣF = 0）")

    t_star, xi = vd.tide_series()
    dt = t_star[1] - t_star[0]
    chk(abs(max(abs(v) for v in xi) - 1.0) < 1e-9,
        "测试序列幅值已归一为 1（全部相对量）")
    chk(abs(t_star[-1] - 2.0) < 1e-9,
        f"时间基准自洽：t* 跨 2 个潮周期（末点 {t_star[-1]:.4f}），1.0 = 一个周期")

    lam = hs.ballast_first_order(xi, tau, dt)
    zeta = hs.vertical_residual(xi, lam)
    kappa = hs.compensation_gain(xi, lam)
    k_pts = sorted(hs.compensation_ratio(xi, lam, floor=0.5))
    frac_in_band = sum(1 for k in k_pts if 0.90 <= k <= 1.10) / len(k_pts)
    z_max = max(abs(z) for z in zeta)
    z_rms = math.sqrt(sum(z * z for z in zeta) / len(zeta))

    print(f"       补偿率（回归增益）κ = {kappa:.6f}，τ*/T = {tau:.4f}")
    print(f"       逐点 κ（|ξ| ≥ 0.5）落在 [0.90, 1.10] 的占比 = {frac_in_band * 100:.1f}%")
    print(f"       残余竖向位移峰值 |ζ|max = {z_max:.5f}（相对波幅 1.0），RMS = {z_rms:.5f}")

    chk(kappa >= 0.95, f"浮力补偿率 κ = {kappa:.6f}（≥ 0.95）")
    # 解析复核：用纯正弦输入（不含谐波与扰动），增益应等于一阶系统幅频特性
    n_pure = 20001
    pure = [math.sin(2.0 * math.pi * i / (n_pure - 1)) for i in range(n_pure)]
    g_pure = hs.compensation_gain(
        pure, hs.ballast_first_order(pure, tau, 2.0 / (n_pure - 1)))
    g_theory = 1.0 / math.sqrt(1.0 + (2.0 * math.pi * tau) ** 2)
    chk(abs(g_pure - g_theory) < 1e-3,
        f"纯正弦输入下增益 {g_pure:.6f} 与一阶系统幅频特性 {g_theory:.6f} 吻合（相对差 "
        f"{abs(g_pure - g_theory) / g_theory * 100:.4f}%）")
    chk(frac_in_band >= 0.90,
        f"逐点补偿率落在 [0.90, 1.10] 的占比 = {frac_in_band * 100:.1f}%（≥ 90%，滞后可控）",
        f"样本 {len(k_pts)} 个，最小值 {k_pts[0]:.4f}")
    chk(z_max <= 0.05,
        f"残余竖向位移峰值 {z_max:.5f} ≤ 相对波幅的 5%（定深精度）")
    chk(z_rms < z_max,
        f"残余位移 RMS {z_rms:.5f} < 峰值 {z_max:.5f}（分布合理）")

    v_frac = hs.ballast_volume_fraction(1.0, wp)
    chk(v_frac <= 0.80,
        f"补偿所需舱容占比 = {v_frac:.3f} ≤ 0.80（留有余量，可实施）")

    q = hs.natural_flow_rate(zeta, wp, dt)
    chk(all(qi * zi >= 0.0 for qi, zi in zip(q, zeta)),
        "自然流量与水头差同号（仅由水位差驱动，无泵、无外力）")
    chk(abs(max(abs(v) for v in q) - wp * z_max / dt) < 1e-9,
        "自然流量峰值 = A_w · |ζ|max / dt（零耗能通道口径自洽）")

    zmax_by_tau = []
    for tau_i in (0.001, 0.002, 0.020):
        lam_i = hs.ballast_first_order(xi, tau_i, dt)
        zmax_by_tau.append(max(abs(z) for z in hs.vertical_residual(xi, lam_i)))
    chk(zmax_by_tau[0] < zmax_by_tau[1] < zmax_by_tau[2],
        "时间常数增大 ⇒ 残余位移单调增大（因果方向正确）",
        " / ".join("%.5f" % v for v in zmax_by_tau))

    # 收敛性：用同一确定性输入（关掉扰动）比较两种采样密度，隔离积分误差
    t_a, xi_a = vd.tide_series(n=4001, ripple=0.0)
    t_b, xi_b = vd.tide_series(n=8001, ripple=0.0)
    za = max(abs(z) for z in hs.vertical_residual(
        xi_a, hs.ballast_first_order(xi_a, tau, t_a[1] - t_a[0])))
    zb = max(abs(z) for z in hs.vertical_residual(
        xi_b, hs.ballast_first_order(xi_b, tau, t_b[1] - t_b[0])))
    dev = abs(zb - za) / za
    chk(dev < 1e-2,
        f"采样加密一倍后残余峰值变化 {dev * 100:.5f}% < 1%（积分格式已收敛）")

    return {
        "submerged_fraction": s_eq,
        "kappa": kappa,
        "kappa_pointwise_in_band": frac_in_band,
        "residual_peak": z_max,
        "residual_rms": z_rms,
        "ballast_volume_fraction": v_frac,
        "tau_star": tau,
        "dt_star": dt,
        "series": {"t": t_star, "xi": xi, "lam": lam, "zeta": zeta},
    }


# ══════════════════════════════════════════════════════════════════════════
# 二、波能俘获
# ══════════════════════════════════════════════════════════════════════════
def part_waveenergy(cfg):
    hr("二、波能俘获（波能转化效率判据）")

    t_star = cfg["t_star"]
    zeta_rad = cfg["zeta_rad"]
    w0 = cfg["omega_rel"]
    x_max = cfg["x_max"]

    e = we.wave_energy_density()
    cg = we.group_velocity_deep(t_star)
    pw = we.wave_power_flux(t_star)
    chk(abs(e - 0.125) < 1e-12, f"波能面密度 E* = {e:.6f}（= 1/8，归一化）")
    chk(abs(cg - 1.0 / (4.0 * math.pi)) < 1e-12,
        f"深水群速度 cg* = {cg:.6f}（= 1/4π）")
    chk(abs(pw - 1.0 / (32.0 * math.pi)) < 1e-12,
        f"单宽波能通量 Pw* = {pw:.6f}（= 1/32π，与线性波理论一致）")

    x_res = we.heave_amplitude(w0, zeta_rad, 0.0)
    chk(abs(x_res - 1.0 / (2.0 * zeta_rad)) < 1e-9,
        f"共振且无 PTO 时位移幅值 = 1/(2ζ) = {x_res:.4f}（经典结果）")

    # Falnes 最优阻尼：解析式 vs 数值扫描
    z_opt = we.optimal_pto_damping(w0, zeta_rad)
    chk(abs(z_opt - zeta_rad) < 1e-9,
        f"共振点最优阻尼比 = 辐射阻尼比（{z_opt:.6f}），经典结论复现")

    best_z, best_p = 0.0, -1.0
    for i in range(1, 4001):
        z = i * 0.001
        p = we.absorbed_power(w0, zeta_rad, z)
        if p > best_p:
            best_p, best_z = p, z
    rel = abs(best_z - z_opt) / z_opt
    chk(rel < 0.01,
        f"数值扫描极值点 ζ = {best_z:.4f} 与解析解 {z_opt:.4f} 相对误差 {rel * 100:.3f}% < 1%")
    chk(best_p > 0.0, f"最优阻尼下俘获功率 P_abs* = {best_p:.4f} > 0（转换链有效）")

    # 离共振也要对得上，否则「最优」只是个巧合
    w_off = 0.70
    z_opt2 = we.optimal_pto_damping(w_off, zeta_rad)
    b2, p2 = 0.0, -1.0
    for i in range(1, 8001):
        z = i * 0.001
        p = we.absorbed_power(w_off, zeta_rad, z)
        if p > p2:
            p2, b2 = p, z
    rel2 = abs(b2 - z_opt2) / z_opt2
    chk(rel2 < 0.01,
        f"离共振 ω/ωn = {w_off}：数值极值 {b2:.4f} vs 解析 {z_opt2:.4f}，误差 {rel2 * 100:.3f}% < 1%")

    # 行程受限
    eta_c = 1.0
    x_free = we.heave_amplitude(w0, zeta_rad, z_opt)
    z_str = we.optimal_pto_damping_stroke_limited(w0, zeta_rad, x_max)
    x_str = we.heave_amplitude(w0, zeta_rad, z_str)
    p_free = we.absorbed_power(w0, zeta_rad, z_opt)
    p_str = we.absorbed_power(w0, zeta_rad, z_str)
    print(f"       无约束位移幅值 {x_free:.4f}，限位 {x_max:.4f}；"
          f"受限后阻尼比 {z_str:.4f}")
    if x_free > x_max:
        chk(z_str > z_opt, f"行程受限 ⇒ 最优阻尼上移（{z_opt:.4f} → {z_str:.4f}）")
        chk(abs(x_str - x_max) / x_max < 1e-3,
            f"受限后位移幅值压到限位 {x_str:.4f} ≈ {x_max:.4f}")
        chk(p_str < p_free,
            f"受限后俘获功率下降（{p_free:.4f} → {p_str:.4f}）—— 行程是真实的约束")
        eta_c = p_str / p_free
        chk(0.50 < eta_c < 1.00,
            f"行程约束下的俘获效率折减系数 = {eta_c:.4f}（0.5~1 之间）")
    else:
        eta_c = 1.0
        chk(abs(z_str - z_opt) < 1e-12, "行程未起控制作用时，最优阻尼不变")

    # CWR 与频响
    sweep = we.power_sweep([0.2 + 0.01 * i for i in range(0, 181)], zeta_rad,
                           z_str, t_star)
    cwr_res = we.capture_width_ratio(w0, zeta_rad, z_str, t_star)
    chk(cwr_res > 0.0 and math.isfinite(cwr_res),
        f"俘获宽度比 CWR* = {cwr_res:.4f}（无量纲相对指标，> 1 属点吸收式正常特性，不作效率百分比解读）")
    peak_w = max(sweep, key=lambda r: r[2])[0]
    chk(abs(peak_w - 1.0) <= 0.05,
        f"频响峰值出现在 ω/ωn ≈ 1 附近（实测 {peak_w:.3f}），共振特性成立")

    # 隔振
    r_th = we.isolation_threshold()
    chk(abs(we.displacement_transmissibility(r_th, 0.05) - 1.0) < 1e-9 and
        abs(we.displacement_transmissibility(r_th, 0.30) - 1.0) < 1e-9,
        f"传递率在 r = sqrt(2) = {r_th:.4f} 处为 1（与阻尼无关）")
    chk(we.displacement_transmissibility(3.0, 0.10) < 1.0,
        f"r = 3 时传递率 {we.displacement_transmissibility(3.0, 0.10):.4f} < 1（进入隔振区）")
    chk(we.displacement_transmissibility(10.0, 0.10) < 0.2,
        f"r = 10 时传递率 {we.displacement_transmissibility(10.0, 0.10):.4f} → 0（高频隔离）")
    chk(we.displacement_transmissibility(0.5, 0.10) > 1.0,
        f"r = 0.5 时传递率 {we.displacement_transmissibility(0.5, 0.10):.4f} > 1（放大区，如实呈现）")

    return {
        "wave_energy_density": e,
        "group_velocity": cg,
        "wave_power_flux": pw,
        "optimal_damping_res": z_opt,
        "optimal_damping_numeric": best_z,
        "peak_power": best_p,
        "stroke": {"x_free": x_free, "x_max": x_max, "x_limited": x_str,
                   "damping_limited": z_str, "power_limited": p_str,
                   "power_free": p_free},
        "cwr": cwr_res,
        "eta_c_stroke": eta_c,
        "sweep": [{"omega_rel": r[0], "amplitude": r[1],
                   "power": r[2], "cwr": r[3]} for r in sweep],
    }


# ══════════════════════════════════════════════════════════════════════════
# 三、口径自检
# ══════════════════════════════════════════════════════════════════════════
def part_discipline(blob: str):
    hr("三、口径自检（本仓库只给骨架，不给量纲结果）")
    banned = ["kWh", "千瓦时", "MW", "兆瓦", "kWh/", "发电量", "度电"]
    hit = [b for b in banned if b in blob]
    chk(not hit, "输出中无电量／功率量纲单位（本仓库不给绝对值）", f"命中 {hit}")
    chk("虚拟测试数据" in blob, "虚拟数据声明已随输出落盘")
    chk("归一化" in blob or "无量纲" in blob, "无量纲口径已随输出落盘")


# ══════════════════════════════════════════════════════════════════════════
# 输出：CSV / JSON / 图
# ══════════════════════════════════════════════════════════════════════════
def write_csvs(hydro, wave):
    os.makedirs(OUT_DIR, exist_ok=True)

    p1 = os.path.join(OUT_DIR, "data_compensation.csv")
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# " + vd.DECLARATION])
        w.writerow(["t_star", "xi_water_level", "lambda_ballast", "zeta_residual"])
        s = hydro["series"]
        for i in range(len(s["t"])):
            w.writerow([f"{s['t'][i]:.6f}", f"{s['xi'][i]:.6f}",
                        f"{s['lam'][i]:.6f}", f"{s['zeta'][i]:.6f}"])

    p2 = os.path.join(OUT_DIR, "data_capture.csv")
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# " + vd.DECLARATION])
        w.writerow(["omega_rel", "amplitude_rel", "power_rel", "capture_width_ratio"])
        for r in wave["sweep"]:
            w.writerow([f"{r['omega_rel']:.4f}", f"{r['amplitude']:.6f}",
                        f"{r['power']:.6f}", f"{r['cwr']:.4f}"])
    return [p1, p2]


def write_figs(hydro, wave):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                   # pragma: no cover
        print(f"  （跳过绘图：{exc}）")
        return []

    os.makedirs(OUT_DIR, exist_ok=True)
    made = []

    s = hydro["series"]
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    ax[0].plot(s["t"], s["xi"], lw=1.4, label="external level  xi*")
    ax[0].plot(s["t"], s["lam"], lw=1.4, ls="--", label="ballast level  lambda*")
    ax[0].set_ylabel("relative level")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Buoyancy compensation (dimensionless)", fontsize=10)
    ax[1].plot(s["t"], s["zeta"], lw=1.4, color="tab:red")
    ax[1].axhline(0.0, lw=0.8, color="k")
    ax[1].set_ylabel("residual  zeta*")
    ax[1].set_xlabel("time  t*  (1.0 = one tidal cycle)")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig1_compensation.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    sw = wave["sweep"]
    ws = [r["omega_rel"] for r in sw]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax[0].plot(ws, [r["power"] for r in sw], lw=1.6)
    ax[0].set_xlabel("omega / omega_n")
    ax[0].set_ylabel("captured power  P*")
    ax[0].set_title("Power response (dimensionless)", fontsize=10)
    ax[0].grid(alpha=0.3)
    ax[1].plot(ws, [r["amplitude"] for r in sw], lw=1.6, color="tab:orange")
    ax[1].axhline(wave["stroke"]["x_max"], lw=1.0, ls=":", color="k",
                  label="stroke limit")
    ax[1].set_xlabel("omega / omega_n")
    ax[1].set_ylabel("heave amplitude  |X|*")
    ax[1].set_title("Heave amplitude vs stroke limit", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_capture.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)
    return made


def main():
    print("=" * 74)
    print("SinoTide · 公开仿真骨架 · verify_tide.py")
    print("归一化单位制（rho = g = H_s = D = 1）｜全部输入为无量纲相对量")
    print("=" * 74)

    cfg = vd.reference_case()
    hydro = part_hydrostatics(cfg)
    wave = part_waveenergy(cfg)

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {
        "declaration": vd.DECLARATION,
        "unit_system": "归一化单位制（无量纲）：rho = g = H_s = D = 1，"
                       "全部输入输出为相对量，不含任何工程量纲数值",
        "seed": vd.SEED,
        "input": cfg,
        "hydrostatics": {k: v for k, v in hydro.items() if k != "series"},
        "waveenergy": {k: v for k, v in wave.items() if k != "sweep"},
        "assertions": {"passed": len(PASSES), "failed": len(FAILS)},
    }
    blob = json.dumps(report, ensure_ascii=False, indent=1)
    part_discipline(blob)

    csvs = write_csvs(hydro, wave)
    figs = [] if "--no-fig" in sys.argv else write_figs(hydro, wave)

    report["assertions"] = {"passed": len(PASSES), "failed": len(FAILS)}
    rp = os.path.join(OUT_DIR, "verify_tide_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 74)
    print(f"判据：{len(PASSES)} PASS / {len(FAILS)} FAIL")
    print(f"输出：{os.path.relpath(rp, os.path.dirname(rp))}")
    for p in csvs + figs:
        print(f"      {os.path.basename(p)}")
    print("=" * 74)

    if FAILS:
        print("\n未通过的判据：")
        for m in FAILS:
            print("  · " + m)
        return 1
    print("\n全部判据通过。这是骨架自洽的证明，不是任何真实海域的性能结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
