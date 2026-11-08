# -*- coding: utf-8 -*-
"""SinoTide · 公开仿真骨架（无量纲）

本包只包含教科书级的通用物理关系，全部以**无量纲比值**表达：
不含任何工程尺寸、构型参数、控制时序或标定系数。

模块
----
hydrostatics : 静力平衡、浮力补偿率、舱容充裕度
waveenergy   : 波能通量、单自由度垂荡俘能、最优 PTO 阻尼、频率隔离
virtualdata  : 虚拟测试序列（固定种子，非任何真实海域实测）
"""

__all__ = ["hydrostatics", "waveenergy", "virtualdata"]
__version__ = "1.0.0"
