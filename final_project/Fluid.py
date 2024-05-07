from dataclasses import dataclass


@dataclass
class Fluid:
    V_inf: float
    rho: float
    mu: float
