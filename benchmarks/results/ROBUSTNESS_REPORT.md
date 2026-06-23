# Robustness benchmark — H-infinity differential game vs LQR

Formal metric: closed-loop worst-case L2 gain `||G_zw||inf` from disturbance
to the weighted performance output (lower = more robust). `gamma = 1.3 * gamma*`.
Nominal LQ cost is the disturbance-free regulation cost (the price of robustness).

Practical significance is judged by the *absolute* LQR gain `‖G‖∞`: a relative
reduction only matters when the disturbance is not already negligibly rejected.

| system | ‖G‖∞ LQR | ‖G‖∞ H∞ | worst-case ↓ | time-peak ↓ | nominal cost ↑ | significant? |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| cart (double integrator) | 1.554 | 1.214 | **+21.8%** | +46.5% | +13.8% | ✅ yes |
| mass-spring-damper | 0.4564 | 0.3746 | **+17.9%** | +37.9% | +6.7% | ✅ yes |
| DC motor servo | 17.3 | 13.45 | **+22.2%** | +30.8% | +21.4% | ✅ yes |
| car cruise control | 0.01067 | 0.009863 | **+7.6%** | +10.9% | +3.0% | ➖ gain≈0 |
| car lateral (bicycle) | 0.0001043 | 7.755e-05 | **+25.6%** | +28.9% | +16.7% | ➖ gain≈0 |
| inverted pendulum | 1.92 | 1.253 | **+34.7%** | +69.4% | +41.8% | ✅ yes |
| aircraft short-period | 1.375 | 1.17 | **+14.9%** | +26.7% | +11.4% | ✅ yes |
| quadrotor (planar) | 0.4251 | 0.3168 | **+25.5%** | +27.0% | +6.6% | ✅ yes |
| flexible two-mass | 0.8279 | 0.6417 | **+22.5%** | +24.6% | +32.0% | ✅ yes |
| PVTOL aircraft | 0.07087 | 0.05201 | **+26.6%** | +33.9% | +1.6% | 🟡 marginal |
| active suspension | 13.97 | 11.98 | **+14.2%** | -2.4% | +9.5% | ✅ yes |
| seismic building | 0.1687 | 0.1283 | **+24.0%** | +36.2% | +11.3% | ✅ yes |
| gantry crane | 1.201 | 1.1 | **+8.4%** | +3.9% | +1.1% | ✅ yes |

**13/13** systems show a relative worst-case-gain reduction; of these, **10** are *practically* significant (non-negligible absolute gain).

Honest reading: robust (H-infinity) control — a controller-vs-disturbance differential
game — reduces the worst-case disturbance gain on every system here, most on the
lightly-damped / unstable plants (inverted pendulum, DC motor, flexible structure, ...)
where the LQR leaves a sharp resonant peak, and always at a documented nominal-cost price.
On the well-damped cars (cruise, bicycle) the absolute gain is already ~0, so the relative
reduction, while real, is not practically meaningful — the disturbance is already rejected.
