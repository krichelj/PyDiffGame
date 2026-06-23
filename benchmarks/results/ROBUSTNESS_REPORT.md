# Robustness benchmark — H-infinity differential game vs LQR

Formal metric: closed-loop worst-case L2 gain `||G_zw||inf` from disturbance
to the weighted performance output (lower = more robust). `gamma = 1.3 * gamma*`.
Nominal LQ cost is the disturbance-free regulation cost (the price of robustness).

| system | ‖G‖∞ LQR | ‖G‖∞ H∞ | worst-case ↓ | time-peak ↓ | nominal cost ↑ | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| cart (double integrator) | 1.554 | 1.214 | **+21.8%** | +46.5% | +13.8% | ✅ boost |
| mass-spring-damper | 0.456 | 0.375 | **+17.9%** | +37.9% | +6.7% | ✅ boost |
| DC motor servo | 17.296 | 13.448 | **+22.2%** | +30.8% | +21.4% | ✅ boost |
| car cruise control | 0.011 | 0.010 | **+7.6%** | +10.9% | +3.0% | ✅ boost |
| car lateral (bicycle) | 0.000 | 0.000 | **+25.6%** | +28.9% | +16.7% | ✅ boost |
| inverted pendulum | 1.920 | 1.253 | **+34.7%** | +69.4% | +41.8% | ✅ boost |
| aircraft short-period | 1.375 | 1.170 | **+14.9%** | +26.7% | +11.4% | ✅ boost |
| quadrotor (planar) | 0.425 | 0.317 | **+25.5%** | +27.0% | +6.6% | ✅ boost |
| flexible two-mass | 0.828 | 0.642 | **+22.5%** | +24.6% | +32.0% | ✅ boost |
| PVTOL aircraft | 0.071 | 0.052 | **+26.6%** | +33.9% | +1.6% | ✅ boost |

**10/10** systems show a worst-case robustness boost.

Honest reading: H-infinity (a controller-vs-disturbance differential game) reduces the
worst-case disturbance gain — most on lightly-damped / resonant systems — at the cost of
some nominal performance. Where the boost is small, the plant is already well-damped and
there is little worst-case gain to recover.
