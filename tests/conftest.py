"""Shared pytest configuration.

Forces a non-interactive matplotlib backend so the plotting paths can be
exercised in headless CI without opening windows.
"""

import matplotlib

matplotlib.use("Agg")
