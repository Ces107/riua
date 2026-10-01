"""Meteorological diagnostics shown as "drivers" next to the risk levels (never used to set them).

``ingredients``      ingredients-based heavy-rain diagnostics (MetPy) from live model fields: ``compute()``,
                     ``point_profile()``.
``hindcast_check``   the same diagnostics from the archived (stitched) forecasts of a past date.

This package imports nothing by itself: MetPy (heavy) is only loaded when ``ingredients`` is imported.
"""
