"""Gaussian Mixture Model baseline package for market regime detection.

This is the baseline compared against the Gaussian HMM: it clusters each
observation independently with no notion of a time-dependent transition
between regimes. Comparing it to the HMM tests whether modeling that
transition structure is actually worth the extra complexity.
"""
