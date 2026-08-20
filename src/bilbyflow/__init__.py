"""
bilbyflow — NPE parameter estimation for gravitational-wave events.

An On-The-Fly-Simulation NPE pipeline: 
a normalising flow conditioned on whitened H1/L1 strain 
(with PSD + observable-amplitude conditioning) 
is trained against extrinsic parameters synthesised per batch, 
then its posteriors are importance-reweighted to the exact likelihood.

Subpackages:
  io           config, strain I/O, published-sample loading
  data         waveform/sky/PSD/noise banks, whitening, the dataset, standardiser
  coordinates  sky reparam (ra/dec <-> dt_HL/phi_det) and parameter transforms
  nn           embedding, conditional flow, auxiliary-supervision head
  training     curriculum trainer, checkpoints, losses, BN recalibration
  likelihood   waveform generator + SNR utilities
  inference    NPE sampling, priors, reweighting, synthetic-phase, two-stage IS
  diagnostics  PSIS reliability, data-consistency audits, BatchNorm checks
  plotting     efficiency summaries, corner overlays, weight + training curves

Top-level imports are kept light so `import bilbyflow` does not pull in torch,
bilby, or sbi. Reach into the subpackage you need (e.g.
`from bilbyflow.inference import reweight_event`).
"""

__version__ = "1.0.1"

__all__ = ["__version__"]