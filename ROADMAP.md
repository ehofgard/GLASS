# GLASS fork: notes and roadmap

Goals: use GLASS as a backbone for **disordered crystalline structures** and incorporate **more experimental spectra**, ideally **without surrogate models**.

## Notes on the current code

### How it works
- Score-based diffusion prior trained on a few periodic cells (a-Si example: 10 cells × 216 atoms)
  - Variance-exploding noise, σ(t) = k·t, k = 0.8 Å (`glass/diffusion/__init__.py`)
  - MeshGraphNets score model: 5 layers, 200 features, 5 Å periodic graph (`glass/lit/modules/prior.py`)
- Guidance on the estimated clean structure x̂₀ = x + σ²·score (`glass/lit/modules/likelihood.py`)
  - Guidance term: −(ρ / ‖mismatch‖) · ∇‖target − F(x̂₀)‖², so the best ρ depends on cell size and grid
  - Targets: pdf, adf, xrd, nd (analytic); exafs, xanes (SpecNet surrogate)
- Sampling: unguided pass t = 1 → 0.001, then guided pass t = 0.6 → 0.001 (512 steps each); optional `--min-dist` veto

### Data
- In repo: 30 a-Si cells (1.5 / 2.5 / 3.5 g/cm³), trained score model, Si EXAFS + XANES surrogate checkpoints
- FigShare (https://doi.org/10.6084/m9.figshare.34515966, ~9 GB)
  - `06_models_xas_training_data.zip` (1.2 GB): FEFF per-atom spectra (Si, SiO₂, possibly more)
  - `06_models_xas_surrogate_models.zip`, `06_models_score_models.zip`: weights for other systems
  - Benchmarks: a-Si, Cu-Zr, a-C, SiO₂, Zr-Ni-Al; paracrystalline Si, liquid S, amorphous ice

### Issues found
- Score model and SpecNet are not rotation-equivariant
  - Edge features are raw (dx, dy, dz, |r|); symmetry only learned via random-rotation augmentation (`glass/lit/datamodules/structure_spec.py`)
  - Matters most with tiny training sets and for surrogate guidance gradients
  - Test first: rotate cell + lattice, check how far the score is from rotating with it at several t
  - Cheap fix: distance-only edge features + score as Σⱼ w(h_ij)·r̂_ij (EGNN-style); stronger: MACE / NequIP
- Surrogate testing is weak
  - Validation holds out 10% of atoms inside training structures (leaks via shared environments); no held-out structures
  - Reported R² (EXAFS 0.95, XANES 0.99) is on training data
  - `best.ckpt` chosen by train loss of raw weights; sampling uses EMA weights
  - Never checked on x̂₀ structures seen during guidance, or whether gradients reduce true FEFF error
  - Spectra stored without an energy / k grid
- Guidance limitations
  - One guidance target per run
  - Experimental data: plain L2 after linear interpolation; no scale / background / resolution / error bars
  - PDF and ADF guidance run on CPU; PDF valid only up to half the cell length
- Gaps for disordered crystals
  - Species and atom count fixed → no substitutional disorder or vacancies
  - XRD / ND is an isolated-cluster Debye sum (no periodic images); wrong for crystals, memory ∝ N²·M
  - Receptive field ~25 Å (5 layers × 5 Å); longer-range correlations only via guidance
  - Tersoff guidance / annealing / metrics are pure Si only
- Training note: batches need ≳1000 atoms or the loss sticks at 3.0

## Differentiable spectra (no surrogates)

### Analytic forward models (exact formulas, cheap)
- Total scattering
  - S(Q), F(Q), G(r), D(r) (X-ray / neutron / electron)
  - Apply experiment's finite Q range + Lorch window to model G(r)
  - Element-specific partial PDFs (isotope-substitution neutron, anomalous X-ray)
- Pair angle distribution function (PADF) Θ(r, r′, θ)
  - Spherical-harmonic expansion of interatomic vectors; 3- and 4-body terms (medium-range order)
  - Measurable via fluctuation X-ray / electron diffraction
  - Compare only the l values the experiment resolves; mask the r = r′ diagonal
  - Validate against a brute-force histogram on small cells
- Diffraction
  - Electron diffraction / ePDF (kinematic, electron form factors)
  - Periodic powder pattern: Bragg peaks + diffuse scattering
  - 3D single-crystal diffuse scattering via FFT (scales to large supercells)
  - 3D-ΔPDF (Fourier transform of the diffuse scattering)
  - SAXS / SANS (low-Q: voids, density fluctuations)

### Physics formulas with precomputed parameters
- EXAFS via path-sum formula, scattering amplitudes/phases precomputed by FEFF (replaces SpecNet)
  - Precompute |f(k)|, δ(k), λ(k) per element pair (FEFF / xraylarch)
  - Single-scattering forward model
  - Fit S₀² and ΔE₀ at each step
  - Turn guidance on only at low noise (small t)
  - Add 3-atom multiple-scattering paths (needed for crystals)
  - Validate against full FEFF spectra on held-out structures (FigShare XAS data)
- Smooth coordination numbers
  - XPS core-level shift populations
  - NMR Qⁿ speciation
  - Bond-valence sums
- Raman TO-peak width ↔ bond-angle spread (Beeman relation, a-Si)
- Point-charge electric field gradient
  - NQR / quadrupolar NMR
  - Mössbauer quadrupole splitting

### Needs a potential (decide if OK)
- Vibrational density of states / inelastic neutron scattering (Hessian of an ML potential)
- IR / Raman intensities (also needs a dipole / polarisability model)

### Not differentiable → use to filter / reweight outputs only
- XANES / ELNES via FEFF or FDMNES on generated structures
- NMR chemical shifts (DFT)

## Infrastructure
- Guidance
  - Several guidance targets in one run, each with its own weight
  - Fit scale + background at each step; per-point error bars in the likelihood
- Performance
  - Neighbour lists / GPU for g(r) and ADF on large cells
- Model
  - Rotation-equivariant score model
  - Changing atom types / vacancies (only if substitutional disorder is needed)
- Surrogate testing (for SpecNet baseline / validation)
  - Structure-level held-out split, stratified by type and density; leave-one-density-out
  - Evaluate on x̂₀ at several noise levels; rotation check; gradient check vs FEFF
  - Validation pass on EMA weights for checkpoint selection

## Priority order
1. Total scattering with experimental Q range + multiple targets per run
2. EXAFS path-sum
3. FFT-based periodic diffraction / diffuse scattering
4. FEFF / FDMNES filter for XANES
