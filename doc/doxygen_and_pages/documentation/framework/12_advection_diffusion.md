# Advection-Diffusion {#advection-diffusion}

Algebraic flux-corrected transport (FCT) for scalar advection–diffusion on the finite-volume (FV) hex
shell grid (fusing two wedges into one hex cell). See `fct_advection_diffusion.hpp`.
 
---
## Physical problem and discretisation assumption
 
We discretise the scalar transport equation in **conservative form**
\f[
    \frac{\partial T}{\partial t}
    + \nabla \cdot (\mathbf{u}\, T)
    - \nabla \cdot (\kappa\, \nabla T)
    = f.
\f]
Here \f$T\f$ is the transported scalar (e.g. temperature), \f$\mathbf{u}\f$ the given velocity
field, \f$\kappa \geq 0\f$ the physical diffusivity, and \f$f\f$ an optional volumetric source
term [T/time] (e.g. radiogenic heat production).
 
The conservative form is used throughout — in the PDE, in the derivation, and in the
implementation — because it maps directly onto face fluxes and the finite-volume divergence
theorem.  The discrete analogue of \f$\nabla\cdot(\mathbf{u}T)\f$ is
\f$\sum_j \beta_{ij}\,T_{ij}^{\mathrm{face}}\f$, where the face flux
\f$\beta_{ij} = \int_{\partial K_{ij}} \mathbf{u}\cdot\hat{n}\,\mathrm{d}S\f$ is evaluated
by numerical quadrature.
 
### Compressibility
 
**On divergence-free vs. compressible velocity fields.**
Expanding the conservative flux:
\f$\nabla\cdot(\mathbf{u}T) = \mathbf{u}\cdot\nabla T + T\,\nabla\cdot\mathbf{u}\f$.
For a **passive scalar** such as temperature the physically correct equation is the
**advective form** \f$\partial_t T + \mathbf{u}\cdot\nabla T = \ldots\f$ — a fluid parcel's
temperature changes only through diffusion and sources, not compression.  The two forms
are equivalent only when \f$\nabla\cdot\mathbf{u} = 0\f$ (Boussinesq).
 
The `subtract_divergence` flag (available on all step functions and on
`UnsteadyAdvectionDiffusion`) subtracts the discrete divergence
\f$(\sum_j \beta_{ij})/M_{ii}\f$ from the upwind operator, recovering the advective form.
It **defaults to `true`** because:
- when \f$\nabla\cdot\mathbf{u} = 0\f$ the correction is exactly zero and costs nothing, and
- when \f$\nabla\cdot\mathbf{u} \neq 0\f$ keeping the conservative form introduces an
  unphysical \f$T\,\nabla\cdot\mathbf{u}\f$ source/sink in the temperature equation.
See Step 4 of the derivation below for the algebra.
 
**Additional source terms for compressible flow.**  Subtracting the divergence error is a
necessary but not sufficient correction for compressible or anelastic velocity fields.
The full temperature (or entropy) equation under compressible flow contains further source
terms — most notably adiabatic heating/cooling (proportional to \f$Dp/Dt\f$, i.e. depending
on pressure and velocity) and viscous dissipation.  These are problem-specific and their
derivation is left to the user; they can be supplied through the `source` parameter as a
pre-computed volumetric rate \f$f\f$ [T/time].
 
### Boundary conditions
 
Dirichlet boundary conditions \f$T = T_{\mathrm{bc}}\f$ at the CMB (\f$r = r_{\min}\f$) and
the outer surface (\f$r = r_{\max}\f$) are enforced strongly (pointwise) at the end of each
time step by overwriting the boundary cell values with the prescribed constants
\f$T_{\mathrm{cmb}}\f$ and \f$T_{\mathrm{surface}}\f$ (see `DirichletBCs` in `helpers.hpp`).
This is first-order in time: the boundary value is fixed after each step and acts as a Dirichlet
datum for the stencil in the next step.  No lateral in/outflow boundary conditions are needed
since flow does not cross the radial shell boundaries by construction.
 
---
## From the continuous PDE to the discrete scheme
 
Integrate the conservative PDE over a single cell \f$K_i\f$:
\f[
    \frac{\mathrm{d}}{\mathrm{d}t}\int_{K_i} T\,\mathrm{d}x
    + \int_{K_i} \nabla\cdot(\mathbf{u}T)\,\mathrm{d}x
    - \int_{K_i} \nabla\cdot(\kappa\nabla T)\,\mathrm{d}x
    = \int_{K_i} f\,\mathrm{d}x.
\f]
 
**Step 1 — Cell average.**  Represent \f$T\f$ as piecewise constant: \f$T|_{K_i} \approx T_i\f$.
Then \f$\int_{K_i} T\,\mathrm{d}x = |K_i|\,T_i =: M_{ii}\,T_i\f$ and
\f$\int_{K_i} f\,\mathrm{d}x \approx M_{ii}\,f_i\f$.
 
**Step 2 — Divergence theorem on the advective term** \f$\nabla\cdot(\mathbf{u}T)\f$.
The volume integral becomes a surface integral, split into one contribution per face:
\f[
    \int_{K_i} \nabla\cdot(\mathbf{u}T)\,\mathrm{d}x
    = \oint_{\partial K_i} T\,(\mathbf{u}\cdot\hat{n})\,\mathrm{d}S
    \approx \sum_{j\in\mathcal{N}(i)} T_{ij}^{\mathrm{face}}\,\beta_{ij},
\f]
where \f$\beta_{ij} = \int_{\partial K_i \cap \partial K_j} \mathbf{u}\cdot\hat{n}\,\mathrm{d}S\f$
(positive = net outflow from \f$K_i\f$) is computed by numerical quadrature over the wedge
faces (see `GeometryHelper::compute_geometry`), and \f$T_{ij}^{\mathrm{face}}\f$ is the face
value chosen by the reconstruction (Step 5).
 
**Step 3 — Divergence theorem on the diffusive term** \f$\nabla\cdot(\kappa\nabla T)\f$.
\f[
    \int_{K_i} \nabla\cdot(\kappa\nabla T)\,\mathrm{d}x
    = \oint_{\partial K_i} \kappa\,\nabla T\cdot\hat{n}\,\mathrm{d}S
    \approx \sum_{j\in\mathcal{N}(i)} d_{ij}\,(T_j - T_i),
\f]
where the normal gradient across face \f$(i,j)\f$ is approximated by a central difference along
the cell-to-cell vector.  The two-point diffusion coefficient
\f[
    d_{ij} = \kappa\,
             \frac{\mathbf{S}_f^{(j)}\cdot\mathbf{S}_f^{(j)}}
                  {(\mathbf{x}_j - \mathbf{x}_i)\cdot\mathbf{S}_f^{(j)}}
\f]
accounts for mesh non-orthogonality via the area-weighted face normal
\f$\mathbf{S}_f^{(j)} = \int_{\partial K_{ij}} \hat{n}_{ij}\,\mathrm{d}S\f$.
It is exact on orthogonal meshes and first-order consistent on smooth non-orthogonal ones.
 
**Step 4 — Collecting terms.**  Substituting Steps 1–3 gives the semi-discrete ODE for cell
\f$i\f$:
\f[
    M_{ii}\,\dot{T}_i
    + \underbrace{\sum_j \beta_{ij}\,T_{ij}^{\mathrm{face}}}_{\text{discrete } \nabla\cdot(\mathbf{u}T)}
    - \underbrace{\sum_j d_{ij}(T_j - T_i)}_{\text{discrete } \nabla\cdot(\kappa\nabla T)}
    = M_{ii}\,f_i.
\f]
Forward Euler in time gives the explicit low-order predictor (Stage 1 of FCT); backward Euler
gives the semi-implicit system solved with FGMRES.
 
**Divergence correction (`subtract_divergence = true`).**  With the upwind reconstruction,
the advective term expands to
\f$\sum_j \beta_{ij}^+\,T_i + \sum_j \beta_{ij}^-\,T_j\f$.
Subtracting the discrete divergence \f$T_i \sum_j \beta_{ij}\f$ gives
\f[
    \sum_j \beta_{ij}^+\,T_i + \sum_j \beta_{ij}^-\,T_j
    - T_i \underbrace{\left(\sum_j \beta_{ij}^+ + \sum_j \beta_{ij}^-\right)}_{\sum_j \beta_{ij}}
    = \sum_j \beta_{ij}^-\,(T_j - T_i),
\f]
which is the discrete **advective form** \f$\mathbf{u}\cdot\nabla T\f$: only inflow faces
contribute, and the difference \f$T_j - T_i\f$ is the upwind gradient.  Concretely, the
correction reduces to subtracting \f$\sum_j \beta_{ij}\f$ from the diagonal of the upwind
operator before the time step — one cheap operation with no additional geometry work, since
all \f$\beta_{ij}\f$ are already computed.
 
**Step 5 — Face reconstruction.**  Two choices for \f$T_{ij}^{\mathrm{face}}\f$:
- *First-order upwind:* \f$T_{ij}^{\mathrm{face}} = T_i\f$ if \f$\beta_{ij} \geq 0\f$
  (flow leaves \f$K_i\f$), \f$T_j\f$ if \f$\beta_{ij} < 0\f$ (flow enters \f$K_i\f$).
  Monotone but strongly diffusive.
- *Second-order central:* \f$T_{ij}^{\mathrm{face}} = \frac{1}{2}(T_i + T_j)\f$.  More
  accurate but generates oscillations near sharp gradients.
 
FCT combines both: the upwind step guarantees monotonicity; the antidiffusive flux
\f$f_{ij} = \frac{|\beta_{ij}|}{2}(T_i - T_j)\f$ — the difference between the central and
upwind face values, multiplied by the face flux magnitude — is added back in a limited fashion
to recover accuracy where the solution is smooth.
 
---
## How FCT / AFC works — the key idea
 
The fundamental challenge in scalar transport is the tension between two conflicting goals:
- **Accuracy**: high-order schemes (central differencing, Lax-Wendroff, …) give sharp,
  low-diffusion results, but produce spurious oscillations ("wiggles") near steep gradients.
- **Monotonicity**: first-order upwind is oscillation-free but smears out sharp features
  quickly because it adds large amounts of numerical diffusion.
 
FCT resolves this conflict by treating the two schemes as complementary:
 
1. **Start with the safe scheme (upwind).**  Take one upwind time step to obtain a monotone
   low-order solution \f$T^L\f$.  This is guaranteed not to create new local extrema (i.e. no
   overshoots or undershoots), but it is too diffusive.
 
2. **Compute what the upwind step "took away".**  The difference between a hypothetical
   central-differencing step and the upwind step is the *antidiffusive flux* \f$f_{ij}\f$.
   Adding it back in full would recover second-order accuracy, but would also restore the
   oscillations.
 
3. **Add back as much as possible without violating monotonicity (Zalesak limiter).**
   For each cell, determine how much of the antidiffusive flux it can absorb before its value
   would exceed the range \f$[T^{\min}_i, T^{\max}_i]\f$ of its low-order neighbours.  Compute
   a limiting factor \f$\alpha_{ij} \in [0, 1]\f$ per face — the minimum of what the sending
   cell can give (\f$R^+\f$ or \f$R^-\f$) and what the receiving cell can accept (\f$R^-\f$
   or \f$R^+\f$) — and apply \f$T^{n+1}_i = T^L_i + \sum_j \alpha_{ij} \tilde{f}_{ij}\f$.
   This is the *local extremum diminishing* (LED) property: no cell can become a new local
   maximum or minimum that was not already present in \f$T^L\f$.
 
The result is a scheme that is as close to second-order as the local solution smoothness
allows, while being provably free of spurious oscillations.
 
**Physical diffusion** (\f$\kappa > 0\f$) is treated differently: it is built into the
low-order predictor (Stage 1) and is *not* FCT-limited.  Physical diffusion is already a
smoothing process; limiting it would reduce it below the physical value, which is incorrect.
Only the purely numerical antidiffusive flux (the gap between central and upwind *advection*)
is subject to FCT limiting.
 
**Source terms** (\f$f \neq 0\f$) are added directly to the low-order predictor as
\f$+\Delta t\,f_i\f$.  They are not FCT-limited, consistent with the physical interpretation
that the source genuinely changes the local \f$T\f$ value.
 
---
## Discrete setting
 
Each cell \f$K_i\f$ carries a cell-averaged value \f$T_i\f$.  The semi-discrete system is
\f[
    M_{ii}\,\dot{T}_i
    + \sum_j \beta_{ij}\, T_j^{\mathrm{upw}}
    - \sum_j d_{ij}\,(T_j - T_i)
    = M_{ii}\,f_i,
\f]
where
- \f$M_{ii} = |K_i|\f$ — cell volume (mass-matrix diagonal for piecewise-constant FV),
- \f$\beta_{ij}\f$ — face-normal velocity flux \f$\int_{\partial K_{ij}} \mathbf{u}\cdot\hat{n}\,\mathrm{d}S\f$
  (positive = outflow from \f$K_i\f$),
- \f$\mathbf{S}_f^{(j)}\f$ — area-weighted outward face normal
  \f$\int_{\partial K_{ij}} \hat{n}_{ij}\,\mathrm{d}S\f$,
- \f$d_{ij}\f$ — two-point diffusion coefficient (see Step 3 above),
- \f$\mathbf{x}_i, \mathbf{x}_j\f$ — cell centres,
- \f$T_j^{\mathrm{upw}} = T_j\f$ if \f$\beta_{ij} < 0\f$ (inflow), \f$T_i\f$ otherwise.
 
All geometric quantities are computed by `GeometryHelper::compute_geometry`
(see `geometry_helper.hpp`).
 
---
## Explicit FCT scheme
 
Each explicit time step consists of four stages.
 
### Stage 1 — Low-order predictor (upwind + physical diffusion + source)
 
Default (`subtract_divergence = true`): advective upwind form.
\f[
    T_i^L = T_i^n
    - \frac{\Delta t}{M_{ii}}
      \Bigl[
        \underbrace{\sum_j \beta_{ij}^-\,(T_j^n - T_i^n)}_{\text{upwind, advective}}
        + \underbrace{\sum_j d_{ij}\,(T_i^n - T_j^n)}_{\text{physical diffusion}}
      \Bigr]
    + \Delta t\, f_i.
\f]
With `subtract_divergence = false`: conservative form (only needed if \f$\nabla\cdot(\mathbf{u}T)\f$
is genuinely the intended equation rather than \f$\mathbf{u}\cdot\nabla T\f$):
\f[
    T_i^L = T_i^n
    - \frac{\Delta t}{M_{ii}}
      \Bigl[
        \underbrace{\beta_{ii}^+\, T_i^n
        + \sum_{j \neq i} \beta_{ij}^-\, T_j^n}_{\text{upwind, conservative}}
        + \underbrace{\sum_j d_{ij}\,(T_i^n - T_j^n)}_{\text{physical diffusion}}
      \Bigr]
    + \Delta t\, f_i.
\f]
Both forms use \f$\beta^+ = \max(\beta, 0)\f$, \f$\beta^- = \min(\beta, 0)\f$.
Both are monotone (LED) under the CFL condition below.
When no source is provided the last term vanishes.
 
### Stage 2 — Antidiffusive flux computation
 
The antidiffusive flux on face \f$(i,j)\f$ is the difference between what a central scheme
and the upwind scheme would have transported:
\f[
    f_{ij} = \frac{|\beta_{ij}|}{2}\,(T_i^n - T_j^n).
\f]
Fluxes are stored pre-scaled as \f$\tilde{f}_{ij} = \frac{\Delta t}{M_{ii}}\,f_{ij}\f$.
A positive \f$f_{ij}\f$ means the central scheme would have pushed more of \f$T_i^n\f$ into
the face than upwind did (or vice versa for negative).
 
### Stage 3 — Zalesak limiter
 
For each cell \f$i\f$, sum the positive and negative antidiffusive contributions it would
receive, and find the local allowable range from the low-order neighbours:
\f[
    P_i^{\pm} = \sum_{j:\,\pm f_{ij} > 0} f_{ij},
    \qquad
    Q_i^+ = \max_j(T_j^L) - T_i^L \geq 0,
    \qquad
    Q_i^- = \min_j(T_j^L) - T_i^L \leq 0.
\f]
The nodal correction factors cap how much flux can be received before the cell would violate
its neighbours' range:
\f[
    R_i^+ = \min\!\left(1,\,\frac{Q_i^+}{P_i^+}\right),
    \qquad
    R_i^- = \min\!\left(1,\,\frac{Q_i^-}{P_i^-}\right)
    \quad (\text{set to } 1 \text{ if denominator is zero}).
\f]
 
### Stage 4 — Limited correction
 
For each face, the limiter takes the *minimum* of what the two cells on either side can handle:
\f[
    \alpha_{ij} =
    \begin{cases}
      \min(R_i^+,\, R_j^-) & \text{if } f_{ij} > 0, \\
      \min(R_i^-,\, R_j^+) & \text{if } f_{ij} < 0.
    \end{cases}
\f]
The factor \f$\alpha_{ij} = 1\f$ means full second-order correction is safe; \f$\alpha_{ij} = 0\f$
means the flux is entirely suppressed (the solution is locally at an extremum).  The final
solution
\f[
    T_i^{n+1} = T_i^L + \sum_j \alpha_{ij}\,\tilde{f}_{ij}
\f]
satisfies \f$\min_j T_j^L \leq T_i^{n+1} \leq \max_j T_j^L\f$ by construction (LED).
 
After the FCT correction, Dirichlet boundary conditions are applied by overwriting the boundary
radial cells with \f$T_{\mathrm{cmb}}\f$ and \f$T_{\mathrm{surface}}\f$ (see
`apply_dirichlet_bcs` in `helpers.hpp`).
 
---
## Semi-implicit FCT variant
 
For large time steps (CFL > 1) the explicit predictor becomes unstable.  In the semi-implicit
variant the low-order predictor is obtained instead by solving the backward-Euler system
\f[
    (M + \Delta t\,A_{\mathrm{upw}} + \Delta t\,D)\,T^L
    = M\,T^n + \Delta t\,M\,f,
\f]
where \f$A_{\mathrm{upw}}\f$ is the upwind advection–divergence matrix and \f$D\f$ is the
diffusion matrix assembled from the \f$d_{ij}\f$ coefficients.  When no source is present the
RHS reduces to \f$M\,T^n\f$.  The solve is performed with FGMRES (see
`UnsteadyAdvectionDiffusion::compute_rhs` and `UnsteadyAdvectionDiffusion::apply_impl` in
`advection_diffusion.hpp`).  The antidiffusive fluxes and Zalesak limiter (Stages 2–4)
are identical to the explicit variant.
 
When `subtract_divergence = true` is passed to `UnsteadyAdvectionDiffusion`, the diagonal
of \f$A_{\mathrm{upw}}\f$ is reduced by \f$\sum_j \beta_{ij}\f$ per cell, so the implicit
system likewise solves the advective form \f$\mathbf{u}\cdot\nabla T\f$ rather than
\f$\nabla\cdot(\mathbf{u}T)\f$.
 
**Note on Dirichlet BCs in the implicit solve.**  The linear system does not enforce boundary
conditions; boundary cells are free unknowns during the FGMRES solve and are overwritten only
after the FCT correction.  This is first-order accurate at the boundary.  A more accurate
treatment (row replacement in the linear system) is not yet implemented.
 
---
## Stability
 
The **explicit** variants require a CFL condition
\f[
    \mathrm{CFL} = \Delta t\,
    \frac{\max_i \sum_j \beta_{ij}^+}{M_{ii}} < 1.
\f]
The **semi-implicit** variant is unconditionally stable for the low-order predictor;
the Zalesak correction cannot introduce new extrema.
 
**Practical time step estimate.**  The flux-based CFL above requires evaluating all face
integrals, which is not convenient at setup time.  A safe and practical substitute is
\f[
    \Delta t < C_{\mathrm{CFL}} \,\frac{h_{\min}}{|\mathbf{u}|_{\max}},
\f]
where \f$h_{\min}\f$ is the smallest cell edge length in the mesh and
\f$|\mathbf{u}|_{\max}\f$ is the global maximum velocity magnitude.  A conservative choice
is \f$C_{\mathrm{CFL}} \approx 0.1\text{–}0.5\f$; start at the lower end and increase
once the solution behaviour is understood.  `grid::shell::min_radial_h` provides the minimum
radial layer thickness, which is typically the smallest length scale on shell meshes.  The
lateral mesh size at refinement level \f$\ell\f$ scales as \f$\pi/(2 \cdot 2^\ell)\f$ and
can be smaller at coarse levels, so verify both directions.
 
---
## Face / neighbour ordering
 
Each hexahedral cell has 6 neighbours, indexed as:
| index | direction |
|-------|-----------|
| 0     | \f$x-1\f$ |
| 1     | \f$x+1\f$ |
| 2     | \f$y-1\f$ |
| 3     | \f$y+1\f$ |
| 4     | \f$r-1\f$ (inner shell) |
| 5     | \f$r+1\f$ (outer shell) |
 
---
## Usage
 
### Which scheme to choose?
 
**Use `fct_explicit_step` for production runs.**  Pure upwind (`upwind_explicit_step`) is
provided as a debugging baseline and for verifying the geometry and time-step choice, but it
is very diffusive: sharp temperature gradients smear out quickly and the thermal structure is
lost long before physical equilibrium is reached.  FCT recovers near-second-order accuracy
in smooth regions while remaining oscillation-free — use it.
 
The semi-implicit variant (`fct_semiimplicit_step` + FGMRES) removes the CFL restriction
but requires solving a linear system each step; at typical mantle convection resolutions the
explicit CFL limit is not prohibitive and the explicit FCT step is simpler, cheaper per step,
and free of solver convergence issues.  Prefer explicit unless very large time steps are
needed.
 
### Explicit FCT (recommended)
 
@code{.cpp}
// --- one-time setup ---
const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords<double>(domain);
const auto coords_radii = grid::shell::subdomain_shell_radii<double>(domain);
 
// Cell centres must be initialised once (fills ghost layers via MPI).
VectorFVVec<double, 3> cell_centers("cell_centers", domain);
fv::hex::initialize_cell_centers(cell_centers, domain, coords_shell, coords_radii);
 
// Pre-allocate scratch buffers (reused every step, no heap allocation in the loop).
FVFCTBuffers<double> bufs(domain);
 
// Time step: use h_min / |u|_max with a safety factor < 1.
const double h   = grid::shell::min_radial_h(domain.domain_info().radii());
const double dt  = 0.25 * h / u_max;  // adjust safety factor to taste
 
// Optional: Dirichlet BCs at CMB and surface.
fv::hex::DirichletBCs<double> bcs{ .T_cmb = 1.0, .T_surface = 0.0,
                                   .apply_cmb = true, .apply_surface = true };
 
// --- time loop ---
for (int ts = 0; ts < n_steps; ++ts)
{
    fv::hex::operators::fct_explicit_step(
        domain, T, u, cell_centers.grid_data(), coords_shell, coords_radii, dt, bufs);
    fv::hex::apply_dirichlet_bcs(T, boundary_mask, bcs, domain);
}
@endcode
 
With a volumetric heat source `q` [T/time]:
@code{.cpp}
    fv::hex::operators::fct_explicit_step(
        domain, T, u, cell_centers.grid_data(), coords_shell, coords_radii, dt, bufs,
        /*diffusivity=*/kappa, /*source=*/q.grid_data());
@endcode
 
### Pure upwind (debugging / baseline only)
 
@code{.cpp}
    // Identical setup as above.  Replace fct_explicit_step with:
    fv::hex::operators::upwind_explicit_step(
        domain, T, u, cell_centers.grid_data(), coords_shell, coords_radii, dt, bufs);
@endcode
 
The result will be noticeably more diffusive than FCT — use it to verify that the geometry
and time-step logic are correct, then switch to `fct_explicit_step`.
 
---
## References
 
- Zalesak, S. T. (1979). Fully multidimensional flux-corrected transport algorithms for fluids.
  Journal of computational physics, 31(3), 335-362.
  https://doi.org/10.1016/0021-9991(79)90051-2
- Boris, J. P., & Book, D. L. (1973). Flux-corrected transport. I. SHASTA, a fluid transport algorithm that works.
  Journal of computational physics, 11(1), 38-69.
- Kuzmin, D., Möller, M., & Turek, S. (2003). Multidimensional FEM‐FCT schemes for arbitrary time stepping.
  International journal for numerical methods in fluids, 42(3), 265-295.
  https://doi.org/10.1002/fld.493
- Kuzmin, D. (2006). On the design of general-purpose flux limiters for finite element schemes. I. Scalar convection.
  Journal of Computational Physics, 219(2), 513-531.
  https://doi.org/10.1016/j.jcp.2006.03.034
- Kuzmin, D. (2009). Explicit and implicit FEM-FCT algorithms with flux linearization.
  Journal of Computational Physics, 228(7), 2517-2534.
  https://doi.org/10.1016/j.jcp.2008.12.011
- Jasak, H. (1996). Error analysis and estimation in the Finite Volume method with applications to fluid flows.