# EpsDivDiv Operator Optimization History

Production file: `src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp`
Predecessor files: `epsilon_divdiv_simple.hpp` (V-2), `epsilon_divdiv.hpp` (V-1)
GitHub base: `https://github.com/mantleconvection/terraneo`
Links use `blob/<commit>/...#L<line>` to pin each optimization to the exact commit where it was introduced.

---

## V-2: EpsilonDivDivSimple (aba88f1 — "Add viscosity weighted mass, block triangular preconditioners")

File: `src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp`

The original, textbook-style implementation. **Assembles the full 18x18 local element matrix A, then does A*src.**
- `MDRangePolicy` — 1 thread per hex cell
  [L196](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L196) `parallel_for("matvec", local_domain_md_range_policy_cells(...))`
- **Assembles full `dense::Mat<18,18> A[2]`** per hex cell (both wedges), then multiplies `dst = A * src`
  [L214](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L214) `dense::Mat< ScalarT, 18, 18 > A[num_wedges_per_hex_cell] = {};`
  [L365-377](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L365-L377) `dst[0] = A[0] * src[0]; dst[1] = A[1] * src[1];`
- **6-point Felippa 3x2 quadrature** (3 triangle x 2 line points)
  [L227](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L227) `quad_felippa_3x2_num_quad_points`
  [L247](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L247) `for ( int q = 0; q < num_quad_points; q++ )`
- Nested `dimi x dimj` loops (O(3x3)) assembling A via `double_contract`
  [L239](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L239) `for ( int dimi = 0; dimi < 3; ++dimi )`
  [L241](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L241) `for ( int dimj = 0; dimj < 3; ++dimj )`
  [L282-284](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L282-L284) `A[wedge](...) += w * k_eval * abs_det * (0.5 * sym_grad_i.double_contract(sym_grad_j) - 2/3 * div_i * div_j)`
- Uses **dense `Mat<3,3>` objects** for Jacobian, symmetric gradients — high abstraction, heavy register use
  [L255-256](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L255-L256) `J`, `J.inv_transposed(det)`
  [L268](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L268) `sym_grad_i = (grad_i + grad_i.transposed())`
- **Boundary handling via 18x18 Hadamard-product mask**: fills a `boundary_mask` matrix with 0/1, then `A.hadamard_product(boundary_mask)` to zero out constrained coupling entries
  [L300-301](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L300-L301) `boundary_mask.fill(1.0);`
  [L337](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L337) `A[wedge].hadamard_product( boundary_mask );`
- Scatter via `atomically_add_local_wedge_vector_coefficients` helper
  [L387-392](https://github.com/mantleconvection/terraneo/blob/aba88f1/src/terra/fe/wedge/operators/shell/epsilon_divdiv_simple.hpp#L387-L392)

## V-1: EpsilonDivDiv — refactored (bdb954c — "Use refactored eps + divdiv in Stokes test")

File: `src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp`

Evolved through several commits (10eb34b → bec4d13 → 63c8f46 → 0b3f724 → bdb954c). Key changes vs Simple:
- **Fused local matvec** — no longer assembles the full 18x18 matrix, instead fuses assembly with application
  [L382-396](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L382-L396) `assemble_trial_test_vecs(...)` then `fused_local_mv(...)`
- **`assemble_trial_test_vecs()`**: computes symmetric gradient vectors for trial and test
  [L227](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L227) `void assemble_trial_test_vecs(...)`
- Gather/scatter via hex-node indexing (`hex_offset_x/y/r[8]`) with `atomic_add` per node
  [L348-350](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L348-L350) `hex_offset_x/y/r` arrays
  [L411](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L411) `Kokkos::atomic_add(&dst_(...))`
- Still uses **6-point Felippa 3x2 quadrature**, `MDRangePolicy`, O(3x3) dimi/dimj loops
  [L198](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L198) `parallel_for("matvec", local_domain_md_range_policy_cells(...))`
  [L354-356](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L354-L356) `for dimi x for dimj`
  [L375](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L375) `for ( int q = 0; q < num_quad_points; q++ )`
- **Supports GCA** (Galerkin Coarse Approximation) via `assemble_local_matrix()` for stored matrices
  [L269](https://github.com/mantleconvection/terraneo/blob/bdb954c/src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp#L269) stored-matrix path in `operator()`

### Key intermediate refactors (all in `epsilon_divdiv.hpp`):
- (bec4d13) Fuse separate `dirichlet_cmb`, `dirichlet_surface`, `neumann`, `diagonal` functions into one
- (63c8f46) Fuse diagonal kernel into single kernel — no duplicate code paths
- (0b3f724) Add on-the-fly single-element assembly + trial/test gradient vector assembly

---

## V01 Initial (e7ae1b3 — "Generate epsdivdiv kernel + benchmark")
- `MDRangePolicy` — 1 thread per hex cell
  [L196](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L196) `parallel_for("matvec", local_domain_md_range_policy_cells(...))`
- Nested `for dimi x for dimj` — O(3x3)=9 dim-pair passes
  [L403](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L403) `for ( dimi = 0; dimi < 3 ...)`
  [L405](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L405) `for ( dimj = 0; dimj < 3 ...)`
- Full 6-point quadrature per wedge
  [L316](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L316) `for ( q = 0; q < 6; q += 1 )`
- `dst_array[3][2][6]` in registers, 24 `atomic_add`s
  [L311](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L311) `double dst_array[3][2][6] = { 0 };`
  [L513](https://github.com/mantleconvection/terraneo/blob/e7ae1b3/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L513) atomic_add scatter block

## V02 Split dimi/dimj (c9c1e21 — "Split dimi/dimj loop -> 2x3 complexity instead of 3x3")
- **Split nested loops**: gather over dimj first, then scatter over dimi → O(3+3)=6 passes
  [L396](https://github.com/mantleconvection/terraneo/blob/c9c1e21/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L396) `for (dimj = 0; dimj < 3 ...)` — trial/gather pass
  [L421](https://github.com/mantleconvection/terraneo/blob/c9c1e21/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L421) `for (dimi = 0; dimi < 3 ...)` — test/scatter pass
- Separate diagonal/boundary loop
  [L440](https://github.com/mantleconvection/terraneo/blob/c9c1e21/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L440) `for (dim_diagBC = 0; dim_diagBC < 3 ...)`

## V03 Teams + Precomputation (b875f4c — "use teams + precomputation in epsdivdiv")
- **Switch from MDRangePolicy to TeamPolicy** (1 team per xy-column, threads along r)
  [L240-241](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L240-L241) `TeamPolicy<>( blocks_, block_size_ ).set_scratch_size(...)`
- `block_size_ = min(128, hex_rad_)`, blocks = subdomains * hex_lat^2 * blocks_per_column
  [L110](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L110) `block_size_ = std::min( 128, threads_per_column );`
  [L112](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L112) `blocks_ = local_subdomains_ * hex_lat_ * hex_lat_ * blocks_per_column_;`
- **Surface coords into shared memory** via team_rank==0 guard + barrier
  [L490](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L490) shmem allocation
  [L493](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L493) `if ( team.team_rank() == 0 )` loads coords
  [L511](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L511) `team.team_barrier()`
- **Collapsed to single quadrature point** (qp=1/3,1/3,0; qw=1)
  [L471-474](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L471-L474)
- Mask-based `has_flag()` replaces raw index comparisons
  [L142-149](https://github.com/mantleconvection/terraneo/blob/b875f4c/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L142-L149)

## V04 Shared Mem Coords via `Kokkos::single` (fe1c12e — "improved 1thread=1cell: more scopes, qp collapsed, shared mem for coords")
- **Full shmem for `wedge_surf_phy_coords[2][3][3]`** loaded via `Kokkos::single(PerTeam)`
  [L734](https://github.com/mantleconvection/terraneo/blob/fe1c12e/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L734) `Scratch3D wedge_surf_phy_coords( shmem, 2, 3, 3 );`
  [L736](https://github.com/mantleconvection/terraneo/blob/fe1c12e/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L736) `Kokkos::single( Kokkos::PerTeam( team ), [&]() { ... });`
- Introduces `column_grad_to_sym()` helper — precomputes symmetric gradient via `switch(dim)`
  [L399](https://github.com/mantleconvection/terraneo/blob/fe1c12e/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L399) `void column_grad_to_sym(...)`

## V05 Shared Mem for src + k (70bacff — "Update with shared mem for src and k dofs per team")
- **Source vector and coefficient loaded into team scratch**
  [L316-319](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L316-L319) `team_shmem_size()` — coords+src+k
- `src_sh(nlev, 4, 3)` and `k_sh(nlev, 4)` scratch views
  [L398](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L398) `Scratch3DLevels src_sh( shmem, nlev, 4, 3 );`
  [L402](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L402) `Scratch2DLevels k_sh( shmem, nlev, 4 );`
- Cooperative load: each thread loads its level, last thread loads extra (for r+1)
  [L460-481](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L460-L481) `auto load_level = [&](int level) { ... }`
  [L492](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L492) `team.team_barrier();`
- `WEDGE_TO_UNIQUE[2][6]` mapping → `dst8[3][8]` (8 unique hex nodes, deduplicating shared wedge nodes)
  [L355-358](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L355-L358) `WEDGE_TO_UNIQUE` array
  [L513](https://github.com/mantleconvection/terraneo/blob/70bacff/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L513) `double dst8[3][8] = { 0.0 };`

## V06 XY Tiling (7f053dd — "Add xy tiling to eps + divdiv")
- **3D (x,y,r) tiling**: `lat_tile=4, r_tile=8` → `team_size=128`
  [L76-81](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L76-L81) `lat_tile_, r_tile_, team_size_`
  [L125-126](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L125-L126) `lat_tile_ = 4; r_tile_ = 8;`
- Thread mapping: `tx = tid%lat_tile, ty = (tid/lat_tile)%lat_tile, tr = tid/(lat_tile*lat_tile)`
  [L364-366](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L364-L366)
- Full tile slab loaded cooperatively via `TeamThreadRange`
  [L402](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L402) coords load
  [L418](https://github.com/mantleconvection/terraneo/blob/7f053dd/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L418) src/k radial load

## V07 Split Paths (95fbf31 — "split freeslip/store path and generated path") — ~47x speedup
- **Host-side kernel dispatch**: `use_slow_path_` decides at launch, no runtime branch in device kernel
  [L118](https://github.com/mantleconvection/terraneo/blob/95fbf31/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L118) `bool use_slow_path_ = false;`
  [L138](https://github.com/mantleconvection/terraneo/blob/95fbf31/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L138) `use_slow_path_ = has_freeslip_bc || has_stored_matrices;`
- Slow path: stored local matrices or freeslip; Fast path: Dirichlet/Neumann matrix-free
  [L325-342](https://github.com/mantleconvection/terraneo/blob/95fbf31/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L325) `if (use_slow_path_) → operator_slow_kernel; else → operator_fast_kernel`

## V08 Scalar Coalesced (03f228d — "separate vec funcs into 3 scalar funcs for coalesced accesses")
- **Thread index reordering**: r is now fastest-varying for memory coalescing
  [L485-487](https://github.com/mantleconvection/terraneo/blob/03f228d/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L485-L487) `tr = tid % r_tile_; tx = (tid/r_tile_) % lat_tile_; ty = ...`
- `r_tile=16` (doubled), `team_size = 4*4*16 = 256`
  [L192](https://github.com/mantleconvection/terraneo/blob/03f228d/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L192) `r_tile_ = 16;`
- **Three-way `KernelPath` enum**: `Slow / FastDirichletNeumann / FastFreeslip`
  [L121-127](https://github.com/mantleconvection/terraneo/blob/03f228d/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L121-L127) `enum class KernelPath { ... }`
- Per-wedge scatter `dst_w[3][6]` (shorter register lifetime than `dst8[3][8]`)
  [L999](https://github.com/mantleconvection/terraneo/blob/03f228d/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L999) `double dst_w[3][6] = { 0.0 };`

## V09 Separate Scatter (d208988 — "separate scatter to 7.6 gdofs") — **7.6 Gdof/s**
- **Split gather/scatter into separate Jacobian scopes**: invJ freed between phases
  [L1005](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1005) `// Phase 1: Jacobian + Gather (gu tensor)`
  [L1082](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1082) `// Phase 2: Recompute Jacobian + Scatter`
- **No register buffer for dst** — inline atomics during scatter (trades 2x J compute for register relief)
  [L1134-1145](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1134-L1145) 3 inline `atomic_add` per node
- **`LaunchBounds<128, 5>`** on fast DN path
  [L377](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L377) `TeamPolicy< LaunchBounds< 128, 5 > >`
- **`template <bool Diagonal>`** specialization eliminates runtime branch
  [L847](https://github.com/mantleconvection/terraneo/blob/d208988/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L847) `template < bool Diagonal > void operator_fast_dirichlet_neumann_path(...)`

## V10 Sequential r_passes (c20ae75 — "use sequential r_passes to reduce register pressure -> 7.8 Gdofs") — **7.8 Gdof/s**
- **`r_passes=2`**: each thread processes 2 radial cells sequentially; shmem loaded once for both
  [L107](https://github.com/mantleconvection/terraneo/blob/c20ae75/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L107) `int r_passes_;`
  [L195](https://github.com/mantleconvection/terraneo/blob/c20ae75/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L195) `r_passes_ = 2;`
- `r_tile_block = r_tile * r_passes = 16` — amortizes shmem loads
  [L196](https://github.com/mantleconvection/terraneo/blob/c20ae75/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L196) `r_tile_block_ = r_tile_ * r_passes_;`
- Sequential pass loop
  [L967](https://github.com/mantleconvection/terraneo/blob/c20ae75/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L967) `for ( int pass = 0; pass < r_passes_; ++pass )`

## Current (f6ae663 — "restructure jacobian comp to save registers -> down to 80 per thread") — ~7.8 Gdof/s
- **Cross-product Jacobian inverse**: L1, L2, Rm → A=L2xRm, B=RmxL1, C=L1xL2 (avoids full 9-entry J matrix)
  [L1020-1028](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1020-L1028) L1, L2, Rm vectors
  [L1030-1038](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1030-L1038) cross products A, B, C
  [L1039](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1039) `lat_det = L1 . A`
  [L1045-1047](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1045-L1047) invJ rows from scaled A, B, C
  [L1048](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1048) `// L1,L2,Rm,A,B,C,...freed`
- **No J recomputation in scatter** (unlike V09): invJ stays live, div_u computed from gu trace
  [L1081](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1081) `// Scatter (reuses i00..i22 — no J recomputation)`
- **FMA-friendly scatter**: precomputes `kwJ2 = 2*kwJ` and `kdiv = kwJ * -2/3 * div_u`
  [L1090-1091](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1090-L1091) `kwJ2`, `kdiv` prefactors
  [L1104-1112](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L1104-L1112) scatter: `(g*gu)*kwJ2 + kdiv*g` per component
- **`LaunchBounds<128, 6>`** (up from 5 → higher occupancy target)
  [L377](https://github.com/mantleconvection/terraneo/blob/f6ae663/src/terra/fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp#L377) `TeamPolicy< LaunchBounds< 128, 6 > >`
- 80 registers/thread (down from ~96+), but no additional throughput gain over V10

## Summary Table

| Version | Commit | File | Key Innovation | Perf |
|---------|--------|------|----------------|------|
| V-2 | [aba88f1](https://github.com/mantleconvection/terraneo/commit/aba88f1) | `_simple.hpp` | Textbook: assemble 18x18 A, then A*src; Hadamard BC mask | — |
| V-1 | [bdb954c](https://github.com/mantleconvection/terraneo/commit/bdb954c) | `.hpp` | Fused local matvec (no full A), trial/test vecs, GCA support | — |
| V01 | [e7ae1b3](https://github.com/mantleconvection/terraneo/commit/e7ae1b3) | `_kerngen.hpp` | Code-gen baseline: MDRange, 6-qp, O(3x3) dimi/dimj, scalar arith | — |
| V02 | [c9c1e21](https://github.com/mantleconvection/terraneo/commit/c9c1e21) | `_kerngen.hpp` | Split dimi/dimj → O(3+3) | — |
| V03 | [b875f4c](https://github.com/mantleconvection/terraneo/commit/b875f4c) | `_kerngen.hpp` | TeamPolicy (r-column), 1-qp collapse, shmem coords | — |
| V04 | [fe1c12e](https://github.com/mantleconvection/terraneo/commit/fe1c12e) | `_kerngen.hpp` | `Kokkos::single(PerTeam)` coord load, `column_grad_to_sym` | — |
| V05 | [70bacff](https://github.com/mantleconvection/terraneo/commit/70bacff) | `_kerngen.hpp` | Shmem src+k, `WEDGE_TO_UNIQUE` → `dst8[3][8]` | — |
| V06 | [7f053dd](https://github.com/mantleconvection/terraneo/commit/7f053dd) | `_kerngen.hpp` | 3D xy+r tiling (4x4x8), cooperative `TeamThreadRange` load | — |
| V07 | [95fbf31](https://github.com/mantleconvection/terraneo/commit/95fbf31) | `_kerngen.hpp` | Host-side fast/slow path dispatch | ~47x fast vs slow |
| V08 | [03f228d](https://github.com/mantleconvection/terraneo/commit/03f228d) | `_kerngen.hpp` | Coalesced r-first mapping, 3-way path, per-wedge scatter | — |
| V09 | [d208988](https://github.com/mantleconvection/terraneo/commit/d208988) | `_kerngen.hpp` | Separate gather/scatter (2x J), LB<128,5>, `template<Diagonal>` | 7.6 Gdof/s |
| V10 | [c20ae75](https://github.com/mantleconvection/terraneo/commit/c20ae75) | `_kerngen.hpp` | Sequential r_passes=2, amortized shmem | 7.8 Gdof/s |
| Cur | [f6ae663](https://github.com/mantleconvection/terraneo/commit/f6ae663) | `_kerngen.hpp` | Cross-product J (80 regs), FMA scatter, LB<128,6> | ~7.8 Gdof/s |
