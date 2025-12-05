### Bugs

- [ ] random value interpolation is not correctly working since values at overlapping nodes are not equal without proper 
      communication, just summing them up destroys randomness though

### Big features (definitely required - order not clear)

- [x] advection-diffusion discretization / solver
- [x] advection-diffusion boundary handling
- [ ] advection-diffusion source term handling (must add SUPG term in linear form)
- [x] ~~GMRES~~ BiCGStab(l)
- [x] BDF2 (not yet in a dedicated function, see test_heat_eq)
- [x] multigrid (some notes: a) we need higher operator quad degree than constant (not sure where exactly: diagonal,
  fine-level, everywhere?), b) two-grid V(10, 10) looks ok, otherwise with multigrid we do not get perfectly h-ind. conv
  rates., I
  suppose we need Galerkin coarse grid operators maybe)
- [x] MPI parallel execution (multi-GPU, multi-node CPU)
- [x] intra-diamond subdomain communication (then also test/fix boundary handling in operators/tests - subdomain
  boundaries are sometimes treated as domain boundaries even if they are not)
- [x] variable viscosity
- [x] full stokes
- [ ] plates
- [ ] free-slip
- [ ] compressible Stokes (Fabi)
- [x] FGMRES (BiCGStab works well mostly - but seems to randomly produce NaNs occasionally (not 100% sure if related to
  the solver, but it is very likely))
- [x] Galerkin coarsening
- [ ] iterative refinement
- [x] spherical harmonics helper
- [x] radial profiles loader
- [x] checkpoints (re-use XDMF bin files!)
- [ ] return unmanaged views from SubdomainNeighborhoodSendRecvBuffer that point to contiguous memory per rank, add
  another getter to the pointer of that array and then pass that to MPI_Send/Recv instead
- [x] radial layer data assimilation
- [x] timing(tree?)
- [x] ~~compress VTK(?)~~ XDMF output (binary, actual float data, and ~~/or~~ ~~HDF5/ADIOS2~~ with a single mesh file)
- [x] CLI interface / parameter files
- [x] viscosity-weighted pressure-mass / lumped mass (Fabi)
- [ ] unify/cleanup boundary handling in operators
- [ ] unify/cleanup operator interfaces and documentation
- [x] concept for GCA-capable operators
- [ ] GCA: add matrix class that can be used to assemble GCA operators that are independent of the type of the operator
      (i.e., a sparse matrix like class (storing local matrices or stencils)). I find the construction of the coarse 
      grid operators a little unintuitive and error-prone (why would I need to allocate the coarse grid viscosity 
      functions if I actually only need the fine grid viscosity functions?).
- [ ] Checkpointing: read in checkpoints with different subdomain distribution somehow (e.g., MT256 -> MT512)
      (maybe with a conversion function with different subdomain distribution but the same number of global elements,
      then first split up (using that conversion function) into more subdomains (same global number of elements), 
      and then in a second step interpolate to a finer level     
      

### Small features / improvements (not necessarily / maybe required)

- [ ] cube-like test case (this may require some new FE features)
- [ ] performance engineering
- [x] ~~curved wedges~~ the wedges are curved (unlike I assumed when writing this)
- [ ] particles(?)
- [x] matrix export / assembly (implemented for debugging - not for actual use)
- [ ] CPU SIMD kernels
- [ ] adapt solver ctor like in FGMRES (I think that is the best design)
- [x] power iteration
- [ ] BFBT-preconditioner (Fabi)
- [ ] Chebychev-smoother (Fabi)

### Documentation / cleanup / refactoring

- [ ] Github page
- [ ] Licensing (GPLv3 since we are including HyTeG/TN code)
- [x] Doxygen
- [ ] Doxygen page
- [x] move mask stuff that generalizes away from shell namespace
- [ ] sort out what is spherical shell specific and what is not