# TerraNeoX

## TODO

### Big features (definitely required - order not clear)

- [x] advection-diffusion discretization / solver
- [x] advection-diffusion boundary handling
- [ ] advection-diffusion source term handling (must add SUPG term in linear form)
- [x] ~~GMRES~~ BiCGStab(l)
- [x] BDF2 (not yet in a dedicated function, see test_heat_eq)
- [ ] multigrid
- [ ] MPI parallel execution (multi-GPU, multi-node CPU)
- [ ] intra-diamond subdomain communication
- [ ] variable viscosity
- [ ] plates
- [ ] free-slip

### Small features / improvements (not necessarily / maybe required)

- [ ] performance engineering
- [ ] curved wedges
- [ ] particles
- [ ] matrix export / assembly
- [ ] CPU SIMD kernels

### Documentation / cleanup / refactoring

- [ ] Github page
- [x] Doxygen
- [ ] Doxygen page
- [x] move mask stuff that generalizes away from shell namespace