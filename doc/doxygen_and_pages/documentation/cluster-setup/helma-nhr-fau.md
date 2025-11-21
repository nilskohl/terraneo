# Helma @ NHR@FAU {#helma-nhr-fau}

Access form: https://hpc.fau.de/access-to-helma/

SSH Connect and other info on helma: https://doc.nhr.fau.de/clusters/helma/

```
$ module load openmpi/5.0.5-nvhpc24.11-cuda
$ module load cmake
$ mkdir terraneo-build
$ cd terraneo-build

# give parallel backend and architecture via cmake (Kokkos may be unable to autodetect the arch)
$ cmake ../terraneo -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON

# Build tests
$ cd tests
$ make -j 16
```