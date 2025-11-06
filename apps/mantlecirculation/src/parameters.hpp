

#include <string>
#include <variant>

namespace terra::mantlecirculation {

struct MeshParameters
{
    int refinement_level_mesh_min   = 0;
    int refinement_level_mesh_max   = 4;
    int refinement_level_subdomains = 0;

    double radius_min = 0.5;
    double radius_max = 1.0;
};

struct IsoViscParameters
{};

struct VarViscParameters
{};

struct PhysicsParameters
{
    double diffusivity     = 1.0;
    double rayleigh_number = 1e5;
};

struct StokesSolverParameters
{
    int    krylov_restart            = 20;
    int    krylov_max_iterations     = 10;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;

    int viscous_pc_num_vcycles                 = 2;
    int viscous_pc_num_smoothing_steps_prepost = 2;
};

struct EnergySolverParameters
{
    int    krylov_restart            = 5;
    int    krylov_max_iterations     = 100;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;
};

struct TimeSteppingParameters
{
    double pseudo_cfl = 0.5;
    double t_end      = 10.0;

    int max_timesteps = 1000;

    int energy_substeps = 1;
};

struct IOParameters
{
    std::string outdir    = "output";
    bool        overwrite = false;

    std::string xdmf_dir            = "xdmf";
    std::string radial_profiles_dir = "radial_profiles";
    std::string timer_trees_dir     = "timer_trees";
};

struct Parameters
{
    MeshParameters                                                       mesh_parameters;
    std::variant< std::monostate, IsoViscParameters, VarViscParameters > viscosity_parameters;
    StokesSolverParameters                                               stokes_solver_parameters;
    EnergySolverParameters                                               energy_solver_parameters;
    PhysicsParameters                                                    physics_parameters;
    TimeSteppingParameters                                               time_stepping_parameters;
    IOParameters                                                         io_parameters;

    [[nodiscard]] bool is_isoviscous() const
    {
        return std::holds_alternative< IsoViscParameters >( viscosity_parameters );
    }

    [[nodiscard]] bool is_varviscous() const
    {
        return std::holds_alternative< VarViscParameters >( viscosity_parameters );
    }
};

}; // namespace terra::mantlecirculation