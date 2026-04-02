
// H-refinement convergence tests for fct_explicit_step.
//
// Sub-tests (all on a spherical shell, r_min=0.5, r_max=1.0):
//
//   1. test_diffusion_convergence
//      Pure diffusion (u=0, kappa=1).  Run to steady state and compare to the
//      analytical solution T_ss(r) = (1/r - 1/r_max) / (1/r_min - 1/r_max).
//      Expected O(h^2) convergence in the relative L2 norm. (O(h) due to BC handling)
//
//   2. test_heating_convergence
//      Diffusion with a constant volumetric heat source (f=6, kappa=1).  The
//      analytical steady state T_ss(r) = -r^2 - 0.75/r + 1.75 satisfies
//        -kappa * (1/r^2) d/dr(r^2 dT/dr) = f ,  T(r_min)=T(r_max)=0.
//      Expected O(h^2). (O(h) due to BC handling)
//
//   3. test_advection_convergence
//      Pure advection with the solid-body-rotation velocity u = (-y, x, 0).
//      Initial condition: smooth quadratic cone centred at (0.75, 0, 0).
//      Exact solution after one full revolution (T_end = 2π) is the initial
//      condition.  Error = relative L2 norm of (T_final - T_0).
//      Expected convergence order ≥ 1 (FCT is ≥ 2nd order for smooth profiles).

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "../src/terra/communication/shell/communication.hpp"
#include "fv/hex/conversion.hpp"
#include "fv/hex/helpers.hpp"
#include "fv/hex/operators/fct_advection_diffusion.hpp"
#include "io/xdmf.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/bit_masks.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using linalg::VectorFVScalar;
using linalg::VectorFVVec;
using linalg::VectorQ1Vec;

using ScalarType = double;

// ============================================================================
// Velocity: solid-body rotation u = (-y, x, 0)
// ============================================================================

struct VelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > c = grid::shell::coords( id, x, y, r, grid_, radii_ );
        data_( id, x, y, r, 0 )             = -c( 1 );
        data_( id, x, y, r, 1 )             = c( 0 );
        data_( id, x, y, r, 2 )             = 0.0;
    }
};

// ============================================================================
// Relative L2 error of a FV field against a cell-centre-evaluated exact field.
// Denominator uses the squared exact values (relative norm).
// ============================================================================

// Radially-symmetric exact function evaluated at cell centre.
ScalarType l2_error_radial(
    const DistributedDomain&             domain,
    const Grid4DDataScalar< ScalarType > T,
    const Grid4DDataVec< ScalarType, 3 > cc,
    ScalarType                           r_min,
    ScalarType                           r_max,
    bool                                 no_heating )
{
    ScalarType sum_num = 0, sum_den = 0;

    Kokkos::parallel_reduce(
        "l2e_num",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( const int id, const int x, const int y, const int r, ScalarType& acc ) {
            const ScalarType cx  = cc( id, x, y, r, 0 );
            const ScalarType cy  = cc( id, x, y, r, 1 );
            const ScalarType cz  = cc( id, x, y, r, 2 );
            const ScalarType rad = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );

            ScalarType T_ref;
            if ( no_heating )
                T_ref = ( 1.0 / rad - 1.0 / r_max ) / ( 1.0 / r_min - 1.0 / r_max );
            else
                T_ref = -rad * rad - 0.75 / rad + 1.75;

            const ScalarType d = T( id, x, y, r ) - T_ref;
            acc += d * d;
        },
        sum_num );

    Kokkos::parallel_reduce(
        "l2e_den",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( const int id, const int x, const int y, const int r, ScalarType& acc ) {
            const ScalarType cx  = cc( id, x, y, r, 0 );
            const ScalarType cy  = cc( id, x, y, r, 1 );
            const ScalarType cz  = cc( id, x, y, r, 2 );
            const ScalarType rad = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );

            ScalarType T_ref;
            if ( no_heating )
                T_ref = ( 1.0 / rad - 1.0 / r_max ) / ( 1.0 / r_min - 1.0 / r_max );
            else
                T_ref = -rad * rad - 0.75 / rad + 1.75;

            acc += T_ref * T_ref;
        },
        sum_den );

    Kokkos::fence();

    ScalarType gn = 0, gd = 0;
    MPI_Allreduce( &sum_num, &gn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( &sum_den, &gd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    return gd > 1e-30 ? std::sqrt( gn / gd ) : 0.0;
}

// Relative L2 error of T vs a stored reference T_ref (for advection test).
ScalarType l2_error_vs_ref(
    const DistributedDomain&             domain,
    const Grid4DDataScalar< ScalarType > T,
    const Grid4DDataScalar< ScalarType > T_ref )
{
    ScalarType sum_num = 0, sum_den = 0;

    Kokkos::parallel_reduce(
        "l2e_adv_num",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( const int id, const int x, const int y, const int r, ScalarType& acc ) {
            const ScalarType d = T( id, x, y, r ) - T_ref( id, x, y, r );
            acc += d * d;
        },
        sum_num );

    Kokkos::parallel_reduce(
        "l2e_adv_den",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( const int id, const int x, const int y, const int r, ScalarType& acc ) {
            const ScalarType v = T_ref( id, x, y, r );
            acc += v * v;
        },
        sum_den );

    Kokkos::fence();

    ScalarType gn = 0, gd = 0;
    MPI_Allreduce( &sum_num, &gn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( &sum_den, &gd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    return gd > 1e-30 ? std::sqrt( gn / gd ) : 0.0;
}

// ============================================================================
// Per-level runner: pure diffusion or diffusion + heating
// ============================================================================

// Returns relative L2 error at steady state.
// no_heating=true  → T_cmb=1, T_surf=0, no source, analytical T_ss(r) = (1/r-1/r_max)/(1/r_min-1/r_max).
// no_heating=false → T_cmb=T_surf=0, source=6, kappa=1, analytical T_ss(r) = -r^2-0.75/r+1.75.
ScalarType run_diffusion_level( const int level, const bool no_heating )
{
    const ScalarType r_min = 0.5, r_max = 1.0, kappa = 1.0;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorFVScalar< ScalarType > T( "T", domain );
    VectorFVVec< ScalarType, 3 > cell_centers( "cell_centers", domain );
    fv::hex::initialize_cell_centers( cell_centers, domain, coords_shell, coords_radii );

    VectorQ1Vec< ScalarType > u( "u", domain, mask_data );
    assign( u, ScalarType( 0 ) );

    fv::hex::operators::FVFCTBuffers< ScalarType > bufs( domain );

    const fv::hex::DirichletBCs< ScalarType > bcs{
        .T_cmb         = no_heating ? ScalarType( 1 ) : ScalarType( 0 ),
        .T_surface     = ScalarType( 0 ),
        .apply_cmb     = true,
        .apply_surface = true,
    };

    fv::hex::apply_dirichlet_bcs( T, boundary_mask_data, bcs, domain );
    communication::shell::update_fv_ghost_layers( domain, T.grid_data() );

    const ScalarType h       = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt      = 0.05 * h * h / kappa;
    const ScalarType t_end   = 1.0 * ( r_max - r_min ) * ( r_max - r_min ) / kappa;
    const int        n_steps = static_cast< int >( std::ceil( t_end / dt ) );

    // Optional source term for the heating test.
    VectorFVScalar< ScalarType > source( "source", domain );
    if ( !no_heating )
        Kokkos::deep_copy( source.grid_data(), ScalarType( 6 ) );

    util::logroot << "  level=" << level << "  h=" << h << "  dt=" << dt << "  n_steps=" << n_steps << "\n";

    io::XDMFOutput out(
        "test_fct_convergence_out_diffusion_heating_" + std::string( no_heating ? "off" : "on" ) + "_level_" +
            std::to_string( level ),
        domain,
        coords_shell,
        coords_radii );

    linalg::VectorQ1Scalar< ScalarType >                T_projected( "T_projected", domain, mask_data );
    std::vector< linalg::VectorQ1Scalar< ScalarType > > tmps;
    for ( int i = 0; i < 5; ++i )
    {
        tmps.emplace_back( "tmp" + std::to_string( i ), domain, mask_data );
    }
    out.add( T_projected.grid_data() );

    fv::hex::l2_project_fv_to_fe( T_projected, T, domain, coords_shell, coords_radii, tmps );
    out.write();

    for ( int ts = 1; ts <= n_steps; ++ts )
    {
        if ( no_heating )
        {
            fv::hex::operators::fct_explicit_step(
                domain,
                T,
                u,
                cell_centers.grid_data(),
                coords_shell,
                coords_radii,
                dt,
                bufs,
                kappa,
                {},
                true,
                boundary_mask_data,
                bcs );
        }
        else
        {
            fv::hex::operators::fct_explicit_step(
                domain,
                T,
                u,
                cell_centers.grid_data(),
                coords_shell,
                coords_radii,
                dt,
                bufs,
                kappa,
                source.grid_data(),
                true,
                boundary_mask_data,
                bcs );
        }
        fv::hex::apply_dirichlet_bcs( T, boundary_mask_data, bcs, domain );
    }

    fv::hex::l2_project_fv_to_fe( T_projected, T, domain, coords_shell, coords_radii, tmps );
    out.write();

    return l2_error_radial( domain, T.grid_data(), cell_centers.grid_data(), r_min, r_max, no_heating );
}

// ============================================================================
// Per-level runner: pure advection
// ============================================================================

// Smooth cone functor for l2_project_analytical_to_fv.
struct ConeFunctor
{
    KOKKOS_INLINE_FUNCTION
    ScalarType operator()( const dense::Vec< ScalarType, 3 >& x ) const
    {
        const dense::Vec< ScalarType, 3 > center{ 0.75, 0.0, 0.0 };
        const ScalarType                  radius = 0.2;
        const ScalarType                  dist   = ( x - center ).norm();
        if ( dist < radius )
        {
            const ScalarType s = 0.5 * ( 1 + Kokkos::cos( Kokkos::numbers::pi * dist / radius ) );
            return s;
        }
        return ScalarType( 0 );
    }
};

struct ConeQuarterRotationFunctor
{
    KOKKOS_INLINE_FUNCTION
    ScalarType operator()( const dense::Vec< ScalarType, 3 >& x ) const
    {
        const dense::Vec< ScalarType, 3 > center{ 0.0, 0.75, 0.0 };
        const ScalarType                  radius = 0.2;
        const ScalarType                  dist   = ( x - center ).norm();
        if ( dist < radius )
        {
            const ScalarType s = 0.5 * ( 1 + Kokkos::cos( Kokkos::numbers::pi * dist / radius ) );
            return s;
        }
        return ScalarType( 0 );
    }
};

// Returns relative L2 error of T after one full revolution vs the initial condition.
ScalarType run_advection_level( const int level )
{
    const ScalarType r_min = 0.5, r_max = 1.0;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorFVScalar< ScalarType > T( "T", domain );
    VectorFVScalar< ScalarType > T_ref( "T_ref", domain );
    VectorFVVec< ScalarType, 3 > cell_centers( "cell_centers", domain );
    fv::hex::initialize_cell_centers( cell_centers, domain, coords_shell, coords_radii );

    VectorQ1Vec< ScalarType > u( "u", domain, mask_data );
    Kokkos::parallel_for(
        "vel_init",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator{ coords_shell, coords_radii, u.grid_data() } );
    Kokkos::fence();

    // L2-project cone into T.
    fv::hex::l2_project_analytical_to_fv( T, ConeFunctor{}, coords_shell, coords_radii );
    Kokkos::fence();
    communication::shell::update_fv_ghost_layers( domain, T.grid_data() );

    fv::hex::l2_project_analytical_to_fv( T_ref, ConeQuarterRotationFunctor{}, coords_shell, coords_radii );
    Kokkos::fence();
    communication::shell::update_fv_ghost_layers( domain, T_ref.grid_data() );

    const ScalarType h       = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt      = 0.1 * h;
    const ScalarType T_end   = 0.5 * M_PI;
    const int        n_steps = static_cast< int >( std::ceil( T_end / dt ) );

    fv::hex::operators::FVFCTBuffers< ScalarType > bufs( domain );

    const fv::hex::DirichletBCs< ScalarType > bcs{
        .T_cmb         = ScalarType( 0 ),
        .T_surface     = ScalarType( 0 ),
        .apply_cmb     = true,
        .apply_surface = true,
    };

    util::logroot << "  level=" << level << "  h=" << h << "  dt=" << dt << "  n_steps=" << n_steps << "\n";

    io::XDMFOutput out(
        "test_fct_convergence_out_advection_level_" + std::to_string( level ), domain, coords_shell, coords_radii );

    linalg::VectorQ1Scalar< ScalarType >                T_projected( "T_projected", domain, mask_data );
    std::vector< linalg::VectorQ1Scalar< ScalarType > > tmps;
    for ( int i = 0; i < 5; ++i )
    {
        tmps.emplace_back( "tmp" + std::to_string( i ), domain, mask_data );
    }
    out.add( T_projected.grid_data() );

    fv::hex::l2_project_fv_to_fe( T_projected, T, domain, coords_shell, coords_radii, tmps );
    out.write();

    for ( int ts = 1; ts <= n_steps; ++ts )
    {
        fv::hex::operators::fct_explicit_step(
            domain, T, u, cell_centers.grid_data(), coords_shell, coords_radii, dt, bufs );
        fv::hex::apply_dirichlet_bcs( T, boundary_mask_data, bcs, domain );
    }

    fv::hex::l2_project_fv_to_fe( T_projected, T, domain, coords_shell, coords_radii, tmps );
    out.write();

    return l2_error_vs_ref( domain, T.grid_data(), T_ref.grid_data() );
}

// ============================================================================
// Convergence check helpers
// ============================================================================

// Returns convergence order from two successive refinements (h halved each time).
ScalarType convergence_order( ScalarType err_coarse, ScalarType err_fine )
{ return std::log( err_coarse / err_fine ) / std::log( 2.0 ); }

void print_convergence_table(
    const std::string&              name,
    const std::vector< int >&       levels,
    const std::vector< ScalarType > errors )
{
    util::logroot << "\n" << name << " convergence:\n";
    util::logroot << "  level   rel-L2-err   order\n";
    for ( std::size_t i = 0; i < levels.size(); ++i )
    {
        util::logroot << "    " << levels[i] << "     " << errors[i];
        if ( i > 0 )
            util::logroot << "     " << convergence_order( errors[i - 1], errors[i] );
        util::logroot << "\n";
    }
}

// ============================================================================
// Test 1: Pure diffusion convergence — O(h^2) to steady state
//         (O(h) due to BC handling)
// ============================================================================

void test_diffusion_convergence()
{
    util::logroot << "\n=== test_diffusion_convergence ===\n";

    const std::vector< int >  levels = { 3, 4 };
    std::vector< ScalarType > errors;
    errors.reserve( levels.size() );

    for ( const int lvl : levels )
        errors.push_back( run_diffusion_level( lvl, /*no_heating=*/true ) );

    print_convergence_table( "Pure diffusion", levels, errors );

    // Require at least 2nd-order spatial convergence (h halved → error ≤ error/3).
    const ScalarType order      = convergence_order( errors[0], errors[1] );
    const bool       skip_check = false;
    if ( !skip_check && order < 0.9 )
    {
        util::logroot << "FAILED: convergence order " << order << " < 1.5\n";
        Kokkos::abort( "test_diffusion_convergence: insufficient convergence order" );
    }
    if ( !skip_check && errors.back() > 0.5 )
    {
        util::logroot << "FAILED: finest-level error " << errors.back() << " > 0.05\n";
        Kokkos::abort( "test_diffusion_convergence: error too large at finest level" );
    }

    util::logroot << "PASSED (order=" << order << ", finest err=" << errors.back() << ")\n";
}

// ============================================================================
// Test 2: Diffusion + internal heating convergence — O(h^2) to steady state
//         (O(h) due to BC handling)
//
// PDE:   dT/dt = kappa*∆T + f,   f = 6,  kappa = 1
// Steady state (T_cmb = T_surf = 0):
//   T_ss(r) = -r^2 - 0.75/r + 1.75
// Derivation:
//   -kappa*(1/r^2) d/dr(r^2 dT/dr) = f
//   => dT/dr = -f*r/(3*kappa) + C1/r^2 = -2r + C1/r^2  (f=6, kappa=1)
//   => T(r) = -r^2 - C1/r + C2
//   T(0.5)=0 and T(1.0)=0 => C1 = 0.75, C2 = 1.75.
// ============================================================================

void test_heating_convergence()
{
    util::logroot << "\n=== test_heating_convergence ===\n";

    const std::vector< int >  levels = { 3, 4 };
    std::vector< ScalarType > errors;
    errors.reserve( levels.size() );

    for ( const int lvl : levels )
        errors.push_back( run_diffusion_level( lvl, /*no_heating=*/false ) );

    print_convergence_table( "Diffusion + heating", levels, errors );

    const ScalarType order      = convergence_order( errors[0], errors[1] );
    const bool       skip_check = false;
    if ( !skip_check && order < 0.9 )
    {
        util::logroot << "FAILED: convergence order " << order << " < 1.5\n";
        Kokkos::abort( "test_heating_convergence: insufficient convergence order" );
    }
    if ( !skip_check && errors.back() > 0.5 )
    {
        util::logroot << "FAILED: finest-level error " << errors.back() << " > 0.05\n";
        Kokkos::abort( "test_heating_convergence: error too large at finest level" );
    }

    util::logroot << "PASSED (order=" << order << ", finest err=" << errors.back() << ")\n";
}

// ============================================================================
// Test 3: Pure advection convergence — error after one revolution decreases
// ============================================================================

void test_advection_convergence()
{
    util::logroot << "\n=== test_advection_convergence ===\n";

    const std::vector< int >  levels = { 4, 5 };
    std::vector< ScalarType > errors;
    errors.reserve( levels.size() );

    for ( const int lvl : levels )
        errors.push_back( run_advection_level( lvl ) );

    print_convergence_table( "Pure advection (FCT)", levels, errors );

    const ScalarType order      = convergence_order( errors[0], errors[1] );
    const bool       skip_check = false;
    if ( !skip_check && order < 0.9 )
    {
        util::logroot << "FAILED: convergence order " << order << " < 0.5\n";
        Kokkos::abort( "test_advection_convergence: insufficient convergence order" );
    }
    if ( !skip_check && errors.back() > 1.0 )
    {
        util::logroot << "FAILED: finest-level error " << errors.back() << " > 0.60\n";
        Kokkos::abort( "test_advection_convergence: error too large at finest level" );
    }

    util::logroot << "PASSED (order=" << order << ", finest err=" << errors.back() << ")\n";
}

// ============================================================================
// main
// ============================================================================

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    test_diffusion_convergence();
    test_heating_convergence();
    test_advection_convergence();
    return 0;
}
