import sympy as sp

# ---- Helpers for vector/matrix creation ----
def vec3(x, y, z):
    return sp.Matrix([x, y, z])

def zero3():
    return sp.zeros(3, 1)

def zero3x3():
    return sp.zeros(3, 3)

# ---- shape_rad ----
def shape_rad(node_idx, zeta):
    """
    Equivalent of C++: shape_rad(node_idx, zeta)
    N_rad = [0.5*(1 - zeta), 0.5*(1 + zeta)]
    returns N_rad[node_idx // 3]
    """
    zeta = sp.sympify(zeta)
    N_rad = [sp.Rational(1, 2) * (1 - zeta), sp.Rational(1, 2) * (1 + zeta)]
    return N_rad[int(node_idx) // 3]

def shape_rad_vec(node_idx, xi_eta_zeta):
    """Overload that accepts a 3-vector like in C++: xi_eta_zeta(2) is zeta."""
    # xi_eta_zeta is expected as a 3x1 Matrix or sequence
    zeta = sp.sympify(xi_eta_zeta[2])
    return shape_rad(node_idx, zeta)

# ---- shape_lat ----
def shape_lat(node_idx, xi, eta):
    """
    Equivalent of C++: shape_lat(node_idx, xi, eta)
    N_lat = [1 - xi - eta, xi, eta]
    returns N_lat[node_idx % 3]
    """
    xi = sp.sympify(xi)
    eta = sp.sympify(eta)
    N_lat = [1 - xi - eta, xi, eta]
    return N_lat[int(node_idx) % 3]

def shape_lat_vec(node_idx, xi_eta_zeta):
    """Overload that accepts a 3-vector; uses xi_eta_zeta[0], xi_eta_zeta[1]."""
    xi = sp.sympify(xi_eta_zeta[0])
    eta = sp.sympify(xi_eta_zeta[1])
    return shape_lat(node_idx, xi, eta)

# ---- full shape (tensor product) ----
def shape(node_idx, xi, eta, zeta):
    return shape_lat(node_idx, xi, eta) * shape_rad(node_idx, zeta)

def shape_vec(node_idx, xi_eta_zeta):
    return shape_lat_vec(node_idx, xi_eta_zeta) * shape_rad_vec(node_idx, xi_eta_zeta)

# ---- radial gradient (d/dzeta of radial part) ----
def grad_shape_rad(node_idx):
    """
    Returns derivative d/dzeta of N^rad for the given node index:
    grad_N_rad = [-0.5, 0.5]
    indexed by node_idx // 3
    """
    grad_N_rad = [sp.Rational(-1, 2), sp.Rational(1, 2)]
    return grad_N_rad[int(node_idx) // 3]

# ---- lateral gradients (d/dxi and d/deta of lateral part) ----
def grad_shape_lat_xi(node_idx):
    """
    d/dxi of N^lat: [-1, 1, 0] indexed by node_idx % 3
    """
    grad_N_lat_xi = [sp.Integer(-1), sp.Integer(1), sp.Integer(0)]
    return grad_N_lat_xi[int(node_idx) % 3]

def grad_shape_lat_eta(node_idx):
    """
    d/deta of N^lat: [-1, 0, 1] indexed by node_idx % 3
    """
    grad_N_lat_eta = [sp.Integer(-1), sp.Integer(0), sp.Integer(1)]
    return grad_N_lat_eta[int(node_idx) % 3]

# ---- gradient of full shape (vector) ----
def grad_shape(node_idx, xi, eta, zeta):
    """
    Returns a 3x1 SymPy Matrix:
      [ d/dxi N_j,
        d/deta N_j,
        d/dzeta N_j ]
    following:
      d/dxi N_j  = N^rad_j * d/dxi N^lat_j
      d/deta N_j = N^rad_j * d/deta N^lat_j
      d/dzeta N_j = N^lat_j * d/dzeta N^rad_j
    """
    Nrad = shape_rad(node_idx, zeta)
    Nx = grad_shape_lat_xi(node_idx)
    Ny = grad_shape_lat_eta(node_idx)
    Nz = grad_shape_rad(node_idx)
    Nlat = shape_lat(node_idx, xi, eta)
    return vec3(Nx * Nrad, Ny * Nrad, Nlat * Nz)

def grad_shape_vec(node_idx, xi_eta_zeta):
    xi = xi_eta_zeta[0]
    eta = xi_eta_zeta[1]
    zeta = xi_eta_zeta[2]
    return grad_shape(node_idx, xi, eta, zeta)

# ---- coarse radial shape functions ----
def shape_rad_coarse(coarse_node_idx, fine_radial_wedge_idx, zeta_fine):
    """
    Matches the switch/case logic in C++.
    coarse_node_idx // 3 selects bottom (0) or top (1).
    fine_radial_wedge_idx selects which fine wedge (0 or 1).
    """
    zeta_fine = sp.sympify(zeta_fine)
    case = int(coarse_node_idx) // 3
    fr = int(fine_radial_wedge_idx)
    if case == 0:
        if fr == 0:
            return sp.Rational(1, 4) * (3 - zeta_fine)
        elif fr == 1:
            return sp.Rational(1, 4) * (1 - zeta_fine)
        else:
            return sp.Integer(0)
    elif case == 1:
        if fr == 0:
            return sp.Rational(1, 4) * (1 + zeta_fine)
        elif fr == 1:
            return sp.Rational(1, 4) * (3 + zeta_fine)
        else:
            return sp.Integer(0)
    else:
        return sp.Integer(0)

# ---- coarse lateral shape functions ----
def shape_lat_coarse(coarse_node_idx, fine_lateral_wedge_idx, xi_fine, eta_fine):
    """
    Matches the C++ nested switch on coarse_node_idx % 3 and fine_lateral_wedge_idx.
    """
    xi_fine = sp.sympify(xi_fine)
    eta_fine = sp.sympify(eta_fine)
    case = int(coarse_node_idx) % 3
    fl = int(fine_lateral_wedge_idx)

    if case == 0:
        if fl == 0:
            return -sp.Rational(1, 2) * eta_fine - sp.Rational(1, 2) * xi_fine + 1
        elif fl == 1:
            return -sp.Rational(1, 2) * eta_fine - sp.Rational(1, 2) * xi_fine + sp.Rational(1, 2)
        elif fl == 2:
            return -sp.Rational(1, 2) * eta_fine - sp.Rational(1, 2) * xi_fine + sp.Rational(1, 2)
        elif fl == 3:
            return sp.Rational(1, 2) * eta_fine + sp.Rational(1, 2) * xi_fine
        else:
            return sp.Integer(0)
    elif case == 1:
        if fl == 0:
            return sp.Rational(1, 2) * xi_fine
        elif fl == 1:
            return sp.Rational(1, 2) * xi_fine + sp.Rational(1, 2)
        elif fl == 2:
            return sp.Rational(1, 2) * xi_fine
        elif fl == 3:
            return sp.Rational(1, 2) - sp.Rational(1, 2) * xi_fine
        else:
            return sp.Integer(0)
    elif case == 2:
        if fl == 0:
            return sp.Rational(1, 2) * eta_fine
        elif fl == 1:
            return sp.Rational(1, 2) * eta_fine
        elif fl == 2:
            return sp.Rational(1, 2) * eta_fine + sp.Rational(1, 2)
        elif fl == 3:
            return sp.Rational(1, 2) - sp.Rational(1, 2) * eta_fine
        else:
            return sp.Integer(0)
    else:
        return sp.Integer(0)

# ---- coarse shape (tensor product) ----
def shape_coarse(coarse_node_idx, fine_radial_wedge_idx, fine_lateral_wedge_idx, xi_fine, eta_fine, zeta_fine):
    return shape_lat_coarse(coarse_node_idx, fine_lateral_wedge_idx, xi_fine, eta_fine) * \
           shape_rad_coarse(coarse_node_idx, fine_radial_wedge_idx, zeta_fine)

def shape_coarse_vec(coarse_node_idx, fine_radial_wedge_idx, fine_lateral_wedge_idx, xi_eta_zeta_fine):
    xi = xi_eta_zeta_fine[0]
    eta = xi_eta_zeta_fine[1]
    zeta = xi_eta_zeta_fine[2]
    return shape_coarse(coarse_node_idx, fine_radial_wedge_idx, fine_lateral_wedge_idx, xi, eta, zeta)

# ---- coarse grads (radial) ----
def grad_shape_rad_coarse(coarse_node_idx, fine_radial_wedge_idx):
    case = int(coarse_node_idx) // 3
    fr = int(fine_radial_wedge_idx)
    if case == 0:
        if fr in (0, 1):
            return sp.Rational(-1, 4)
        else:
            return sp.Integer(0)
    elif case == 1:
        if fr in (0, 1):
            return sp.Rational(1, 4)
        else:
            return sp.Integer(0)
    else:
        return sp.Integer(0)

# ---- coarse lateral grads (xi/eta) ----
def grad_shape_lat_coarse_xi(coarse_node_idx, fine_lateral_wedge_idx):
    case = int(coarse_node_idx) % 3
    fl = int(fine_lateral_wedge_idx)
    if case == 0:
        if fl in (0, 1, 2):
            return sp.Rational(-1, 2)
        elif fl == 3:
            return sp.Rational(1, 2)
        else:
            return sp.Integer(0)
    elif case == 1:
        if fl in (0, 1, 2):
            return sp.Rational(1, 2)
        elif fl == 3:
            return sp.Rational(-1, 2)
        else:
            return sp.Integer(0)
    elif case == 2:
        # all zero in the C++ code for xi-derivative
        return sp.Integer(0)
    else:
        return sp.Integer(0)

def grad_shape_lat_coarse_eta(coarse_node_idx, fine_lateral_wedge_idx):
    case = int(coarse_node_idx) % 3
    fl = int(fine_lateral_wedge_idx)
    if case == 0:
        if fl in (0, 1, 2):
            return sp.Rational(-1, 2)
        elif fl == 3:
            return sp.Rational(1, 2)
        else:
            return sp.Integer(0)
    elif case == 1:
        # all zero in the C++ code for eta-derivative
        return sp.Integer(0)
    elif case == 2:
        if fl in (0, 1, 2):
            return sp.Rational(1, 2)
        elif fl == 3:
            return sp.Rational(-1, 2)
        else:
            return sp.Integer(0)
    else:
        return sp.Integer(0)

# ---- gradient of coarse shape (vector) ----
def grad_shape_coarse(node_idx, fine_radial_wedge_idx, fine_lateral_wedge_idx, xi, eta, zeta):
    gx = grad_shape_lat_coarse_xi(node_idx, fine_lateral_wedge_idx) * \
         shape_rad_coarse(node_idx, fine_radial_wedge_idx, zeta)
    gy = grad_shape_lat_coarse_eta(node_idx, fine_lateral_wedge_idx) * \
         shape_rad_coarse(node_idx, fine_radial_wedge_idx, zeta)
    gz = shape_lat_coarse(node_idx, fine_lateral_wedge_idx, xi, eta) * \
         grad_shape_rad_coarse(node_idx, fine_radial_wedge_idx)
    return vec3(gx, gy, gz)

def grad_shape_coarse_vec(node_idx, fine_radial_wedge_idx, fine_lateral_wedge_idx, xi_eta_zeta_fine):
    xi = xi_eta_zeta_fine[0]
    eta = xi_eta_zeta_fine[1]
    zeta = xi_eta_zeta_fine[2]
    return grad_shape_coarse(node_idx, fine_radial_wedge_idx, fine_lateral_wedge_idx, xi, eta, zeta)

# ---- forward maps and their grads ----
def forward_map_rad(r_1, r_2, zeta):
    r_1 = sp.sympify(r_1)
    r_2 = sp.sympify(r_2)
    zeta = sp.sympify(zeta)
    return r_1 + sp.Rational(1, 2) * (r_2 - r_1) * (1 + zeta)

def grad_forward_map_rad(r_1, r_2):
    r_1 = sp.sympify(r_1)
    r_2 = sp.sympify(r_2)
    return sp.Rational(1, 2) * (r_2 - r_1)

def forward_map_lat(p1_phy, p2_phy, p3_phy, xi, eta):
    """
    p*_phy are 3x1 Matrix or sequences.
    Returns a 3x1 Matrix: barycentric mapping (1-xi-eta)*p1 + xi*p2 + eta*p3
    """
    xi = sp.sympify(xi)
    eta = sp.sympify(eta)
    p1 = sp.Matrix(p1_phy)
    p2 = sp.Matrix(p2_phy)
    p3 = sp.Matrix(p3_phy)
    return (1 - xi - eta) * p1 + xi * p2 + eta * p3

def grad_forward_map_lat_xi(p1_phy, p2_phy, p3_phy):
    p1 = sp.Matrix(p1_phy)
    p2 = sp.Matrix(p2_phy)
    # derivative wrt xi is p2 - p1
    return p2 - p1

def grad_forward_map_lat_eta(p1_phy, p2_phy, p3_phy):
    p1 = sp.Matrix(p1_phy)
    p3 = sp.Matrix(p3_phy)
    # derivative wrt eta is p3 - p1
    return p3 - p1

# ---- Jacobians ----
def jac_lat(p1_phy, p2_phy, p3_phy, xi, eta):
    col_0 = grad_forward_map_lat_xi(p1_phy, p2_phy, p3_phy)
    col_1 = grad_forward_map_lat_eta(p1_phy, p2_phy, p3_phy)
    col_2 = forward_map_lat(p1_phy, p2_phy, p3_phy, xi, eta)
    # assemble as columns -> Matrix.hstack
    return sp.Matrix.hstack(sp.Matrix(col_0), sp.Matrix(col_1), sp.Matrix(col_2))

def jac_rad(r_1, r_2, zeta):
    r = forward_map_rad(r_1, r_2, zeta)
    grad_r = grad_forward_map_rad(r_1, r_2)
    # return [r, r, grad_r] as 3x1 vector to be placed on diagonal
    return vec3(r, r, grad_r)

def jac(p1_phy, p2_phy, p3_phy, r_1, r_2, xi, eta, zeta):
    Jlat = jac_lat(p1_phy, p2_phy, p3_phy, xi, eta)
    jr = jac_rad(r_1, r_2, zeta)   # 3x1 vector: [r, r, dr/dzeta]
    # create diagonal matrix from jr
    D = sp.diag(jr[0], jr[1], jr[2])
    return Jlat * D

def jac_from_array(p_phy, r_1, r_2, xi_eta_zeta_fine):
    # p_phy is a sequence/list/tuple of 3 points p_phy[0], p_phy[1], p_phy[2]
    xi = xi_eta_zeta_fine[0]
    eta = xi_eta_zeta_fine[1]
    zeta = xi_eta_zeta_fine[2]
    return jac(p_phy[0], p_phy[1], p_phy[2], r_1, r_2, xi, eta, zeta)

# ---- symmetric_grad ----
def symmetric_grad(J_inv_transposed, quad_point, dof, dim):
    """
    J_inv_transposed: 3x3 Matrix (inverse-transposed Jacobian mapping to physical element)
    quad_point: 3x1 vector of coordinates on reference element
    dof: local index of shape function
    dim: which column (0..2) of the vector-valued shape function we're computing the gradient for
    Returns the symmetric gradient: 0.5*(grad + grad.T), where grad = J_inv_transposed * E
    and E is a 3x3 matrix with grad_shape(dof, quad_point) as column 'dim' and zeros elsewhere.
    """
    Jt = sp.Matrix(J_inv_transposed)
    qp = quad_point
    # compute grad_shape(dof, qp) -> expects components (xi,eta,zeta) from quad_point
    gs = grad_shape_vec(dof, qp)  # 3x1
    # build E: 3x3 zeros with gs placed in column 'dim'
    E = sp.zeros(3, 3)
    # place column
    if not (0 <= int(dim) <= 2):
        raise ValueError("dim must be 0, 1 or 2")
    for i in range(3):
        E[i, int(dim)] = gs[i]
    grad = Jt * E
    return (grad + grad.T) * sp.Rational(1, 2)