import sympy as sp


def inv(mat: sp.MatrixBase) -> sp.Matrix:
    """Optimized implementation of matrix inverse for 2x2, and 3x3 matrices. Use this instead of sympy's mat**-1."""
    if isinstance(mat, sp.MatrixBase):
        rows, cols = mat.shape
        if rows != cols:
            raise Exception("Input matrix must be square.")
        if rows == 2:
            a = mat[0, 0]
            b = mat[0, 1]
            c = mat[1, 0]
            d = mat[1, 1]
            det = a * d - b * c
            invmat = (1 / det) * sp.Matrix([[d, -b], [-c, a]])
            return invmat
        elif rows == 3:
            a = mat[0, 0]
            b = mat[0, 1]
            c = mat[0, 2]
            d = mat[1, 0]
            e = mat[1, 1]
            f = mat[1, 2]
            g = mat[2, 0]
            h = mat[2, 1]
            i = mat[2, 2]
            det = a * e * i + b * f * g + c * d * h - g * e * c - h * f * a - i * d * b
            invmat = (1 / det) * sp.Matrix(
                [
                    [e * i - f * h, c * h - b * i, b * f - c * e],
                    [f * g - d * i, a * i - c * g, c * d - a * f],
                    [d * h - e * g, b * g - a * h, a * e - b * d],
                ]
            )
            return invmat
        else:
            return mat ** -1
    elif isinstance(mat, sp.Expr):
        return 1 / mat
    else:
        raise Exception("Input must be a Sympy matrix or expression.")


def det(mat: sp.Matrix) -> sp.Expr:
    if mat.rows != mat.cols:
        raise Exception("det() of non-square matrix?")

    if mat.rows == 0:
        return mat.one
    elif mat.rows == 1:
        return mat[0, 0]
    elif mat.rows == 2:
        return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    elif mat.rows == 3:
        return (
                mat[0, 0] * mat[1, 1] * mat[2, 2]
                + mat[0, 1] * mat[1, 2] * mat[2, 0]
                + mat[0, 2] * mat[1, 0] * mat[2, 1]
                - mat[0, 2] * mat[1, 1] * mat[2, 0]
                - mat[0, 0] * mat[1, 2] * mat[2, 1]
                - mat[0, 1] * mat[1, 0] * mat[2, 2]
        )
    return mat.det()


# Reference coordinates
xi, eta, zeta = sp.symbols('xi eta zeta')

# Node coordinates (physical space)
x = sp.symbols('x1:9')
y = sp.symbols('y1:9')
z = sp.symbols('z1:9')

# Trilinear shape functions
corner_coords = []

for rr in [-1, 1]:
    for yy in [-1, 1]:
        for xx in [-1, 1]:
            corner_coords.append((xx, yy, rr))

N = [
    (1 + xi * xi_i) * (1 + eta * eta_i) * (1 + zeta * zeta_i) / 8
    for xi_i, eta_i, zeta_i in corner_coords
]

# Mapping
X = sum(Ni * xi_ for Ni, xi_ in zip(N, x))
Y = sum(Ni * yi_ for Ni, yi_ in zip(N, y))
Z = sum(Ni * zi_ for Ni, zi_ in zip(N, z))

# Jacobian (symbolic)
J = sp.Matrix([
    [sp.diff(X, xi), sp.diff(X, eta), sp.diff(X, zeta)],
    [sp.diff(Y, xi), sp.diff(Y, eta), sp.diff(Y, zeta)],
    [sp.diff(Z, xi), sp.diff(Z, eta), sp.diff(Z, zeta)],
])
Jinv = inv(J)
JinvT = Jinv.T

# Shape function gradients on the reference element
grad_N_ref = [
    sp.Matrix([sp.diff(Ni, xi), sp.diff(Ni, eta), sp.diff(Ni, zeta)])
    for Ni in N
]

w = sp.symbols("www")
# Gradient in physical coordinates
grad_N_phys = [w * (JinvT @ g_ref).dot(JinvT @ h_ref) * sp.Abs(det(J)) for g_ref in grad_N_ref for h_ref in grad_N_ref]

# Flatten everything BEFORE substitution
exprs_all = [g for g in grad_N_phys]
exprs_flat = exprs_all  # [item for sublist in exprs_all for item in sublist]

# CSE BEFORE substitution
tmp_syms = sp.symbols('tmp_:1000')
repls, reduced = sp.cse(exprs_flat, symbols=tmp_syms)

# Substitute quadrature point now
qp = sp.symbols('qp0 qp1 qp2')
subs = {xi: qp[0], eta: qp[1], zeta: qp[2]}
repls = [(s, e.subs(subs)) for s, e in repls]
reduced = [e.subs(subs) for e in reduced]

# qs = [-1, 0, 1]
# ws = [1.0 / 3.0, 4.0 / 1.0, 1.0 / 3.0]
#
# quads = []
#
# for q0, w0 in zip(qs, ws):
#     for q1, w1 in zip(qs, ws):
#         for q2, w2 in zip(qs, ws):
#             quads.append((q0, q1, q2, w0 * w1 * w2))
#
# grad_N_phys = [sum([integrand.subs({xi: qp0, eta: qp1, zeta: qp2}) * w
#                     for (qp0, qp1, qp2, w) in quads]) for integrand in grad_N_phys]

# Emit code
print("// --- CSE temporaries ---")
for var, expr in repls:
    print(f"double {sp.ccode(var)} = {sp.ccode(expr)};")

# print("\n// --- Inverse Jacobian ---")
# for i in range(3):
#     for j in range(3):
#         idx = i * 3 + j
#         print(f"Jinv[{i}][{j}] = {sp.ccode(reduced[idx])};")

print("\n// --- Shape function gradients ---")
for a in range(8):
    for i in range(8):
        idx = a * 8 + i
        print(f"dNdx({a}, {i}) = {sp.ccode(reduced[idx])};")

# # CSE BEFORE substitution
# tmp_syms = sp.symbols('tmp_det_:1000')
# repls, reduced = sp.cse(sp.Abs(det(J)), symbols=tmp_syms)
#
# # Substitute quadrature point now
# qp = sp.symbols('qp0 qp1 qp2')
# subs = {xi: qp[0], eta: qp[1], zeta: qp[2]}
# repls = [(s, e.subs(subs)) for s, e in repls]
# reduced = reduced[0].subs(subs)
#
# # Emit code
# print("// --- CSE temporaries ---")
# for var, expr in repls:
#     print(f"double {sp.ccode(var)} = {sp.ccode(expr)};")
#
# print("\n// --- Abs Det ---")
# print(f"abs_det = {sp.ccode(reduced)};")
