from sympy.codegen.ast import (
    Token, CodeBlock
)
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy import symbols, IndexedBase, Idx
from sympy.utilities.codegen import codegen
from sympy.tensor.indexed import IndexedBase
from sympy import ccode, Symbol
from sympy.codegen.ast import (
    Assignment,
    For,
    CodeBlock,
    Variable,
    Declaration,
    Pointer,
    AugmentedAssignment,
    aug_assign,
    integer,
    Comment,
    String,
    Element,
)
import sympy as sp
import os, sys
from integrands import *
from kernel_helpers import *
from ast_extensions import *
from sympy.codegen.ast import (
    Assignment,
    For,
    CodeBlock,
    Variable,
    Declaration,
    Pointer,
    AugmentedAssignment,
    aug_assign,
    integer,
    Comment,
    String,
    Element,
)
from sympy import symbols, IndexedBase, Idx
from sympy.tensor.indexed import IndexedBase


class Conditional(Token):
    __slots__ = _fields = ('condition', 'body')
    _construct_target = staticmethod(_sympify)

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def _construct_condition(cls, itr):
        return itr
    


def flattened_cse(M, prefix):

    exprs = M.tolist()
    exprs_flat = [item for row in exprs for item in row]

    return sp.cse(exprs=exprs_flat, symbols=numbered_symbols(prefix=prefix, real=True))


def create_3x3_mat_symbols(prefix):
    mat_symbols = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            mat_symbols[i, j] = sp.symbols(f"{prefix}_{i}_{j}", real=True)
    return mat_symbols


def create_3x3_mat_from_col_vec(prefix, col, col_vec):
    mat_symbols = create_3x3_mat_symbols(prefix)
    mat_decls = []
    for i in range(3):
        for j in range(3):
            mat_decls.append(
                (
                    mat_symbols[i, j],
                    col_vec[i]
                    * sp.Piecewise(
                        (1, sp.Eq(col, j)),
                        (0, True),
                    ),
                )
            )
    return mat_symbols, mat_decls


def replace_matrix(matrix, prefix):
    replace_assignments = []
    replaced_matrix = []
    row, col = matrix.shape
    for i in range(row):
        for j in range(col):
            tmp_ij = sp.symbols(f"{prefix}_{i}_{j}", real=True)
            replaced_matrix.append(tmp_ij)
            replace_assignments.append((tmp_ij, matrix[i, j]))
    replaced_matrix = sp.Matrix(row, col, replaced_matrix)
    return replace_assignments, replaced_matrix


def make_ast_from_exprs(exprs):
    ast = []
    for expr in exprs:
        if isinstance(expr, str):
            ast.append(String(expr))
        else:
            lhs, rhs = expr
            if isinstance(lhs, Symbol):
                if lhs in rhs.free_symbols:
                    ast.append(Assignment(lhs, rhs))
                else:
                    ast.append(Variable.deduced(lhs).as_Declaration(value=rhs))
            elif isinstance(lhs, Indexed):
                ast.append(Assignment(lhs, rhs))
            else:
                exit(f"Unexpected expression: {expr}")
    return ast


from sympy.printing.c import C89CodePrinter

class TerraNeoASTPrinter(C89CodePrinter):
    def _print_Conditional(self, expr):
        return f"if({self._print(expr.condition)}) {{\n {self._print(expr.body)}\n }}"
   
    def _print_For(self, expr):
        if isinstance(expr.iterable, Range):
            start = expr.iterable.start
            stop = expr.iterable.stop
            step = expr.iterable.step
        elif isinstance(expr.iterable, list):
            start, stop, step = expr.iterable
        elif isinstance(expr.iterable, Tuple):
            start, stop, step = expr.iterable
        else:
            raise NotImplementedError("Unknown type of iterable: %s" % type(expr.iterable))
        
        return f"for ({self._print(expr.target)} = {start};  {self._print(expr.target)} < {stop}; {self._print(expr.target)} += {step}) {{\n{self._print(expr.body)}\n}}"


    def _print_Indexed(self, expr):
        # calculate index for 1d array
        offset = getattr(expr.base, 'offset', S.Zero)
        strides = getattr(expr.base, 'strides', None)
        indices = list(expr.indices)
        print(indices)
        access = "%s" % self._print(expr.base.label)
        for idx in indices:
            access += f"[{ self._print(idx)}]"
        return access
    
def terraneo_ccode(stmts):
    return TerraNeoASTPrinter({"contract" : False}).doprint(stmts)