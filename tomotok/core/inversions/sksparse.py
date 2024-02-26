from scipy import sparse

from .bob import Bob
from .mfr import Mfr
from .fixed import Fixt


class Cholmod(object):
    """
    Uses sparse cholesky decomposition for solving the parameter optimisation task in MFI loop.
    Requires scikit sparse to be installed in order to initialize properly.

    Uses sksparse.cholmod.cholesky to solve the regularised task in parameter optimisation.
    """
    def __init__(self):
        """
        Executes standard initialization and imports sksparse.cholmod
        """
        from sksparse.cholmod import cholesky
        self.cholesky = cholesky
        super().__init__()

    def invert(self, a : sparse.spmatrix, b):
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using sksparse.cholmod.cholesky

        Parameters
        ----------
        a : scipy.sparse.csr_matrix
            square and positive definite matrix
        b : array_like
            right hand side vector
        """
        factor = self.cholesky(a)
        return factor(b)


class CholmodMfr(Cholmod, Mfr):
    pass


class CholmodFixt(Cholmod, Fixt):
    pass


class CholmodBob(Bob):
    """
    Decomposition optimized for sparse matrices using Cholesky decomposition

    Uses sksparse.cholmod.cholesky to solve the decomposition
    Requires positive definite symmetrized geometry matrix in reconstruction plane basis.
    """
    def __init__(self):
        """
        Executes standard initialization and imports sksparse.cholmod
        """
        from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError
        self._cholesky = cholesky
        self._NotPositiveDefiniteError = CholmodNotPositiveDefiniteError
        super().__init__()

    def compute_coefficients(self, a: sparse.spmatrix, solver_kw: dict = None) -> sparse.csr_matrix:
        try:
            factor = self._cholesky(a, **solver_kw)
        except self._NotPositiveDefiniteError:
            raise ValueError('Symmetrized matrix was not positive definite. Try increasing regularisation factor.')
        b = sparse.eye(a.shape[0])
        c = factor(b)
        return c

