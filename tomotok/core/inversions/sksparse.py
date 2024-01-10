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
        super().__init__()
        from sksparse.cholmod import cholesky
        self.cholesky = cholesky

    def invert(self, a, b):
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
