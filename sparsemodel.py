import numpy as np
import scipy.sparse as sp

class SparseModel:
    def __init__(self, **kwargs):

        self.t = kwargs.get('t', 1.0) # Hopping parameter
        self.mu = kwargs.get('mu', 0.0) # Chemical potential

        self.nx = kwargs.get('nx', 10) # system size in x-direction
        self.ny = kwargs.get('ny', 10) # system size in y-direction

        self.bcx = kwargs.get('bcx', False)
        self.bcy = kwargs.get('bcy', False)

        # Internal variables are named with _ for convention
        self._sublattice_dim = 3 # Number of sublattices
        self._siteList = [(m1, m2, s) for m1 in range(self.nx) for m2 in range(self.ny) for s in range(self._sublattice_dim)]
        self._indexList = {site: n for n, site in enumerate(self._siteList)}  # Map sites to indices

        self.Hk = self.sparseHamiltonian()

    def sparseHamiltonian(self):
        """Construct the real space sparse Hamiltonian function."""
        # Create a sparse matrix representation of the Hamiltonian
        
        dim = self.nx * self.ny * self._sublattice_dim # Dimension of the Hamiltonian matrix
        sH = sp.csr_matrix((dim, dim))
        sH += sp.diags([-self.mu], 0, shape=sH.shape, format='csr')

        sH_lil = sH.tolil()        
         
        for idxFrom, siteFrom in enumerate(self._siteList):
            m1, m2, s = siteFrom  # Unpack site coordinates and sublattice index
            # To do: Add hopping terms to the Hamiltonian
            match s%self._sublattice_dim:
                case 0: #A-site
                    siteToList = [(m1,m2,1), (m1+1,m2,1)]
                    for siteTo in siteToList:
                        if self.bcx == False and siteTo[0] >= self.nx:
                            continue
                        elif siteTo[0] >= self.nx:
                            siteTo = ((m1+1)%self.nx , m2,1)
                        idxTo = self._indexList[siteTo]
                        sH_lil[idxFrom, idxTo] += -self.t        
                case 1: #B-site
                    siteToList = [(m1,m2,0), (m1,m2,2), (m1-1,m2,0), (m1,m2-1,2)]
                    for siteTo in siteToList:
                        if self.bcx == False and siteTo[0]<0:
                            continue
                        elif siteTo[0]<0:
                            siteTo = ((m1-1)%self.nx , m2,0)
                        elif self.bcy == False and siteTo[1]<0:
                            continue
                        elif siteTo[1]<0:
                            siteTo = (m1, (m2-1)%self.ny,2)
                        
                        idxTo = self._indexList[siteTo]
                        sH_lil[idxFrom, idxTo] += -self.t
                case 2: #C-site
                    siteToList = [(m1,m2,1), (m1,m2+1,1)]
                    for siteTo in siteToList:
                        if self.bcy == False and siteTo[1]>=self.ny:
                            continue
                        elif siteTo[1]>=self.ny:
                            siteTo = (m1, (m2+1)%self.ny ,1)
                        
                        idxTo = self._indexList[siteTo]
                        sH_lil[idxFrom, idxTo] += -self.t

        sH = sH_lil.tocsr()

        return sH
    
    def Es(self):
        E = np.linalg.eigvalsh(self.Hk.toarray())
        return E

    def DOS(self, E, sig=5e-2):
        ''' 
        returns an array with the density of state values to each E

        E is an array (against which DOS is plotted)
        En is an array with the E-values for all k grid points of shape (3, ???) because we currently have 3 bands
        sigma is the width of the gauss function
        '''
        def Gauss(E, En, sig):
            return np.exp(-(E-En)**2/(2*sig**2))

        En = self.Es()
        l = np.shape(En)[0]
        s1 = 0
        res = np.ones(np.shape(E)[0])
        for c,j in enumerate(E):
            s1 += np.sum(Gauss(j*np.ones(l), En, sig))
            res[c] = s1
            s1=0
        return res
