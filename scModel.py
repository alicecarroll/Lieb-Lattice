import numpy as np


class Model:
    def __init__(self, **kwargs):

        self.t = kwargs.get('t', 1.0) # Hopping parameter
        self.mu = kwargs.get('mu', 0.0) # Chemical potential

        # Internal variables are named with _ for convention
        self.Del0 = kwargs.get('Del0', 0.0)
        self.DelA = kwargs.get('DelA', 0.0)
        self.DelB = kwargs.get('DelB', 0.0)

        self.dwave = kwargs.get('dwave', False)

        self.Hk = self.HBdG()

    def HBdG(self):
        """Construct the real space sparse Hamiltonian function."""
        # Create a sparse matrix representation of the Hamiltonian
        H = np.zeros((12, 12), dtype=object)  # Placeholder for Hamiltonian matrix
        #H = self.mu*H
        for index, _ in np.ndenumerate(H): H[index] = lambda kx, ky: 0

        #diagonals
        for i in range(6): H[i,i] = lambda kx, ky: self.mu
        for i in range(6,12): H[i, i] = lambda kx, ky: - self.mu
        #H[np.array]
        #h
        H[np.array([0, 1, 3, 4]), np.array([1, 0, 4, 3])] = lambda kx, ky: -2*self.t * np.cos(kx/2) # Some function of kx, ky
        H[np.array([1, 2, 4, 5]), np.array([2, 1, 5, 4])] = lambda kx, ky: -2*self.t * np.cos(ky/2) 
        #h*
        H[np.array([6, 7, 9, 10]), np.array([7, 6, 10, 9])] = lambda kx, ky: 2*self.t * np.cos(kx/2) # Some function of kx, ky
        H[np.array([7, 8, 10, 11]), np.array([8, 7, 11, 10])] = lambda kx, ky: 2*self.t * np.cos(ky/2)
        
        #Del
        #on site
        H[np.array([0, 1, 2, 9, 10, 11]), np.array([9, 10, 11, 0, 1, 2])] = lambda kx, ky: self.Del0
        H[np.array([3, 4, 5, 6, 7, 8]), np.array([6, 7, 8, 3, 4, 5])] = lambda kx, ky: -self.Del0
        #nn AB
        H[np.array([0, 1, 9, 10]), np.array([10, 9, 1, 0])] = lambda kx, ky: 2*self.DelA*np.cos(kx/2)
        H[np.array([3, 4, 6, 7]), np.array([7, 6, 4, 3])] = lambda kx, ky: -2*self.DelA*np.cos(kx/2)
        #nn BC
        H[np.array([1, 2, 10, 11]), np.array([11, 10, 2, 1])] = lambda kx, ky: 2*self.DelB*np.cos(ky/2)
        H[np.array([4, 5, 7, 8]), np.array([8, 7, 5, 4])] = lambda kx, ky: -2*self.DelB*np.cos(ky/2)

        if self.dwave:
            self.DelB = -self.DelA 
        else:
            self.DelB = self.DelA
        
        def Hk(kx, ky): 
            """Evaluate the Hamiltonian at given kx, ky."""
            hk = np.empty_like(H, dtype=complex)
            #hk = np.empty_like(H, dtype=float)
            eps = 1e-15
            for index in np.ndindex(H.shape): hk[index] = H[index](kx, ky)
            hk[np.abs(hk) < eps] = 0

            return hk
        

        return Hk
    
    def solvHam(self, kx, ky):
        '''
        solves hamiltonian for each pair of coordinates
        '''
        eps = 1e-15
        n = np.shape(kx)[0]
        eig = np.zeros((n, 12))
    
        for i in range(n):
            e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])) #np.linalg.eig(self.Hk(kx[i], ky[i]))[0]
            e[np.abs(e)<eps]=0
            eig[i]=np.sort(e)
            
        return eig.T
    
    def Es(self, k):
        #E=np.array([[],[],[]])
        l = np.shape(k)[0]
        a1 = np.ones(l)
        E=np.empty((12, l))
        eps = 1e-15

        for i in k:
            Erow = self.solvHam(i*a1, k)
            E = np.concatenate((E, Erow), axis=1)
            E[np.abs(E)<eps]=0
        return E
    
    def DOS(self, E, k, sig=5e-2):
        ''' 
        returns an array with the density of state values to each E

        E is an array (against which DOS is plotted)
        En is an array with the E-values for all k grid points of shape (3, ???) because we currently have 3 bands
        sigma is the width of the gauss function
        '''
        def Gauss(E, En, sig):
            return np.exp(-(E-En)**2/(2*sig**2))
            
        En = self.Es(k)
        b, l = np.shape(En)
        arr0 = np.ones(l)
        s1 = 0
        res = np.ones(np.shape(E)[0])
        c=0
        for j in E:
            for i in range(b):
                s1 += np.sum(Gauss(j*arr0,En[i], sig))
            res[c] = s1
            s1=0
            c+=1
        return res
    