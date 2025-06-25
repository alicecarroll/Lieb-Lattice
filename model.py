import numpy as np

class Model:
    def __init__(self, **kwargs):

        self.t = kwargs.get('t', 1.0) # Hopping parameter
        self.mu = kwargs.get('mu', 0.0) # Chemical potential

        self.Hk = self.Hamiltonian()
        

    def Hamiltonian(self):
        """Construct the k-space Hamiltonian function."""
        
        H = np.zeros((3, 3), dtype=object)  # Placeholder for Hamiltonian matrix

        # Initialize the Hamiltonian matrix with zero functions
        for index, _ in np.ndenumerate(H): H[index] = lambda kx, ky: 0

        # Modify the Hamiltonian matrix with specific functions
        # Example: Dirac Hamiltonian for a 2D system
        H[0,1] = lambda kx, ky: -2*self.t * np.cos(kx/2) # Some function of kx, ky
        H[1,0] = lambda kx, ky: -2*self.t * np.cos(kx/2)
        H[1,2] = lambda kx, ky: -2*self.t * np.cos(ky/2)
        H[2,1] = lambda kx, ky: -2*self.t * np.cos(ky/2)
        
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
        eig = np.zeros((n, 3))
    
        for i in range(n):
            e = np.linalg.eig(self.Hk(kx[i], ky[i]))[0]
            e[np.abs(e)<eps]=0
            eig[i]=np.sort(e)
            
        return eig.T
    
    def Es(self, k):
        E=np.array([[],[],[]])
        l = np.shape(k)[0]
        for i in k:
            Erow = self.solvHam(i*np.ones(l), k)
            E = np.concatenate((E, Erow), axis=1)
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
        s1 = 0
        res = np.ones(np.shape(E)[0])
        c=0
        for j in E:
            for i in range(b):
                s1 += np.sum(Gauss(j*np.ones(l),En[i], sig))
            res[c] = s1
            s1=0
            c+=1
        return res
    



    # Some stuff for later
    @property
    def prop(self): return self._prop
    @prop.setter
    def prop(self, val):
        self._prop = val

    
# self.nx = kwargs.get('nx', 10) # System size in x-direction
# self.ny = kwargs.get('ny', 10)
#     def Hamiltonian(self):
# """Construct the Hamiltonian matrix."""
# # Placeholder for Hamiltonian construction logic
# H = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
# # Fill in the Hamiltonian matrix based on the model specifics
# return H