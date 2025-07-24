import numpy as np


class Model:
    def __init__(self, **kwargs):

        self.t = kwargs.get('t', 1.0) # Hopping parameter
        self.U = kwargs.get('U', 1) #strength of attractive interaction
        self.mu = kwargs.get('mu', 0.0) # Chemical potential
        self.muB = kwargs.get('muB', 0.0)
        self.nA = kwargs.get('nA', 0.0)
        self.nB = kwargs.get('nB', 0.0)
        self.nC = kwargs.get('nC', 0.0)

        # Internal variables are named with _ for convention
        self.Del0A = kwargs.get('Del0A', 0.0)
        self.Del0B = kwargs.get('Del0B', 0.0)
        self.Del0C = kwargs.get('Del0C', 0.0)

        # Inhomogeneous pairing/site?
        self.inhomp = kwargs.get('inhomp', False)
        self.inhomi = kwargs.get('inhomi', False)

    
        self.Hk = self.HBdG()

        #special treatment of B site        
        if not self.inhomp:
            self.muB = self.mu
        
            #self.muA = self.mu
            #self.muC = self.mu
        if not self.inhomi:
            #self.Del0 = 0
            self.Del0A = self.Del0B
            self.Del0C = self.Del0B

    def HBdG(self):
        """Construct the real space sparse Hamiltonian function."""
        # Create a sparse matrix representation of the Hamiltonian
        H = np.zeros((12, 12), dtype=object)  # Placeholder for Hamiltonian matrix
        #H = self.mu*H
        for index, _ in np.ndenumerate(H): H[index] = lambda kx, ky: 0


        #diagonals
        H[np.array([0, 3]), np.array([0, 3])] = lambda kx, ky: self.mu+self.nA/4*self.U
        H[np.array([2, 5]), np.array([2, 5])] = lambda kx, ky: self.mu+self.nC/4*self.U
        H[np.array([6, 9]), np.array([6, 9])] = lambda kx, ky: - self.mu-self.nA/4*self.U
        H[np.array([8, 11]), np.array([8, 11])] = lambda kx, ky: - self.mu-self.nC/4*self.U
        #special treatment of B site
        H[np.array([1, 4]), np.array([1, 4])] = lambda kx, ky: self.muB + self.nB/4*self.U
        H[np.array([7, 10]), np.array([7, 10])] = lambda kx, ky: -self.muB- self.nB/4*self.U
        
        #h
        H[np.array([0, 1, 3, 4]), np.array([1, 0, 4, 3])] = lambda kx, ky: -2*self.t * np.cos(kx/2) # Some function of kx, ky
        H[np.array([1, 2, 4, 5]), np.array([2, 1, 5, 4])] = lambda kx, ky: -2*self.t * np.cos(ky/2) 
        #h*
        H[np.array([6, 7, 9, 10]), np.array([7, 6, 10, 9])] = lambda kx, ky: 2*self.t * np.cos(kx/2) # Some function of kx, ky
        H[np.array([7, 8, 10, 11]), np.array([8, 7, 11, 10])] = lambda kx, ky: 2*self.t * np.cos(ky/2)
        

        #Del
        #on site
        H[np.array([0, 9]), np.array([9, 0])] = lambda kx, ky: self.Del0A
        H[np.array([3, 6]), np.array([6, 3])] = lambda kx, ky: -self.Del0A

        H[np.array([1, 10]), np.array([10, 1])] = lambda kx, ky: self.Del0B
        H[np.array([4, 7]), np.array([7, 4])] = lambda kx, ky: -self.Del0B
        
        H[np.array([2, 11]), np.array([11, 2])] = lambda kx, ky: self.Del0C
        H[np.array([5, 8]), np.array([8, 5])] = lambda kx, ky: -self.Del0C
      
        
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
            e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])) #

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
        arr1 = np.ones(l)
        s1 = 0
        res = np.ones(np.shape(E)[0])
        c=0
        for j in E:
            for i in range(b):
                s1 += np.sum(Gauss(j*arr1,En[i], sig))
            res[c] = s1
            s1=0
            c+=1
        return res
    def simple_stats(self):
        '''
        Simple calculation of statistical values over path in BZ: Gamma -> X -> M -> Gamma = (0,0) -> (2pi, 0) -> (2pi, 2pi) -> 0,0
        
        returns: 
            average energy and standard deviation of each band
            max and min E for each band
            minimal energy gap between each neighbouring band (in order from lowest to highest band)
            average energy gap between each neighbouring band (in order from lowest to highest band)
        '''
        k = np.linspace(0, 2*np.pi, 100)

        k1 = np.ones(100)
        k0 = np.zeros(100)

        kx = np.concatenate((k,np.pi*2*k1, k[::-1]))
        ky = np.concatenate((k0, k, k[::-1]))

        E = self.solvHam(kx, ky)

        maxe = np.amax(E, axis=1)
        av = np.average(E, axis=1)
        stde = np.std(E, axis=1)
        maxe = np.amax(np.abs(E), axis=1)
        mine = np.amin(np.abs(E), axis=1)
        egap = np.ones(5)
        for i in range(5):
            egap[i] = np.amin(np.abs(E[2*i]-E[2*(i+1)]))
        egap_av = np.ones(5)
        for i in range(5):
            egap_av[i] = np.average(np.abs(E[2*i]-E[2*(i+1)]))
        stats = {'maxe': maxe, 'av': av, 'std': stde, 'max': maxe, 'min': mine, 'mingap': egap, 'avgap': egap_av}

        return stats