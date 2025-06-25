import numpy as np

class Model:
    def __init__(self, **kwargs):

        self.t = kwargs.get('t', 1.0) # Hopping parameter
        self.mu = kwargs.get('mu', 0.0) # Chemical potential

        self.Hk = self.Hamiltonian()

    def Hamiltonian(self):
        """Construct the k-space Hamiltonian function."""

        H = np.empty((2, 2), dtype=object)  # Placeholder for Hamiltonian matrix

        # Initialize the Hamiltonian matrix with zero functions
        for index, _ in np.ndenumerate(H): H[index] = lambda kx, ky: 0

        # Modify the Hamiltonian matrix with specific functions
        # Example: Dirac Hamiltonian for a 2D system
        H[0,1] = lambda kx, ky: self.t * (kx + 1j*ky) # Some function of kx, ky
        H[1,0] = lambda kx, ky: self.t * (kx - 1j*ky)
        H[0,0] = lambda kx, ky: -self.mu
        H[1,1] = lambda kx, ky: -self.mu

        def Hk(kx, ky): 
            """Evaluate the Hamiltonian at given kx, ky."""
            hk = np.empty_like(H, dtype=complex)
            for index in np.ndindex(H.shape): hk[index] = H[index](kx, ky)
            return hk
        
        return Hk


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