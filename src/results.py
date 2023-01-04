"""
class:
    - system
    - lnlike
    - post
    - version_num
    - curr_pos

    init: 
        - set all params
    
    - add_samples copy-paste
    - save_results : dont need samplername and version num but copypaste
    - load_results: same as above
    
- system should be saved too (to know starting values wtc)
"""
import numpy as np

class Results():
    """
    The class holds the current results of the MCMC process. A system object is held along
    with the lnlike values at a position. The posteriors are also saved here. 

    Attributes:
            system  (System): Holds the system object containing the initial information
            lnlike  (np.array of float): M array of log-likelihoods corresponding to 
                                        the orbits described in ``post`` (default: None).
            post (np.array of float): MxN array of orbital parameters
                                    (posterior output from orbit-fitting process), where M is the
                                    number of orbits generated, and N is the number of varying orbital
                                    parameters in the fit (default: None).
            curr_pos (np.array of float): A multi-D array of the  current walker positions
                                         that is used for restarting a MCMC sampler. 

    """
    def __init__(self, system, lnlike=None, post=None, curr_pos=None):
        self.system = system
        self.lnlike = lnlike
        self.post = post
        self.curr_pos = curr_pos
        #self.version = version
    
    def add_samples(self, params, lnlike, curr_pos=None):
        """
        Add accepted orbits, their likelihoods to the results
        Args:
            params (np.array): add sets of orbital params (could be multiple) 
                to results
            lnlike (np.array): add corresponding lnlike values to results
            curr_pos (np.array of float): A multi-D array of the current walker positions
        """

        # If no exisiting results then it is easy
        if self.post is None:
            self.post = params
            self.lnlike = lnlike

        # Otherwise, need to append properly
        else:
            self.post = np.vstack((self.post, params))
            self.lnlike = np.append(self.lnlike, lnlike)

        if curr_pos is not None:
            self.curr_pos = curr_pos


        