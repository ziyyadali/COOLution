import numpy as np
import h5py
import os
import src.system as system
import pandas as pd

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
    
    def save_results(self, filename):
        """
        Saves the results object to a h5py file
        """
        hf = h5py.File(filename, 'w')  # Creates h5py file object

        # Add the system object
        #hf.create_dataset('system', data=self.system.save_format(hf))

        # Now add post and lnlike from the results object as datasets
        hf.create_dataset('post', data=self.post)
        # hf.create_dataset('data', data=self.data)
        if self.lnlike is not None:
            hf.create_dataset('lnlike', data=self.lnlike)

        if self.curr_pos is not None:
            hf.create_dataset("curr_pos", data=self.curr_pos)

        hf.close()  # Closes file object, which writes file to disk
    
    def load_results(self, filename):
            """
            Populate the ``results.Results`` object with data from a datafile
            Args:
                filename (string): filepath where data is saved
                append (boolean): if True, then new data is added to existing object.
                    If False (default), new data overwrites existing object
            See the ``save_results()`` method in this module for information on how the
            data is structured.
            """
            #TODO:remove this if later
            hf = h5py.File(filename, 'r')  # Opens file for reading
            post = np.array(hf.get("post"))    

            try:
                curr_pos = np.array(hf.get('curr_pos'))
                lnlike = np.array(hf.get("lnlike"))
            except KeyError:
                curr_pos = None
                lnlike = None

            hf.close()  # Closes file object

            self.curr_pos = curr_pos
            self.lnlike = lnlike
            self.post = post

            """
            hf = h5py.File(filename, 'r')  # Opens file for reading
            data_table = pd.DataFrame(hf.get('data'))
            filts = np.array(hf.get('filters'), dtype='S').astype("str")
            data_table.columns = ["App Mag", "Errors"]
            data_table["Filter"] = filts
            plx = hf.attrs["plx"]
            plx_err = hf.attrs["plx_err"]
            keyword = hf.attrs["keyword"]

            # Make system object
            self.system = system.System(data_table, plx, plx_err, keyword=keyword)   
            post = np.array(hf.get("post"))    

            try:
                curr_pos = np.array(hf.get('curr_pos'))
                lnlike = np.array(hf.get("lnlike"))
            except KeyError:
                curr_pos = None
                lnlike = None

            hf.close()  # Closes file object

            if (curr_pos is not None) and (lnlike is not None) and (post is not None):
                self.curr_pos = curr_pos
                self.lnlike = lnlike
                self.post = post
            else:
                raise Exception(
                    'Unable to load file {} to Results object. Error reading data'.format(filename))
            """

            