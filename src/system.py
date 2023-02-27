import pandas as pd
import numpy as np
import datetime
import src.priors as priors

class System():
    """
    The structure class will contain the data along with the priors used in the 
    MCMC sampler. 

    Attributes:
        data_table (DataFrame): holds the data in a table with columns: App_Mag, Errors, Filter
        plx (float): the parallax value of the star
        plx_err (float): the error on the parallax measurement
        sys_priors ([Prior]): a list of priors on the parameters (in the order of labels)
        labels ([str]): names of the paramaters
        keyword ('informative' or 'uninformative') : informative places a Gaussian prior on the plx and 
                                                      a Uniform prior on the age. The other option puts a
                                                      Uniform prior on all parameters.
        creation_date (str): date the System object was created
    """

    def __init__(self, data_table, plx, plx_err, keyword='informative', prev_creation_date=None) -> None:
        """
        Args:
            data_table (DataFrame): holds the data in a table with columns: App_Mag, Errors, Filter
            plx (float): the parallax value of the star
            plx_err (float): the error on the parallax measurement
            keyword ('informative' or 'uninformative') : informative places a Gaussian prior on the plx and 
                                                        a Uniform prior on the age. The other option puts a
                                                        Uniform prior on all parameters.
            prev_creation_date (str): date the System object was previously created

        """
        # Set the attributes
        self.data_table = data_table
        self.plx = plx
        self.plx_err = plx_err
        self.keyword = keyword

        #if (keyword != 'informative') or (keyword != 'uninformative'):
        #    raise ValueError("keyword must be 'informative' or 'uninformative'")

        # Create list of priors
        self.labels = ['mass', 'age', 'plx']
        if self.keyword == 'informative':
            self.sys_priors = [priors.UniformPrior(0.2, 1.3), 
                               priors.UniformPrior(0.0, 1.564e10), 
                               priors.GaussianPrior(self.plx, self.plx_err)]
        else:
            # Puts a uniform prior on all paramaters, puts a 10*sigma range on either side of values
            self.sys_priors = [priors.UniformPrior(0.2, 1.3), 
                               priors.UniformPrior(0, 1.564e10), 
                               priors.UniformPrior(self.plx - 10*self.plx_err, self.plx + 10*self.plx_err)]
        self.init_mass_prior = priors.UniformPrior(0.0, 5.65e9) #TODO age
        # Set creation date
        if prev_creation_date is None:
            self.creation_date = str(datetime.datetime.now())
        else:
            self.creation_date = prev_creation_date
    
        
    def update_priors(self, new_priors):
        """
        Sets the default priors to the new priors in new_priors.
        
        Args:
            new_priors ([Prior]): a list of Prior objects
        TODO: Complete
        """
        raise NotImplementedError("Update priors is not yet completed")
    
    def save_format(self, hf):
        """
        Transforms the attributes of the System object in a format so
        Results can save the data.

        Args:
            hf (h5py._hl.files.File): a currently open hdf5 file in which
                to save the object.  
        """
        hf.create_dataset('data', data=self.data_table)
        hf.attrs['plx'] = self.plx
        hf.attrs['plx_err'] = self.plx_err
        #for i in range(3):
        #    hf.attrs[f"Prior({self.labels[i]})"] = str(self.priors[i])
    
