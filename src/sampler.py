import numpy as np
import emcee
import ptemcee
import multiprocessing as mp
from multiprocessing import Pool
import priors
import results
import system

import matplotlib.pyplot as plt

class MCMCSampler():
    """
    A MCMC (Markov chain Monte Carlo) sampler that supports parallel tempering in
    addition to simple MCMC.

    Args:
        system (System): a System object.
        num_temps (int): number of temperatures to run the sampler. 
            Parallel tempering will be used if num_temps > 1 (default=20)
        num_walkers (int): number of walkers at each temperature (default=200)
        num_threads (int): number of threads to use for parallelization (default=1)
    Written: Ziyyad Ali, 2022
    """

    def __init__(self, system, num_temps=20, num_walkers=200, num_threads=1) -> None:
        # Set the constructor attributes
        self.system = system
        self.num_temps = num_temps
        self.num_walkers = num_walkers
        self.num_threads = num_threads

        # Create a results object
        #TODO: Create self.results

        #TODO: self.model_table = model.maketable(table_type='WD')

        # Set the parallel tempering attributes
        if self.num_temps > 1:
            self.use_pt = True
        else:
            self.use_pt = False
            self.num_temps = 1
        
        # Set the list of priors
        self.priors = self.system.priors

        # Initialize the walker positions
        self.num_params = len(self.priors)
        init_pos = []

        for prior in self.priors:
            # Draw samples uniformly from each prior
            rand_init = prior.draw_samples(self.num_walkers * self.num_temps)
            if self.num_temps > 1:
                rand_init = rand_init.reshape([self.num_temps, self.num_walkers])
            init_pos.append(rand_init)
        
        # Set this as the current position for the walkers
        if self.use_pt:
            self.curr_pos = np.dstack(init_pos)
        else:
            self.curr_pos = np.stack(init_pos).T
        
    def _logl(self, params, include_logp=False):
        """
        A log likelihood function that will compute the sum of the log likelihoods
        
        Args:
            params (np.array of float): An array of fitting parameters of shape MxR.
                                        M is the number of steps per walker and R is the 
                                        number of parameters.
        Returns:
            lnlikes (float): sum of all the log likelihoods
        """
        if include_logp:
            logp = priors.all_lnpriors(params, self.priors)
            # escape if logp == -np.inf
            if np.isinf(logp):
                return -np.inf

        else:
            logp = 0  # don't include prior

        filts = self.system.datatable["Filter"]
        data = self.system.datatable["App Mag"]
        errors = self.system.datatable["Mag Error"]

        #TODO: model = model.wd_model(params, filts) #not a class
        
        #TODO: logl = chi_squared function

        return logl + logp

    def _update_chains_from_sampler(self, sampler, num_steps=None):
        """
        Updates the self.post (posterior), self.chain, and the self.lnlike

        Args:
            sampler (emcc.EnsembleSampler or ptemcee.Sampler): a sampler object
            num_steps (int): if not None, stores the first num_steps number of steps
        """
        if num_steps is None:
            # use all the steps, grab total number of steps from dimension of chains
            num_steps = sampler.chain.shape[-2]

        self.chain = sampler.chain
        num_params = self.chain.shape[-1]

        if self.use_pt:
            # chain is shape: Ntemp x Nwalkers x num_steps x Nparams
            self.post = sampler.chain[0, :, :num_steps].reshape(-1, num_params) # the reshaping flattens the chain
            # should also be picking out the lowest temperature logps
            self.lnlikes = sampler.loglikelihood[0, :, :num_steps].flatten()
            self.lnlikes_alltemps = sampler.loglikelihood[:, :, :num_steps]
        else:
            # chain is shape: Nwalkers x num_steps x Nparams
            self.post = sampler.chain[:, :num_steps].reshape(-1, num_params)
            self.lnlikes = sampler.lnprobability[:, :num_steps].flatten()

            # convert posterior probability (returned by sampler objects) to likelihood (required by src.results.Results)
            for i, orb in enumerate(self.post):
                self.lnlikes[i] -= priors.all_lnpriors(orb, self.priors)

    
    def run_sampler(self, num_steps:int, burn_steps:int, thin:int, output_filename:str):
        """
        Runs the sampler based on the parallel tempering option. 

        Args:
            num_steps (int): the number of steps to be taken by EACH walker.
            burn_steps (int): the number of burn in steps.
            thin (int): the frequency of save points. Must be > 0. A thin of 1 would
                        save every step while a thin of n would save every n steps. 
            output_filename (string): the path of the filename in which the steps will be
                                      be saved.
        Returns:
            sampler: a sampler object
        """

        if output_filename is None:
            raise ValueError("output_filename must be defined for saving")
        if thin <= 0:
            raise ValueError("thin must be > 0")
        if num_steps <= 0:
            raise ValueError("The number of steps must be > 0")

        with Pool(processes=self.num_threads) as pool: 
            if self.use_pt:
                sampler = ptemcee.Sampler(
                    self.num_walkers, self.num_params, self._logl, priors.all_lnpriors,
                    ntemps=self.num_temps, threads=self.num_threads, logpargs=[self.priors, ]
                )
            else:
                sampler = emcee.EnsembleSampler(
                    self.num_walkers, self.num_params, self._logl, pool=pool,
                    kwargs={'include_logp': True}
                )

            print("Starting Burn in")
            for i, state in enumerate(sampler.sample(self.curr_pos, iterations=burn_steps, thin=thin)):
                if self.use_pt:
                    self.curr_pos = state[0]
                else:
                    self.curr_pos = state.coords

                if (i+1) % 5 == 0:
                    print(str(i+1)+'/'+str(burn_steps)+' steps of burn-in complete', end='\r')

            sampler.reset()
            print('')
            print('Burn in complete. Sampling posterior now.')

            saved_upto = 0 # keep track of how many steps of this chain we've saved. this is the next index that needs to be saved 
            for i, state in enumerate(sampler.sample(self.curr_pos, iterations=num_steps, thin=thin)):
                if self.use_pt:
                    self.curr_pos = state[0]
                else:
                    self.curr_pos = state.coords
                    
                # print progress statement
                if (i+1) % 5 == 0:
                    print(str(i+1)+'/'+str(num_steps)+' steps completed', end='\r')

            print('')
            self._update_chains_from_sampler(sampler)
            self.results.add_samples(self.post, self.lnlikes, curr_pos=self.curr_pos)

            if output_filename is not None:
                self.results.save_results(output_filename)

            print('Run complete')
        # Close pool
        return sampler
    
    def examine_chains(self, param_list=None, walker_list=None, n_walkers=None, step_range=None, transparency = 1):
        """
        Plots position of walkers at each step from Results object. Returns list of figures, one per parameter
        Args:
            param_list: List of strings of parameters to plot (e.g. "sma1")
                If None (default), all parameters are plotted
            walker_list: List or array of walker numbers to plot
                If None (default), all walkers are plotted
            n_walkers (int): Randomly select `n_walkers` to plot
                Overrides walker_list if this is set
                If None (default), walkers selected as per `walker_list`
            step_range (array or tuple): Start and end values of step numbers to plot
                If None (default), all the steps are plotted
            transparency (int or float): Determines visibility of the plotted function
                If 1 (default) results plot at 100% opacity
        Returns:
            List of ``matplotlib.pyplot.Figure`` objects:
                Walker position plot for each parameter selected
        (written): Henry Ngo, 2019
        """

        # Get the flattened chain from Results object (nwalkers*nsteps, nparams)
        flatchain = np.copy(self.results.post)
        total_samples, n_params = flatchain.shape
        n_steps = np.int(total_samples/self.num_walkers) 
        # Reshape it to (nwalkers, nsteps, nparams)
        chn = flatchain.reshape((self.num_walkers, n_steps, n_params))

        # Get list of walkers to use
        if n_walkers is not None:  # If n_walkers defined, randomly choose that many walkers
            walkers_to_plot = np.random.choice(self.num_walkers, size=n_walkers, replace=False)
        elif walker_list is not None:  # if walker_list is given, use that list
            walkers_to_plot = np.array(walker_list)
        else:  # both n_walkers and walker_list are none, so use all walkers
            walkers_to_plot = np.arange(self.num_walkers)

        # Get list of parameters to use
        if param_list is None:
            params_to_plot = np.arange(n_params)
        else:  # build list from user input strings
            params_plot_list = []
            for i in param_list:
                if i in self.system.basis.param_idx:
                    params_plot_list.append(self.system.basis.param_idx[i])
                else:
                    raise Exception('Invalid param name: {}. See system.basis.param_idx.'.format(i))
            params_to_plot = np.array(params_plot_list)

        # Loop through each parameter and make plot
        output_figs = []
        for pp in params_to_plot:
            fig, ax = plt.subplots()
            for ww in walkers_to_plot:
                ax.plot(chn[ww, :, pp], 'k-', alpha = transparency)
            ax.set_xlabel('Step')
            if step_range is not None:  # Limit range shown if step_range is set
                ax.set_xlim(step_range)
            output_figs.append(fig)

        # Return
        return output_figs

    def chop_chains(self, burn, trim=0):
        """
        Permanently removes steps from beginning (and/or end) of chains from the 
        Results object. Also updates `curr_pos` if steps are removed from the 
        end of the chain.
        Args:
            burn (int): The number of steps to remove from the beginning of the chains
            trim (int): The number of steps to remove from the end of the chians (optional)
        .. Warning:: Does not update bookkeeping arrays within `MCMC` sampler object.
        (written): Henry Ngo, 2019
        TODO: Compare to ours which stores a file, which might be something we want
        """

        # Retrieve information from results object
        flatchain = np.copy(self.results.post)
        total_samples, n_params = flatchain.shape
        n_steps = np.int(total_samples/self.num_walkers)
        flatlnlikes = np.copy(self.results.lnlike)

        # Reshape chain to (nwalkers, nsteps, nparams)
        chn = flatchain.reshape((self.num_walkers, n_steps, n_params))
        # Reshape lnlike to (nwalkers, nsteps)
        lnlikes = flatlnlikes.reshape((self.num_walkers, n_steps))

        # Find beginning and end indices for steps to keep
        keep_start = burn
        keep_end = n_steps - trim
        n_chopped_steps = n_steps - trim - burn

        # Update arrays in `sampler`: chain, lnlikes, lnlikes_alltemps (if PT), post
        chopped_chain = chn[:, keep_start:keep_end, :]
        chopped_lnlikes = lnlikes[:, keep_start:keep_end]

        # Update current position if trimmed from edge
        if trim > 0:
            self.curr_pos = chopped_chain[:, -1, :]

        # Flatten likelihoods and samples
        flat_chopped_chain = chopped_chain.reshape(self.num_walkers*n_chopped_steps, n_params)
        flat_chopped_lnlikes = chopped_lnlikes.reshape(self.num_walkers*n_chopped_steps)

        # Update results object associated with this sampler
        # TODO: change to our results
        self.results = orbitize.results.Results(
            self.system, 
            sampler_name=self.__class__.__name__,
            post=flat_chopped_chain,
            lnlike=flat_chopped_lnlikes,
            version_number = orbitize.__version__,
            curr_pos = self.curr_pos
        )
        #TODO: results.savefile()

        # Print a confirmation
        print('Chains successfully chopped. Results object updated.')



        
        