import numpy as np
import abc

"""
Note: All classes were derived from Orbitize! priors.py written by Sarah Blunt
"""

class Prior(abc.ABC):
    """
    Abstract base class for prior objects.
    All prior objects inherits from this class.
    """

    is_correlated = False
    @abc.abstractmethod
    def draw_samples(self, num_samples):
        pass

    @abc.abstractmethod
    def compute_lnprob(self, element_array):
        pass

class GaussianPrior(Prior):
    """Gaussian prior.
    .. math::
        log(p(x|\\sigma, \\mu)) \\propto \\frac{(x - \\mu)}{\\sigma}
    Args:
        mu (float): mean of the distribution
        sigma (float): standard deviation of the distribution
        no_negatives (bool): if True, only positive values will be drawn from
                            this prior, and the probability of negative values
                            will be 0 (default:True).
    """

    def __init__(self, mu, sigma, no_negatives=True):
        self.mu = mu
        self.sigma = sigma
        self.no_negatives = no_negatives

    def __repr__(self):
        return f'GaussianPrior(mu={self.mu}, sigma={self.sigma}, no_negatives={self.no_negatives})'

    def draw_samples(self, num_samples):
        """
        Draw positive samples from a Gaussian distribution.
        Negative samples will not be returned.
        Args:
            num_samples (float): the number of samples to generate
        Returns:
            numpy array of float: samples drawn from the appropriate
            Gaussian distribution. Array has length `num_samples`.
        """

        samples = np.random.normal(
            loc=self.mu, scale=self.sigma, size=num_samples
        )
        bad = np.inf

        if self.no_negatives:

            while bad != 0:
                bad_samples = np.where(samples < 0)[0]
                bad = len(bad_samples)

                samples[bad_samples] = np.random.normal(loc=self.mu, scale=self.sigma, size=bad)

        return samples

    def compute_lnprob(self, param):
        """
        Compute log(probability) of an array of numbers wrt a Gaussian distibution.
        Negative numbers return a probability of -inf.
        Args:
            param (float): A parameter in which the probability would be calculated from
                           a Gaussian distribution
        Returns:
            float: log(probability) of param value.
        """
        lnprob = -0.5*np.log(2.*np.pi) - 0.5*np.log(self.sigma**2) - 0.5*((param - self.mu) / self.sigma)**2

        return lnprob


class LogUniformPrior(Prior):
    """
    This is the probability distribution :math:`p(x) \\propto 1/x`
    The __init__ method should take in a "min" and "max" value
    of the distribution, which correspond to the domain of the prior.
    (If this is not implemented, the prior has a singularity at 0 and infinite
    integrated probability).
    Args:
        minval (float): the lower bound of this distribution
        maxval (float): the upper bound of this distribution
    """

    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

        self.logmin = np.log(minval)
        self.logmax = np.log(maxval)

    def __repr__(self):
        return f'LogUniformPrior(minval={self.minval}, maxval={self.maxval})'

    def draw_samples(self, num_samples):
        """
        Draw samples from this 1/x distribution.
        Args:
            num_samples (float): the number of samples to generate
        Returns:
            np.array:  samples ranging from [``minval``, ``maxval``) as floats.
        """
        # sample from a uniform distribution in log space
        samples = np.random.uniform(self.logmin, self.logmax, num_samples)

        # convert from log space to linear space
        samples = np.exp(samples)

        return samples

    def compute_lnprob(self, param):
        """
        Compute the prior probability of element given that its drawn from a Log-Uniofrm  prior
        Args:
            param (float): value compute the prior probability of from a Log Uniform distribution
        Returns:
            float: Log Uniform probability of drawing param
        """
        normalizer = self.logmax - self.logmin

        lnprob = -np.log((param*normalizer))

        if (param > self.maxval) or (param< self.minval):
            return -np.inf

        return lnprob


class UniformPrior(Prior):
    """
    This is the probability distribution p(x) propto constant.
    Args:
        minval (float): the lower bound of the uniform prior
        maxval (float): the upper bound of the uniform prior
    """

    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def __repr__(self):
        return f'UniformPrior(minval={self.minval}, maxval={self.maxval})'

    def draw_samples(self, num_samples):
        """
        Draw samples from this uniform distribution.
        Args:
            num_samples (float): the number of samples to generate
        Returns:
            np.array:  samples ranging from [0, pi) as floats.
        """
        # sample from a uniform distribution in log space
        samples = np.random.uniform(self.minval, self.maxval, num_samples)

        return samples

    def compute_lnprob(self, param):
        """
        Compute the prior probability of element given that its drawn from this uniform prior
        Args:
            param (float): value  of the element to compute the prior probability of from a 
                           uniform distribution
        Returns:
            float: the log probability of param
        """
        lnprob = np.log(1/(self.maxval - self.minval))

        # account for scalar inputs
        if (param > self.maxval) or (param < self.minval):
            lnprob = -np.inf

        return lnprob

def all_lnpriors(params, priors):
    """
    Calculates log(prior probability) of a set of parameters and a list of priors
    Args:
        params (np.array): size of N parameters
        priors (list): list of N prior objects corresponding to each parameter
    Returns:
        float: prior probability of this set of parameters
    """
    logp = 0.

    for param, prior in zip(params, priors):
        logp += prior.compute_lnprob(param)  # return a float

    return logp