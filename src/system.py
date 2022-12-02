"""
Structure:
class:
    data_table: filt, appmag, error (measured)
    plx
    plx_err
    sys_priors: LogUniform:mass, LogUniform/Uniform:age, Gaussian/LogUniorm:plx
    labels: names of params 9mass, age , plx
    creation_date

    - init:
        - stora attributes
        - assign priors: to self.sys_priors
        - keyword: 'informative':gaussian for plx and uniform for age / 'uninformative': Uniform on everything, have guess for plx (10sigma for bounds)
    - update_priors (if people want to change it)
    - a function to save the object attributes so results.load can recreate the object
    - print_results
    - plot_corner

"""