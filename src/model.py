import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def make_master_df(filters):
    """
    Makes the master dataframe for all masses ranging from 0.2 to 1.1 
    solar masses.
    
    Paramaters:
        - filters: [String] -> a string of filters
    Returns:
        A DataFrame of the filters, age and mass
    """
    files = ['WD_Tables\Table_Mass_{:.1f}'.format(m) for m in np.arange(0.2, 1.2, 0.1)]

    mdf = pd.DataFrame()

    for file in files:
        mdf1 = pd.read_csv(file, header=1, usecols=filters+['Age'], delim_whitespace=True)
        # Remove Helium data
        rw = np.where(mdf1['Age'] == 'NUV')[0][0]
        lr = len(mdf1)-1
        mdf1.insert(3, "Mass", float(file[-3:]))
        mdf1 = mdf1[:rw-1].astype(float)
        mdf = pd.concat([mdf, mdf1])
    return mdf

def make_mass_df(mass, filters):
    """
    Makes the master dataframe for a specific mass with the specified
    filters.
    
    Paramaters:
        - mass:    float    -> a float of the mass containing a precision of 
                               one decimal place
        - filters: [String] -> a string of filters
    Returns:
        A DataFrame of the filters, age and mass
    """
    file = 'WD_Tables\Table_Mass_{:.1f}'.format(mass)

    mdf1 = pd.read_csv(file, header=1, usecols=filters+['Age'], delim_whitespace=True)
    # Remove Helium data
    rw = np.where(mdf1['Age'] == 'NUV')[0][0]
    lr = len(mdf1)-1
    mdf1.insert(3, "Mass", float(file[-3:]))
    mdf1 = mdf1[:rw-1].astype(float)
    
    return mdf1



"""


solar_m = 0.93
mbounds = (np.around(np.floor(10*solar_m)/10, decimals=1), np.around(np.floor(10*solar_m)/10 + 0.1, decimals=1))
mbounds

df = mdf.loc[mdf["Mass"] == mbounds[0]]
#print(df)
age_bounds = df.iloc[(df["Age"]-1.345e10).abs().argsort()[:2]][["H1", 'Age']].sort_values("Age", ascending=False).values
f = interp1d(age_bounds[:,1:].flatten(), age_bounds[:,:-1].flatten())
mag = f(1.345e10)
#print(mag)
age_bounds

findMags(mdf, solar_m, 1.215e10, ["H1", "K"])
"""
solar_m = 0.93
age = 1.215e10
count = 0
while age > 1.:
    findMags(mdf, solar_m, age, ["H1", "K"])
    count += 1
    age -= 0.2e7
print(count)


def findMags(mdf, solar_m, age, filters):
    """
    Version 1 O(n): 
            <<ESTIMATES>>
            - runs    = 150*time(s) + 0.87
            - time(s) = 0.007*runs - 0.01
    Given the solar mass in a range of 0.2 to 1.1 inclusively, interpolate
    linearly from the upper and lower bounds (from the white dwarf cooling 
    models) with respect to the age.
    Parameters:
        - mdf:         DataFrame -> the master dataframe containing the WD Cooling Models
        - solar_m:     float     -> must be rounded to the nearest tenth decimal
        - filters:     [String]  -> a list containing the filters as strings.
        - age:         [float]   -> the estimated age of the white dwarf
        
    Returns:
        - The estimated magnitude for the each filter in the order of filters
    """
    
    # Make array to hold magnitudes. Elm 0 is are the interp values for the lowerbound; Elm 1 is the upper.
    mags = np.zeros((2,len(filters)))
    # TODO: Check age limits for dataframes
    
    # Makes the bounds and ensures floats have one decimal (avoiding floating point errors)
    mbounds = (np.around(np.floor(10*solar_m)/10, decimals=1), np.around(np.floor(10*solar_m)/10 + 0.1, decimals=1))
    #print("Mass bounds:\n", mbounds)
    
    # Outer loop runs twice only
    for m in range(len(mbounds)):
        # Make sub dataframe for specific mass
        df = mdf.loc[mdf["Mass"] == mbounds[m]]
        #print("\nSpecific Mass dataframe:\n", df)
        
        # For each filter, the upper and lower bounds of the dataframe is found with respect to the filter value
        for i in range(len(filters)):
            #print(filters[i])
            sub = df["Age"]-age
            age_upper = df.loc[[df.loc[sub > 0, "Age"].idxmin()]][[filters[i], 'Age']]
            age_lower = df.loc[[df.loc[sub < 0, "Age"].idxmax()]][[filters[i], 'Age']]
            age_bounds = pd.concat([age_upper, age_lower]).values
            #print("Age bounds:\n", age_bounds)

            # y = f(x) (y would be magnitude, x is the age)
            f = interp1d(age_bounds[:,1:].flatten(), age_bounds[:,:-1].flatten())
            
            # Add the magnitude in the bound array in the order of the filter array ie. elm 0 is filt 0's magnitude
            mags[m][i] = f(age)
            #print("Current mags:\n", mags)
    
    # Calculates the magnitude for the age given 
    fract = (solar_m - mbounds[0])/(mbounds[1]-mbounds[0])
    #print("Fract val:\n", fract)
    plus = fract * (mags[1]-mags[0])
    #print("Plus:\n", plus)
    fmags = mags[0] + plus
    
    return fmags
