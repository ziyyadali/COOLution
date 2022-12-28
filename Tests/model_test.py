import pytest
from src.model import *


filts = ["K", "H"]
mdf = maketable("WD", filters=filts)

# Expect errors
def test_neg_age(): 
    with pytest.raises(ValueError, match="Invalid value for age parameter"):
        findMags(mdf, 0.3, -2.2, 0, filters=filts)
def test_high_age(): 
    with pytest.raises(ValueError, match="Invalid value for age parameter"):
        findMags(mdf, 0.3, 1.6e10, 0, filters=filts)
def test_neg_mass(): 
    with pytest.raises(ValueError, match="Solar mass input creates invalid bounds"):
        findMags(mdf, -0.3, 1e10, 0, filters=filts)
def test_low_mass(): 
    with pytest.raises(ValueError, match="Solar mass input creates invalid bounds"):
        findMags(mdf, 0.1, 1e10, 0, filters=filts)
def test_high_mass():
    with pytest.raises(ValueError, match="Solar mass input creates invalid bounds"):
        findMags(mdf, 2, 1e10, 0, filters=filts)

    # Expect correct calculations
    #def out_test1(): pass
    #def out_test2(): pass
    #def out_test3(): pass

