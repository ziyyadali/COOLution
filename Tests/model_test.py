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
def test_nofilts():
    with pytest.raises(ValueError, match="Filters array must be a subset of the filters in master dataframe"):
        findMags(mdf, 2, 1e10, 0, filters=[])
def test_excessfilts():
    with pytest.raises(ValueError, match="Filters array must be a subset of the filters in master dataframe"):
        findMags(mdf, 2, 1e10, 0, filters=filts+["B"])

# Expect correct calculations. First set with one filter
def test_FMout1():
    mags = findMags(mdf, 0.92, 1.143e10, 41.47, ["K"])
    dist = 1/(41.47/1000)
    app_fmag = 5*np.log10(dist/10) + 16
    assert mags == pytest.approx([app_fmag], rel=0.3)
def test_FMout2():
    mags = findMags(mdf, 0.95, 1.143e10, 41.47, ["K"])
    dist = 1/(41.47/1000)
    app_fmag = 5*np.log10(dist/10) + 16.75
    assert mags == pytest.approx([app_fmag], rel=0.3)
def test_FMout3():
    mags = findMags(mdf, 0.98, 1.143e10, 41.47, ["K"])
    dist = 1/(41.47/1000)
    app_fmag = 5*np.log10(dist/10) + 17.56
    assert mags == pytest.approx([app_fmag], rel=0.3)
# Two filters
def test_FMout1b():
    mags = findMags(mdf, 0.92, 1.143e10, 41.47, ["K", "H"])
    dist = 1/(41.47/1000)
    app_fmag = 5*np.log10(dist/10) + np.array([16, 15.8])
    assert mags == pytest.approx(app_fmag, rel=0.3)
def test_FMout2b():
    mags = findMags(mdf, 0.95, 1.143e10, 41.47, ["K", "H"])
    dist = 1/(41.47/1000)
    app_fmag = 5*np.log10(dist/10) + np.array([16.75, 16.3])
    assert mags == pytest.approx(app_fmag, rel=0.3)
def test_FMout3b():
    mags = findMags(mdf, 0.98, 1.143e10, 41.47, ["K", "H"])
    dist = 1/(41.47/1000)
    app_fmag = 5*np.log10(dist/10) + np.array([17.56, 17.4])
    assert mags == pytest.approx(app_fmag, rel=0.3)


