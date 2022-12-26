import pytest
from src.model import maketable

# Test the validity of the master dataframe including reading in the files
def test_firstK():
    mdf = maketable("WD", filters=["K"])
    assert mdf["K"].to_list()[0] == 22.217
def test_lastK():
    mdf = maketable("WD", filters=["K"])
    assert mdf["K"].to_list()[-1] == 10.273

def test_firstK3():
    mdf = maketable("WD", filters=["u", "K", "g"])
    assert mdf["K"].to_list()[0] == 22.217
def test_lastK3():
    mdf = maketable("WD", filters=["u", "K", "g"])
    assert mdf["K"].to_list()[-1] == 10.273

if __name__ == '__main__':
    test_firstK
    test_lastK
    test_firstK3
    test_lastK3