import numpy
import pandas as pd

data = [10, 20, 30, 40]
series = pd.Series([10, 20, 30, 40])
print(numpy.sum(series*series))