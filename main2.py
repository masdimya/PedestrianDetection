# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:25:07 2019

@author: user
"""

import matplotlib.pyplot as plt

from selective_search import selective_search

a = plt.imread('../INRIAPerson/Train/pos/crop_000010.png')

coba = selective_search(a)

