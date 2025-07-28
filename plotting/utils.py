# import os
# import pprint
# from itertools import product

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.stats import median_abs_deviation

# import wandb


def _abbreviate_val(val):
    if isinstance(val, bool):
        return "T" if val else "F"
    return val
