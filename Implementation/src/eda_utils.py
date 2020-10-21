import pandas as pd
import numpy as np
import rfpimp


def get_correlated_columns(df: pd.DataFrame, threshold):
    limit = 0.95
    corr = df.corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_no_diag = corr.where(mask)
    coll = [c for c in corr_no_diag.columns if any(abs(corr_no_diag[c]) > limit)]
    return coll
