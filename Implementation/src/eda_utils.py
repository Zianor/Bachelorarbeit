import numpy as np
import pandas as pd


def get_correlated_columns(df: pd.DataFrame, threshold):
    limit = 0.95
    corr = df.corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_no_diag = corr.where(mask)
    coll = [c for c in corr_no_diag.columns if any(abs(corr_no_diag[c]) > limit)]
    return coll


def get_latex_column_names():
    return {
        'error': '$E\\textsubscript{HR}$',
        'hf ratio data': '$\\text{ratio}\\textsubscript{data}$',
        'hf ratio acf': '$\\text{ratio}\\textsubscript{acf}$',
        'hf diff acf': '$\\text{diff}\\textsubscript{acf}$',
        'hf diff data': '$\\text{diff}\\textsubscript{data}$',
        'abs energy': '$E\\textsubscript{abs}$',
        'interval lengths std': '$\\text{IL}\\textsubscript{std}$',
        'interval lengths range': '$\\text{IL}\\textsubscript{range}$',
        'sqi mean': '$\\text{SQI}\\textsubscript{mean}$',
        'sqi median': '$\\text{SQI}\\textsubscript{median}$',
        'sqi min': '$\\text{SQI}\\textsubscript{min}$',
        'sqi std': '$\\text{SQI}\\textsubscript{std}$',
        'sqi max': '$\\text{SQI}\\textsubscript{max}$',
        'peak range': '$\\text{P}\\textsubscript{range}$',
        'peak mean': '$\\text{P}\\textsubscript{mean}$',
        'peak std': '$\\text{P}\\textsubscript{std}$',
        'template corr highest sqi mean': '$\\text{mean}\\textsubscript{T\\textsubscript{SQI}}$',
        'template corr highest sqi std': '$\\text{std}\\textsubscript{T\\textsubscript{SQI}}$',
        'template corr median sqi mean': '$\\text{mean}\\textsubscript{T\\textsubscript{median}}$',
        'template corr median sqi std': '$\\text{std}\\textsubscript{T\\textsubscript{median}}$',
        'interval means std': '$\\text{mean}\\textsubscript{std}$',
        'interval stds std': '$\\text{std}\\textsubscript{std}$',
        'interval ranges std': '$\\text{range}\\textsubscript{std}$',
        'sqi coverage 03': '$C\\textsubscript{0{,}3}$',
        'sqi coverage 04': '$C\\textsubscript{0{,}4}$',
        'sqi coverage 05': '$C\\textsubscript{0{,}5}$'
    }
