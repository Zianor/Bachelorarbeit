from scipy.io import loadmat
from src.utils import get_project_root
import matplotlib.pyplot as plt
import os

mat_dict = loadmat(os.path.join(get_project_root(), 'data/ML_data_patient_1.mat'))
print(mat_dict.keys())
plt.plot(mat_dict['BBI_BCG']-mat_dict['BBI_ECG'])
plt.show()
