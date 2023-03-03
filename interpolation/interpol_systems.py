import pickle
import os
import numpy as np
import pandas as pd
import pdb
import glob
import itertools
import functools
import itertools
import scipy.interpolate
import scipy.linalg
import scipy.optimize
import sharpy.linear.src.libsparse


import sharpy.rom.utils.librom as librom
import sharpy.rom.utils.librom_interp as librom_interp
import sharpy.linear.src.libss as libss



class SS_interpol:
    """
    d
    """

    def __init__(self):

        self.bases = None
        self.kernel = None
        self.kernel_parameters = None
        self.interpol_settings = None
        self.balfreq_settings = None
        
    def read_bases(self, _file):
        
        self.bases = None
        self.parameters = None
        
    def build_ss(self, X):

        if self.interpol_type == 'interpol_matrix':
            ss = build_ss_interpol(X)
        elif self.interpol_type == 'interpol_balfreq':
            model_interpol = Interpolation_ss(self.model_points,
                                              np.ones((self.num_model_points,1)),
                                              **self.interpol_settings)
            interpolation_factors = model_interpol.interpol_weight_factors(X)
            ss, hsv = build_balfreq_interpol(ddf, interpolation_factors)

        return ss

def build_full_interpol():
    pass



ss_model = ReadDoE_ss('linear.linear_system.uvlm.ss')

path_model = '/mnt/work/Programs/RHEA/software/gemseo_rhea/examples/sharpy_smithwing/roms/ROMs'
path_training = '/mnt/work/Programs/RHEA/software/gemseo_rhea/examples/sharpy_smithwing/roms/'
path_testing = '/mnt/work/Programs/RHEA/software/gemseo_rhea/examples/sharpy_smithwing/roms/'

ss_model.get_model_points(path_model)
ss_model.get_training_points(path_training)
ss_model.get_testing_points(path_testing)


# ss_model.doe_info['model_points']['df']

interpol_settings = {'base_name': 'rb_multiquadric',
                     'base_inputs': {},
                     'scipy_inputs': {'fun_name': 'Rbf',
                                      'function': 'multiquadric'}}

balfreq_settings = {'frequency': 1.2,
                    'method_low': 'trapz',
                    'options_low': {'points': 12},
                    'method_high': 'gauss',
                    'options_high': {'partitions': 2, 'order': 8},
                    'check_stability': True}

model_ddf = build_balfreq(ss_model.dict_model_data, balfreq_settings)
model_interpol = Interpolation_ss(ss_model.model_points,
                                  np.ones((ss_model.num_model_points,1)),
                                  **interpol_settings)

ss_int = [[] for i in range(ss_model.num_testing_points)]
hsv_int = [[] for i in range(ss_model.num_testing_points)]
for ti in range(ss_model.num_testing_points):
    interpolation_factors = model_interpol.interpol_weight_factors(
        ss_model.testing_points[ti, :])
    ss_int[ti], hsv_int[ti] = build_balfreq_interpol(model_ddf, interpolation_factors)
