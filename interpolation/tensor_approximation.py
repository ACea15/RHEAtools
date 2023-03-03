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
import sharpy.rom.utils.librom as librom
import sharpy.rom.utils.librom_interp as librom_interp
import sharpy.linear.src.libss as libss
from sharpy.rom.utils.bases import Bases
from sharpy.rom.utils.doe import ReadDoE, ReadDoE_ss


class Interpolation_ss(Bases):
    """
    Documentation for Approximation_SS
    """
    def __init__(self,
                 model_points,
                 model_data,
                 base_name,
                 base_inputs={},
                 scipy_inputs={},
                 *args, **kwargs):
        
        super(Interpolation_ss, self).__init__(*args, **kwargs)

        self.data_shape = np.shape(model_data)
        if len(self.data_shape) > 2:
            self.reshape = 1
            self.model_data = reshape_tensor(model_data)
        else:
            self.reshape = 0
            self.model_data = model_data
        self.data_nx, self.data_ny = np.shape(self.model_data)
        self.model_points = np.array(model_points)
        self.basei_inputs = base_inputs
        self.basei_name = base_name
        self.epsilon = None
        self.scipy_inputs = scipy_inputs
        #self.interpol_scp = kwargs.get('interpol_scp', None)
        self.interpol_build()

    @classmethod
    def from_pandas(cls, df, **settings):

        model_points = df[df.columns[:-1]].to_numpy()
        model_data = df[df.columns[-1]].to_numpy()
        return cls(model_points, model_data, **settings)
    
    def __call__(self, X, from_interpolation=1):

        if from_interpolation == 1:
            A = self.interpol_model1(X)
        elif from_interpolation == 2:
            A = self.interpol_model2(X)
        elif from_interpolation == 3:
            A = self.interpol_model3(X)

        if self.reshape:
            A = shape_tensor_back(A, self.data_shape[1:])
        return A

    # def interpol_entries(x, Mz, function='multiquadric',smooth=0):

    #     num_samples, num_entries = np.shape(Mz)
    #     interpol_entries = []
    #     for i in range(num_entries):
    #         interpol_entries.append(Rbf(*x.T, Mz[:,i],function=function, smooth=smooth))
    #         print('Interpolated %s entries of a total of %s'%(i,num_entries))

    #     return interpol_entries
                    
    def interpol_build(self):
        """
        Build the interpolation sequentially
        """
        
        if self.scipy_inputs: # build interpolation from scipy
            self.interpol_scipy(**self.scipy_inputs)
            self.epsilon = self.interpolant_scp[0].epsilon
        if self.basei_name: # build interpolation from Base functions
            self.basei_fun = getattr(self, self.basei_name)
            self.kernel_interpol()
            self.interpol_matrix()
            self.interpol_weights()

    def interpol_scipy(self, fun_name, **settings):
        """
        Interpolation from Scipy
        """
        
        fun = getattr(scipy.interpolate, fun_name)
        
        self.interpolant_scp = [[] for j in range(self.data_ny)]
        for j in range(self.data_ny):
            y = self.model_data[:, j]
            try:
                self.interpolant_scp[j] = fun(self.model_points, y, **settings)
            except ValueError:
                self.interpolant_scp[j] = fun(*self.model_points.T, y, **settings)
                
    def kernel_interpol(self):
        """
        Prepares the interpolation kernel at the specified points
        """
        
        self.basei_inputs.update(points=self.model_points)
        if 'rb_' in self.basei_name and self.epsilon: # use epsilon from scipy interpolation
            self.basei_inputs.update(b=self.epsilon)

        self.bases_inter = self.basei_fun(**self.basei_inputs)

    def interpol_matrix(self):
        """
        Builds interpolation matrix F_inter:
        F_inter * w = model_data
        """
        self.F_inter = np.zeros((self.data_nx, self.data_nx))
        for i in range(self.data_nx):
            for j in range(self.data_nx):
                self.F_inter[i, j] = self.bases_inter[j](self.model_points[i])
        self.F_inter_inv = np.linalg.inv(self.F_inter)

    def interpol_weights(self):
        """
        builds weights of interpolation
        """
        # W = []
        # for i in range(self.num_model_points):
        #     for j in range(self.num_model_points):
        #         W.append(self.F_inter_inv[i,j]*self.model_data[j])
        
        self.W_interpol = self.F_inter_inv.dot(self.model_data)

    def interpol_weight_factors(self, X):
        """
        Weights factor that multiplies the input data as another mean
        to build the interpolation
        """
        
        self.Wf_interpol = np.zeros(self.data_nx)
        for j in range(self.data_nx):
            for i in range(self.data_nx):
                self.Wf_interpol[j] += (self.bases_inter[i](X)*self.F_inter_inv[i,j])
        return self.Wf_interpol
    
    def interpol_model1(self, X):
        """
        Find interpolation vector A at X using weight factors
        """

        weight_factors = self.interpol_weight_factors(X)
        A = self.Wf_interpol.dot(self.model_data)
        return A
    
    def interpol_model2(self, X):
        """
        Find interpolation vector A at X using weights
        """
        
        A = np.zeros(np.shape(self.W_interpol)[1])
        for i in range(self.data_nx):
            A += self.W_interpol[i]*self.bases_inter[i](X)
        return A
    
    def interpol_model3(self, X):
        """
        Find interpolation vector A at X using scipy function
        """
        A = [self.interpolant_scp[i](*X) for i in range(self.data_ny)]
        A = np.array(A)
        return A

def build_ss_interpol(X, A, B, C, D, dt, model_points=None,
                      interpol_settings={}, *args, **kwargs):
    """
    Builds state-space interpolated system at X
    """

    if isinstance(A, Interpolation_ss):
        A_intX = A(X)
        B_intX = B(X)
        C_intX = C(X)
        D_intX = D(X)
    elif isinstance(A, (np.ndarray, list)) and model_points is not None:
        #################
        A_int = Interpolation_ss(model_points,
                                 A,
                                 **interpol_settings)
        A_intX = A_int(X)
        #################
        B_int = Interpolation_ss(model_points,
                                 B,
                                 **interpol_settings)
        B_intX = B_int(X)
        #################
        C_int = Interpolation_ss(model_points,
                                 C,
                                 **interpol_settings)
        C_intX = C_int(X)
        #################
        D_int = Interpolation_ss(model_points,
                                 D,
                                 **interpol_settings)
        D_intX = D_int(X)

    return libss.StateSpace(A_intX, B_intX, C_intX, D_intX, dt=dt)

def build_balfreq_interpol(ddf, interpolation_factors, N=0,
                           system_labels=['Ab','Bb', 'Cb', 'Db', 'dt'],
                           svd_labels=['U_hsv', 'hsv_Vt', 'U_hsv_Vt']):
    """
    Builds frequency balancing interpolation
    """
    ss_list = []
    if isinstance(ddf, dict):
        num_samples = len(ddf[system_labels[0]])
    elif isinstance(ddf, pd.DataFrame):
        num_samples = len(ddf)

    for i in range(num_samples):
        Ai, Bi, Ci, Di, dti = ddf_get_entry(ddf, i, system_labels)
        ss_list.append(libss.StateSpace(Ai, Bi, Ci, Di, dt=dti))
    
    #interpolation_factors = 4
    U_list, VT_list, M_list = ddf_get_colums(ddf, svd_labels)

    ss_int, hsv_int = librom_interp.FLB_transfer_function(ss_list,
                                                          interpolation_factors,
                                                          U_list,
                                                          VT_list,
                                                          hsv_list=None,
                                                          M_list=M_list,
                                                          N=N)

    return ss_int, hsv_int

def build_balfreq(ddf, settings):
    """
    Builds frequecy-limited balanced models
    """
    if isinstance(ddf, dict):
        num_samples = len(ddf['A'])
    elif isinstance(ddf, pd.DataFrame):
        num_samples = len(ddf)
    for i in range(num_samples):

        Ai, Bi, Ci, Di, dti = ddf_get_entry(ddf, i, ['A', 'B', 'C', 'D', 'dt'])
        
        ss_i = libss.StateSpace(Ai, Bi, Ci, Di, dt=dti)
        SSb, hsv, T, Ti, Zc, Zo, U, Vt = librom.balfreq(ss_i, settings)
        print(i, len(hsv))
        #import pdb; pdb.set_trace()
        to_add = [['Ab', SSb.A], ['Bb', SSb.B], ['Cb', SSb.C], ['Db', SSb.D]]
        to_add += [['hsv',hsv], ['T',T], ['Ti',Ti],
                   ['Zc',Zc], ['Zo', Zo], ['U', U], ['Vt', Vt]]
        try:
            U_hsv_Vt = np.dot(U * hsv, Vt)
            U_hsv = U * hsv**0.5
            hsv_Vt = np.diag(hsv**0.5).dot(Vt)
        except ValueError:
            
            n_hsv = len(hsv)
            U_hsv_Vt = np.dot(U[:, :n_hsv] * hsv, Vt[:n_hsv, :])
            U_hsv = U[:, :n_hsv] * hsv**0.5
            hsv_Vt = np.diag(hsv**0.5).dot(Vt[:n_hsv, :])
            
        to_add += [['U_hsv_Vt',U_hsv_Vt], ['U_hsv', U_hsv], ['hsv_Vt', hsv_Vt]]

        if i == 0:
            ddf_initialise_array(ddf, to_add, num_samples)
        ddf_update_entry(ddf, i, to_add)
        
    return ddf

def build_ss(ddf, labels=['A', 'B', 'C', 'D', 'dt'], name='ss'):
    """
    Builds frequecy-limited balanced models
    """
    if isinstance(ddf, dict):
        num_samples = len(ddf[labels[0]])
    elif isinstance(ddf, pd.DataFrame):
        num_samples = len(ddf)
    ddf[name] = [[] for i in range(num_samples)]
    for i in range(num_samples):

        Ai, Bi, Ci, Di, dti = ddf_get_entry(ddf, i, ['A', 'B', 'C', 'D', 'dt'])
        ddf[name][i] = libss.StateSpace(Ai, Bi, Ci, Di, dt = dti)

def truncate_balfreq(ddf, N):
    """
    Builds frequecy-limited balanced models
    """
    if isinstance(ddf, dict):
        num_samples = len(ddf['Ab'])
    elif isinstance(ddf, pd.DataFrame):
        num_samples = len(ddf)
    for i in range(num_samples):

        Ati, Bti, Cti, Dti, hsvti, Uti, Vtti = ddf_get_entry(ddf, i, ['Ab',
                                                                      'Bb',
                                                                      'Cb',
                                                                      'Db',
                                                                      'hsv',
                                                                      'U',
                                                                      'Vt'])
        At = Ati[:N, :N]
        Bt = Bti[:N, :]
        Ct = Cti[:, :N]
        hsvt = hsvti[:N]
        Ut = Uti[:, :N]
        Vtt = Vtti[:N, :]
        U_hsv_Vtt = np.dot(Ut * hsvt, Vtt)
        U_hsvt = Ut * hsvt**0.5
        hsv_Vtt = np.diag(hsvt**0.5).dot(Vtt)
        
        to_add = [['At', At], ['Bt', Bt], ['Ct', Ct], ['Dt', Dti]]
        to_add += [['hsvt',hsvt], ['Ut', Ut], ['Vtt', Vtt]]
        to_add += [['U_hsv_Vtt', U_hsv_Vtt], ['U_hsvt', U_hsvt],
                   ['hsv_Vtt', hsv_Vtt]]

        if i == 0:
            ddf_initialise_array(ddf, to_add, num_samples)
        ddf_update_entry(ddf, i, to_add)
        
    return ddf



def ddf_initialise_array(ddf, columns, num_samples):
    
    for col_i, val_i in columns:
        if isinstance(ddf, dict):
            ddf[col_i] = [np.zeros_like(val_i) for i in range(num_samples)]
        elif isinstance(ddf, pd.DataFrame):
            ddf[col_i] = ddf.apply(lambda r: tuple(np.zeros_like(val_i)),
                                    axis = 1).apply(np.array)

def ddf_update_entry(ddf, row, columns):
    
    if isinstance(ddf, pd.DataFrame):
        for col_i, val_i in columns:
            ddf.loc[row, col_i][:] = val_i
    elif isinstance(ddf, dict):
        for col_i, val_i in columns:
            ddf[col_i][row] = val_i
        
def ddf_get_entry(ddf, i, columns):

    if isinstance(ddf, dict):
        if isinstance(columns, (tuple, list)):
            return [ddf[entry_i][i] for entry_i in columns]
        else:
            return ddf[columns][i]
    elif isinstance(ddf, pd.DataFrame):
        if isinstance(columns, (tuple, list)):
            return [ddf.loc[i, xi] for xi in columns]
        else:
            return ddf.loc[i, columns]
    else:
        raise TypeError('Not supported type %s'%(type(ddf)))
    
def ddf_get_colums(ddf, columns, to_numpy=True):

    if isinstance(ddf, dict):
        if isinstance(columns, (tuple, list)):
            return [ddf[xi] for xi in columns]
        else:
            return ddf[columns]
    elif isinstance(ddf, pd.DataFrame):
        if isinstance(columns, (tuple, list)):
            return [ddf[xi].to_numpy() for xi in columns]
        else:
            return ddf[columns].to_numpy()
    else:
        raise TypeError('Not supported type %s'%(type(ddf)))


# ss_model = ReadDoE_ss('linear.linear_system.uvlm.ss')

# path_model = '/mnt/work/Programs/RHEA/software/gemseo_rhea/examples/sharpy_smithwing/roms/'
# path_training = '/mnt/work/Programs/RHEA/software/gemseo_rhea/examples/sharpy_smithwing/roms/'
# path_testing = '/mnt/work/Programs/RHEA/software/gemseo_rhea/examples/sharpy_smithwing/roms/'

# ss_model.get_model_points(path_model)
# ss_model.get_training_points(path_training)
# ss_model.get_testing_points(path_testing)
# ss_model.build_testing_ss()
# # ss_model.doe_info['model_points']['df']

# interpol_settings = {'base_name': 'rb_multiquadric',
#                      'base_inputs': {},
#                      'scipy_inputs': {'fun_name': 'Rbf',
#                                       'function': 'multiquadric'}}

# balfreq_settings = {'frequency': 1.2,
#                     'method_low': 'trapz',
#                     'options_low': {'points': 12},
#                     'method_high': 'gauss',
#                     'options_high': {'partitions': 2, 'order': 8},
#                     'check_stability': False}

# # for i in range(36):
# #     ss_i = libss.StateSpace(ss_model.model_data[0][i],
# #                         ss_model.model_data[1][i],
# #                         ss_model.model_data[2][i],
# #                         ss_model.model_data[3][i],
# #                         ss_model.model_data[4][i])

# #     SSb, hsv, T, Ti, Zc, Zo, U, Vt = librom.balfreq(ss_i, balfreq_settings)
# #     print(i,len(hsv))
# N=30    
# model_ddf = build_balfreq(ss_model.dict_model_data, balfreq_settings)
# model_ddf2 = truncate_balfreq(model_ddf, N)
# model_interpol = Interpolation_ss(ss_model.model_points,
#                                   np.ones((ss_model.num_model_points,1)),
#                                   **interpol_settings)

# ss_int = [[] for i in range(ss_model.num_testing_points)]
# hsv_int = [[] for i in range(ss_model.num_testing_points)]
# ss_intt = [[] for i in range(ss_model.num_testing_points)]
# hsv_intt = [[] for i in range(ss_model.num_testing_points)]

# for ti in range(ss_model.num_testing_points):
#     interpolation_factors = model_interpol.interpol_weight_factors(
#         ss_model.testing_points[ti, :])
#     ss_int[ti], hsv_int[ti] = build_balfreq_interpol(model_ddf, interpolation_factors)
#     ss_intt[ti], hsv_intt[ti] = build_balfreq_interpol(model_ddf2, interpolation_factors, N=N,
#                                             system_labels=['At','Bt', 'Ct', 'Dt', 'dt'],
#                                             svd_labels=['U_hsvt', 'hsv_Vtt', 'U_hsv_Vtt'])

# error_metric = Error_metric()
# wv = np.array([0., 0.3, 0.65, 1.])
# e_f, e_avg = error_metric.freqresp(wv, testing_points=ss_model.testing_points,
#                                    ss_models=ss_int,
#                                    ss_testing=ss_model.testing_ss,
#                                    settings={})

# e_ft, e_avgt = error_metric.freqresp(wv, testing_points=ss_model.testing_points,
#                                    ss_models=ss_intt,
#                                    ss_testing=ss_int,
#                                    settings={})

    
class Error_metric:
    """
    Documentation for Error_metric
    """

    def __init__(self, err_settings={}):

        self.err_settings = err_settings

    def err_vect(self, err_v, _type='norm'):
        """
        Given a vector of error metrics err_v with length the number of samples, 
        compute the total error
        """

        if _type == 'norm':
            err = scipy.linalg.norm(err_v)/len(err_v)
        elif _type == 'max':
            err = np.max(np.abs(err_v))
        elif _type == 'avg':
            err = np.sum(np.abs(err_v))/len(err_v)
        return err
    
    def err_matrix(self, _A, A, err_tolerance=1e-9, _type='norm'):

        num_samples = len(_A)
        if len(np.shape(_A)) == 2:
            _A = [_A]
            A = [A]
            num_samples = 1

        err = []
        for i, Ai in enumerate(_A):
            if _type == 'norm':
                if scipy.linalg.norm(A[i]) > err_tolerance:
                    err.append(scipy.linalg.norm((Ai-A[i])
                                                 /scipy.linalg.norm(A[i]), **{}))
                else:
                    err.append(0.)
            else:
                n_ix, n_jx = np.shape(Ai)
                err_local = []
                for ix in range(n_ix):
                    for jx in range(n_jx):
                        if abs(Ai[ix,jx]) > err_tolerance:
                            err_local.append((Ai[ix,jx]-A[ix,jx])/Ai[ix,jx])
                        else:
                            err_local.append(0.)
                err.append(self.err_vect(err_local, _type=_type))
        if num_samples == 1:
            return err[0]
        else:
            err_avg = self.err_vect(err, _type=_type)
            return err, err_avg

    # def err_matrix2(self, _A, A, err_tolerance=1e-8, _type='norm'):

    #     err = []
    #     nx, ny = np.shape(_A)
    #     for i, in range():
    #         if _type == 'norm':
    #             if scipy.linalg.norm(Ai) > err_tolerance:
    #                 err.append(scipy.linalg.norm((Ai-A[i])
    #                           /scipy.linalg.norm(Ai), **self.err_settings))
    #             else:
    #                 err.append(0.)
    #         else:
    #             n_ix = len(Ai)
    #             err = []
    #             for ix in range(n_ix):
    #                 if self.err_vect(Ai[ix]) > err_tolerance:
    #                     err.append(self.err_vect(Ai[ix]-A[ix])/self.err_vect(Ai[ix]))
    #                 else:
    #                     err.append(0.)
    #     err_avg = self.err_vect(err, _type=_type)
    #     return err, err_avg
        
    def freqresp(self, wv, testing_points, ss_models, ss_testing, settings={}):

        num_freqpoints = len(wv)
        num_samples = len(testing_points)
        self.err_freq = np.zeros((num_samples, num_freqpoints))
        for i in range(num_samples):
            X = testing_points[i]
            y_model = ss_models[i].freqresp(wv)
            y_testing = ss_testing[i].freqresp(wv)
            for j in range(num_freqpoints):
                self.err_freq[i, j] = (scipy.linalg.norm(y_model[:, :, j]-y_testing[:, :, j])
                /scipy.linalg.norm(y_model[:, :, j]))
                # self.err_freq[i, j] = self.err_matrix(y_model[:, :, j],
                #                                       y_testing[:, :, j],
                #                                       **settings)
        err = self.err_freq[:, 0]
        err_avg = scipy.linalg.norm(self.err_freq)/(num_freqpoints*num_samples)
        return err, err_avg
    


# loads parametric space
# laods data 
# create interpolation function for

class Approximation_data(Bases):
    """
    Documentation for Approximation_SS
    """
    def __init__(self,
                 model_points,
                 model_data,
                 base_name,
                 base_inputs={},
                 base_index={},
                 F_inter_inv=None,
                 *args, **kwargs):

        self.model_points = model_points
        self.num_model_points = len(model_points)
        self.model_data = model_data
        if isinstance(base_name, str):
            self.base_name = [base_name]
        elif isinstance(base_name, list):
            self.base_name = base_name
        self.num_kernels = len(self.base_name)
        self.base_fun = [getattr(self, self.base_name[i]) for i in
                         range(self.num_kernels)]
        if isinstance(base_inputs, dict):
            self.base_inputs = [base_inputs]
        elif isinstance(base_inputs, list):
            self.base_inputs = base_inputs
        if isinstance(base_index, dict):
            self.base_index = [base_index]
        elif isinstance(base_index, list):
            self.base_index = base_index
        for i in range(self.num_kernels):
            self.base_inputs[i].update(points=self.model_points)
        self.F_inter_inv = F_inter_inv    
        super(Approximation_data, self).__init__(model_points,
                                                 model_data,
                                                 self.base_name[0],
                                                 self.base_inputs[0],
                                                 *args,
                                                 **kwargs)
        self.approx_build()
        
    def approx_build(self):

        pass
        # if not isinstance(self, 'fun_jac'):
        #     self.fun_jac = None
        # if not isinstance(self, 'fun_hess'):
        #     self.fun_hess = None

    @staticmethod
    def unpack_x(x, base_index, num_points):

        num_kernels = len(base_index)
        x_unp = [dict() for i in range(num_kernels)]
        counter = 0
        for i in range(num_kernels):
            for k, v in base_index[i].items():
                if v > 1:
                    xu = x[counter:counter + num_points*v]
                    x_unp[i][k] = np.reshape(xu,(num_points, v))
                else:
                    x_unp[i][k] = x[counter:counter + num_points*v]
                counter += num_points*v
        return x_unp
        
    def kernel_approx(self, x):
        """
        Kernel of functions
        """
        
        x_unp = self.unpack_x(x, self.base_index, self.num_model_points)
        self.bases_appr = [[] for i in range(self.num_kernels)]
        for i in range(self.num_kernels):
            self.base_inputs[i].update(x_unp[i])
            #self.base_inputs[i].update(points=self.model_points)
            self.bases_appr[i] = self.base_fun[i](**self.base_inputs[i])

    def approx_factors(self, X, x):
        """
        Weights factor that multiplies the input data as another mean
        to build the interpolation
        """
        
        self.kernel_approx(x) # kernel bases updated with x
        self.Wf_appr = []     # approximation weight factors
        for Xi in X:          # cycle through training points 
            self.Wf_appri = np.zeros(self.num_model_points)
            for j in range(self.num_model_points):
                for i in range(self.num_model_points):
                    for ki in range(self.num_kernels):
                        self.Wf_appri[j] += (self.bases_appr[ki][i](Xi)
                                              *self.F_inter_inv[i, j])
            self.Wf_appr.append(self.Wf_appri)

    def approx_model(self, x, X):

        pass
        # A = np.zeros_like(self.model_data[0])
        # self.kernel_approx(x)
        # for ik in range(self.num_kernels):
        #     for j in range(self.num_model_points):
        #         A += self.bases_appr[ik][j](X)*self.model_bases[j]
        # return A
    
    # def fun(self, x, *args):

    #     pass

    # def fun_error(self, err, _yi, yi):

    #     pass

    # def fun_jac(self):
    #     pass

    # def fun_hess(self):
    #     pass

    # def optimize(self, approach='minimize', **kwargs):

    #     opt = getattr(scipy.optimize, approach)
    #     self.res = opt(self.fun, **kwargs)

    # def test_opt_techniques(self, approaches):

    #     pass

# ss_int = [[] for i in range(ss_model.num_testing_points)]
# hsv_int = [[] for i in range(ss_model.num_testing_points)]
# ss_intt = [[] for i in range(ss_model.num_testing_points)]
# hsv_intt = [[] for i in range(ss_model.num_testing_points)]

# for ti in range(ss_model.num_testing_points):
#     interpolation_factors = model_interpol.interpol_weight_factors(
#         ss_model.testing_points[ti, :])
#     ss_int[ti], hsv_int[ti] = build_balfreq_interpol(model_ddf, interpolation_factors)
#     ss_intt[ti], hsv_intt[ti] = build_balfreq_interpol(model_ddf2, interpolation_factors, N=N,
#                                             system_labels=['At','Bt', 'Ct', 'Dt', 'dt'],
#                                             svd_labels=['U_hsvt', 'hsv_Vtt', 'U_hsv_Vtt'])

# error_metric = Error_metric()
# wv = np.array([0., 0.3, 0.65, 1.])
# e_f, e_avg = error_metric.freqresp(wv, testing_points=ss_model.testing_points,
#                                    ss_models=ss_int,
#                                    ss_testing=ss_model.testing_ss,
#                                    settings={})

# e_ft, e_avgt = error_metric.freqresp(wv, testing_points=ss_model.testing_points,
#                                    ss_models=ss_intt,
#                                    ss_testing=ss_int,
#                                    settings={})
# base_name = ['rb_inverse','pb_polyfull']
# base_inputs = [{'points':None},{'points':None, 'dim':2, 'degree':2, 'coeff':None}]
# base_index = [{'a':1,'b':1,'c':1}, {'coeff':6}]

# interpol_settings = {'base_name': 'rb_multiquadric',
#                      'base_inputs': {},
#                      'scipy_inputs': {'fun_name': 'Rbf',
#                                       'function': 'inverse'}}

class Approx_freqresp(Approximation_data, Interpolation_ss):
    """
    Documentation for Approx_interpol
    """
    
    def __init__(self, ss_model,
                 interpol_settings, appr_settings, opt_settings={},err_settings={},
                 fun_settings={}, model_data = None, x0_in=None, *args, **kwargs):
        
        self.ss_model = ss_model
        #self.num_model_points = len(ss_model.model_points)
        #self.num_training_points = len(ss_model.training_points)
        self.interpol_settings = interpol_settings
        self.appr_settings = appr_settings
        #import pdb; pdb.set_trace();
        
        # self.opt_settings = {'method': 'Nelder-Mead',
        #                      'approach': 'minimize'}
        self.opt_settings = {'approach': 'minimize'}
        self.opt_settings.update(opt_settings)
        self.err_settings = err_settings
        self.fun_settings = fun_settings
        if 'fun_args' in self.fun_settings:
            self.fun_args = fun_settings.pop('fun_args')
        else:
            self.fun_args = ()
        if model_data is None:
            self.model_data = np.ones((self.ss_model.num_model_points,1))
        elif isinstance(model_data, np.ndarray):
            self.model_data = model_data
        # self.appr_model = Approximation_data(model_points=4,
        #                                      model_data=self.mock_data,
        #                                      **self.appr_settings)
        if not hasattr(self, 'fun_jac'):
            self.fun_jac = None
        if not hasattr(self, 'fun_hess'):
            self.fun_hess = None
        kwargs.update(appr_settings)
        kwargs.update(interpol_settings)
        self.error_metric = Error_metric(err_settings)
        super(Approx_freqresp, self).__init__(self.ss_model.model_points,
                                              self.model_data,
                                              *args,
                                              **kwargs)
        if x0_in:
            self.x0_in = x0_in
        else:
            self.x0_in = self.build_ic()

    def build_ic(self):

        args1 = []
        for i, ni in enumerate(self.appr_settings['base_name']):
            f1 = getattr(self, ni)
            if i==0:
                args1 += f1(out_inputs=True,**self.basei_inputs)
            else:
                args1 += f1(out_inputs=True, **self.appr_settings['base_inputs'][i])
            
        return np.hstack(args1)
    
    def run(self, use_fun='fun1'):
        
        self.fun = getattr(self, use_fun)
        if self.fun_jac:
            self.optimize(x0=self.x0_in,
                          args=self.fun_args,
                          jac=self.fun_jac,
                          hess=self.fun_hess,
                          **self.opt_settings)
        else:
            self.optimize(x0=self.x0_in,
                          args=self.fun_args,
                          **self.opt_settings)
        
    def optimize(self, approach='minimize', **kwargs):

        opt = getattr(scipy.optimize, approach)
        self.res = opt(self.fun, **kwargs)

                
    def fun1(self, x, *args):

        system_labels = self.fun_settings['system_labels']
        svd_labels = self.fun_settings['svd_labels']
        N = self.fun_settings['N']
        freq_points = self.fun_settings['freq_points']
        self.approx_factors(self.ss_model.training_points, x)
        ss = [[] for ii in range(self.ss_model.num_training_points)]
        hsv_intt = [[] for ii in range(self.ss_model.num_training_points)]
        for ti in range(self.ss_model.num_training_points):
            ss[ti], hsv_intt[ti] = build_balfreq_interpol(self.ss_model.dict_model_data,
                                                          self.Wf_appr[ti],
                                                          N,
                                                          system_labels,
                                                          svd_labels)

        e_ft, err = self.error_metric.freqresp(freq_points,
                                               testing_points=self.ss_model.training_points,
                                               ss_models=ss,
                                               ss_testing=self.ss_model.training_ss)

        return err

    def fun2(self, x, *args):

        freq_points = self.fun_settings['freq_points']
        self.approx_factors(self.ss_model.training_points, x)
        G = [[] for ii in range(self.ss_model.num_training_points)]
        for ti in range(self.ss_model.num_training_points):
            G[ti] = np.zeros_like(self.ss_model.model_G[0])
            for mi in range(self.ss_model.num_model_points):
                G[ti] += (self.Wf_appr[ti][mi] *
                          self.ss_model.model_G[mi])
        #import pdb; pdb.set_trace();
        try:
            G_norm=scipy.linalg.norm(G)
            e_ft, err = self.error_metric.err_matrix(G,
                                                     self.ss_model.training_G)
            if self.err_settings['error_type'] == 'max':
                return np.max(e_ft)
            if self.err_settings['error_type'] == 'norm':
                return err
        except ValueError:
            err = 100
            #print('##### %s'%x)
            #G = np.zeros_like(self.ss_model.training_G)        
            #import pdb; pdb.set_trace();
            return err
    
def reshape_tensor(tensor):

    shape = np.shape(tensor)
    num_samples = shape[0]
    vect_size = np.prod(shape[1:])
    matrix = np.zeros((num_samples, vect_size))
    for i in range(num_samples):
        matrix[i] = np.reshape(tensor[i],[vect_size])
        
    return matrix

def shape_tensor_back(matrix, shape):

    #num_samples = shape[0]
    #assert len(matrix) == num_samples, "matrix[0] != shape[0]"
    tensor = np.reshape(matrix, shape)
    return tensor

# def reshape_tensor0(tensor0, entries, entries_size):

#     tensor = np.zeros(list(entries_size)+list(np.shape(tensor0)[1:]))
#     for i in range(len(tensor0)):
#         tensor[tuple(entries[i])] = tensor0[i]
#     return tensor

# def reshape_tensor(tensor0, entries, entries_size):

#     num_samples = len(tensor0)
#     for i_A, Ai in enumerate(tensor0):
#         nx, ny = np.shape(Ai)
#         for i in range(nx):
#             for j in range(ny):
#                 M[i_A].append(Ai[i,j])
#     tensor = np.zeros(list(entries_size)+list(np.shape(tensor0)[1:]))
#     for i in range(len(tensor0)):
#         tensor[tuple(entries[i])] = tensor0[i]
#     return tensor


# l=[('A', 'a'),  ('A', 'b'), ('B','a'),  ('B','b')]
# df = pd.DataFrame(np.random.randn(5,4), columns = l)
# df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Caps','Lower'])
