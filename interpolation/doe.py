import pickle
import os
import numpy as np
import pandas as pd
import pdb
import glob
import itertools
import functools
import matplotlib.pyplot as plt
import sharpy.linear.src.libsparse
import sharpy.linear.src.libss as libss

class DoE:

    def __init__(self,
                 samples=0,
                 dimensions=0,                 
                 bounds=[],
                 variables_list=None,
                 samples_list=None,
                 **kwargs):
        
        if variables_list:
            variables = [[] for i in range(len(variables_list[0]))]
            for i in range(len(variables)):
                for j in range(len(variables_list)):
                    variables[i].append(variables_list[j][i])
            self.variables_names = range(len(variables_list))
        elif samples_list:
            variables = samples_list
            self.variables_names = range(len(samples_list[0]))
        elif kwargs:
            variables = kwargs
            self.variables_names = kwargs.keys()
        else:
            variables = None
            
        if variables:
            self.num_dim = len(variables)
            self.num_samples = [len(variables[i]) for i in self.variables_names]
        else:
            self.num_dim = dimensions
            self.num_samples = samples
            self.variables_names = range(dimensions)
            
        self.variables = variables
        if len(np.shape(bounds)) == 1 and bounds:
            self.bounds = [bounds for i in range(self.num_dim)]
        else:
            self.bounds = bounds

    def __call__(self, _type):

        if (_type == 'full_factorial' or
            _type == 'ff'):
            self._full_factorial()
            return self.doe
        elif (_type == 'latin_hypercube' or
              _type == 'lhc'):
            self._lhc()
            return self.doe
        
    def _full_factorial(self):

        if not self.variables:
            if isinstance(self.num_samples, int):
                self.variables_ff = np.array([np.linspace(0, 1, self.num_samples)
                                           for i in range(self.num_dim)])
            else:
                self.variables_ff = np.array([np.linspace(0, 1, self.num_samples[i])
                                           for i in range(self.num_dim)])
        else:
            if isinstance(self.variables, dict):
                self.variables_ff = [self.variables[k] for k in self.variables.keys()]
            else:  
                self.variables_ff = np.array(self.variables).T
        self.doe = np.array(list(itertools.product(*self.variables_ff)))
        if self.bounds:   
            self._bounds_transformation()
            
    def _lhc(self):

        if self.variables:
            if isinstance(self.variables, dict):
                self.doe = np.array([self.variables[k] for k in self.variables.keys()]).T
            else:  
                self.doe = np.array(self.variables)
        else:
            self.doe = lhsmdu.sample(self.num_dim, self.num_samples)# (variables, samples)
            self.doe = self.doe.T
        if self.bounds:
            self._bounds_transformation()
            
    def _bounds_transformation(self):

        for i in range(self.num_dim):
            self.doe[:, i] = self.bounds[i][0] + self.doe[:, i]*(self.bounds[i][1]
                                                                 -self.bounds[i][0])
    @property
    def doe_dic(self):
        dic = dict(zip(self.variables_names, self.doe.T))
        return dic

    def built_dataframe(self, write_file=''):

        self.df = pd.DataFrame.from_dict(self.doe_dic)
        if write_file:
            self.df.to_csv(write_file)

    def update_dataframe(self, column, row=None, x=None, astype=0.):
        
        if column not in self.df.columns:
            self.df[column] = astype
        if x:
            self.df.at[row, column] = x

    def write_dataframe(self, _file, _type, *args):

        pd_fun = getattr(self.df,'to_'+_type)
        pd_fun(_file, *args)


def spiral_recursive(x0, points_dim, points_list=[]):
    """
    Creates an spiral Latin HyperCube designs of experiments 
    by recursively calling itself

    Args:
        x0 (np.ndarray): initial point coordinates
        points_dim (int): total number of points in the DoE
        points_list (list): initial point in the list of points, to be left
        at [0 for i in range(num_dim)]

    Returns:
        points_list (np.ndarray): DoE of shape(points_dim, num_dimenstions)
    """ 
    num_dim = len(x0)

    cycle_points1 = [i for i in range(1,num_dim+1)]
    cycle_points2 = [i for i in (points_dim-1) - np.arange(num_dim)]
    num_cycle_points = 2*num_dim
    points = np.zeros((num_cycle_points, num_dim))
    for di in range(num_dim):
        points[:,di] = cycle_points1[0:di] + cycle_points2 + cycle_points1[di:]
        points[:,di] += x0[di]
    if points_dim-2*num_dim-1 <= 0:
        points = points[:points_dim-1]
        points_list = np.concatenate((points_list, points))
    else:
        points_list = np.concatenate((points_list, points))
        points_list = spiral_recursive(points[-1], points_dim-2*num_dim, points_list)

    return points_list


class ReadDoE:

    def __init__(self, data_string, get_points_from_csv=True, data_string2=None):

        self.data_string = data_string
        self.get_points_from_csv = get_points_from_csv
        if not isinstance(self.data_string, list):
            self.data_string = [self.data_string]
        self.num_data_in = len(self.data_string)
        self.doe_info = dict()
        self.model_points = None
        self.model_data = None
        self.training_points = None
        self.training_data = None
        self.testing_points = None
        self.testing_data = None
        self.data_in = {}
        if data_string2 is None:
            self.data_string2 = []
        elif isinstance(data_string2, str):
            self.data_string2 = [data_string2]
        elif isinstance(data_string2, list):
            self.data_string2 = data_string2
            
    def get_model_data(self, label, _folder=None, _file=None, save_datain=True):

        self.doe_info[label] = dict()
        if self.get_points_from_csv:
            if not _file:
                _file = glob.glob(_folder+'/*.csv')[0]
            elif not _folder:
                _folder = '/'.join(_file.split('/')[:-1])
            self.df = pd.read_csv(_file)
            model_points = self.df.to_numpy()
            # model_points = model_points[:,columns[0]:columns[1]]
            self.doe_info[label]['df'] = self.df
        else:
            model_points = np.array(_file)
        sharpy_files = glob.glob(_folder+'/*/*.sharpy')
        sharpy_model = sharpy_files[0].split('/')[-1]
        sharpy_model = '_'.join(sharpy_model.split('_')[:-1])
        self.doe_info[label]['sharpy_files'] = sharpy_files
        self.doe_info[label]['sharpy_model'] = sharpy_model
        model_data = [[] for j in range(self.num_data_in)]
        self.data_in[label] = []
        for i in range(len(self.df)):
            file_pickle = glob.glob(_folder +'/'+ sharpy_model + '_%s'%i +
                                    '/**/*.pkl', recursive=True)[0]
        
            with open(file_pickle, 'rb') as file1:
                data = pickle.load(file1)
            if self.data_string2:
                for j in range(len(self.data_string2)):
                    self.data_in[label].append(functools.reduce(getattr, [data]+self.data_string2[j].split('.')))
            for j in range(self.num_data_in):    
                A = functools.reduce(getattr, [data]+self.data_string[j].split('.'))
                if isinstance(A, sharpy.linear.src.libsparse.csc_matrix):
                    A = A.todense()
                model_data[j].append(A)

        return model_points, model_data

    def to_df():
        pass
    
    def to_dict(self, model_data):

        dict_data = {self.data_string[i].split('.')[-1]: model_data[i]
                     for i in range(self.num_data_in)}
        return dict_data
    
    def get_model_points(self, _folder=None, _file=None, columns=[], rows=[]):

        self.model_points, self.model_data = self.get_model_data('model_points',
                                                                 _folder, _file)
        if columns:
            self.model_points = self.model_points[:,columns[0]:columns[1]]
        if rows:
            self.model_points = self.model_points[rows[0]:rows[1], :]
        self.num_model_points = len(self.model_points)
        self.dict_model_data = self.to_dict(self.model_data)
        
    def get_training_points(self, _folder=None, _file=None, columns=[], rows=[]):

        self.training_points, self.training_data = self.get_model_data('training_points',
                                                                       _folder, _file)
        if columns:
            self.training_points = self.training_points[:,columns[0]:columns[1]]
        if rows:
            self.training_points = self.training_points[rows[0]:rows[1], :]
        self.num_training_points = len(self.training_points)
        self.dict_training_data = self.to_dict(self.training_data)
        
    def get_testing_points(self, _folder=None, _file=None, columns=[], rows=[]):

        self.testing_points, self.testing_data = self.get_model_data('testing_points',
                                                                     _folder, _file)
        if columns:
            self.testing_points = self.testing_points[:,columns[0]:columns[1]]
        if rows:
            self.testing_points = self.testing_points[rows[0]:rows[1], :]

        self.num_testing_points = len(self.testing_points)
        self.dict_testing_data = self.to_dict(self.testing_data)
        
class ReadDoE_ss(ReadDoE):

    def __init__(self, data_string):
        
        data_string2 = data_string
        data_string = [data_string+'.A',
                       data_string+'.B',
                       data_string+'.C',
                       data_string+'.D',
                       data_string+'.dt']
        
        super().__init__(data_string, data_string2=data_string2)

    def build_model_ss(self, freqpoints=None):
        
        self.model_ss = []
        self.model_G = []
        if self.num_model_points > 0:
            for i in range(self.num_model_points):
                self.model_ss.append(libss.StateSpace(
                    self.model_data[0][i],
                    self.model_data[1][i],
                    self.model_data[2][i],
                    self.model_data[3][i],
                    dt=self.model_data[4][i]))
                if freqpoints is not None:
                    self.model_G.append(self.model_ss[i].freqresp(freqpoints))

    def build_training_ss(self, freqpoints=None):

        self.training_ss = []
        self.training_G = []
        if self.num_training_points > 0:
            for i in range(self.num_training_points):
                self.training_ss.append(libss.StateSpace(
                    self.training_data[0][i],
                    self.training_data[1][i],
                    self.training_data[2][i],
                    self.training_data[3][i],
                    dt=self.training_data[4][i]))
                if freqpoints is not None:
                    self.training_G.append(self.training_ss[i].freqresp(freqpoints))

    def build_testing_ss(self, freqpoints=None):
        
        self.testing_ss = []
        self.testing_G = []
        if self.num_testing_points > 0:
            for i in range(self.num_testing_points):
                self.testing_ss.append(libss.StateSpace(
                    self.testing_data[0][i],
                    self.testing_data[1][i],
                    self.testing_data[2][i],
                    self.testing_data[3][i],
                    dt=self.testing_data[4][i]))
                if freqpoints is not None:
                    self.testing_G.append(self.testing_ss[i].freqresp(freqpoints))
                
if  (__name__ == '__main__'):
    N = 50 # 50 points 
    P = spiral_recursive([0, 0], N, np.array([[0, 0]])) # 2D DoE
    #P = spiral_recursive([0, 0, 0, 0], N, np.array([[0, 0, 0, 0]])) # 4D DoE etc.

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(P[:,0], P[:,1], marker='.', c='b', s=100)

    plt.plot(P[:,0], P[:,1], 'r--', linewidth=0.5)
 
    major_ticks = np.arange(0, N+1, 1)
    minor_ticks = np.arange(0, N+1, N-1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=1)
    ax.grid(which='major', alpha=0.2)

    plt.show()
    
