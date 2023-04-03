import numpy as np
import argparse
import pyNastran.op4.op4 as op4
from pyNastran.bdf.bdf import BDF
from scipy.io import savemat
import pathlib

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--model_path", const=None, type=str)
parser.add_argument("-g", "--gafs_file", const=None, type=str)

args = parser.parse_args()

model_path = pathlib.Path(parser.model_path)
gafs_file = pathlib.Path(parser.gafs_file)
#model_path = "/mnt/work/Programs/RHEAtools/data/in/SOLGAFs/polimi-145cam_078M.bdf"
#gafs_file = "Qhh10-078.op4"
model = BDF()
model.read_bdf(model_path)
matrices = op4.read_op4(model_path.parent / gafs_file)
num_matrices = matrices['Q_HH'][0]
QHH = matrices['Q_HH'][1]

reduced_freqs = []
for mki in model.mkaeros:
    [reduced_freqs.append(i) for i in mki.reduced_freqs]
reduced_freqs = np.array(reduced_freqs)

mdic = {"QHH": QHH, "reduced_freqs": reduced_freqs}
savemat("matlab_matrix.mat", mdic)
