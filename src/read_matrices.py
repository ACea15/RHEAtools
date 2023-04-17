import numpy as np
import argparse
import pyNastran.op4.op4 as op4
from pyNastran.bdf.bdf import BDF
from scipy.io import savemat
import pathlib

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--model_path", const=None, type=str)
parser.add_argument("-g", "--gafs_file", const=None, type=str)
parser.add_argument("-n", "--nastran_file", const=None, type=str)
parser.add_argument("-f", "--freqs_file", const=None, type=str)

args = parser.parse_args()
model_path = pathlib.Path(args.model_path)
#model_path = "/mnt/work/Programs/RHEAtools/data/in/SOLGAFs/polimi-145cam_078M.bdf"
#gafs_file = "Qhh10-078.op4"
print('Reading BDF to get reduced frequencies')
model = BDF()
model.read_bdf(model_path / args.nastran_file)
print('Reading OP4 where GAFs are located')
matrices = op4.read_op4(model_path / args.gafs_file)
num_matrices = len(matrices['Q_HH'][0])
QHH = matrices['Q_HH'][1]
print('Reading natural frequencies')
natural_frequencies = np.load(args.freqs_file)
reduced_freqs = []
for mki in model.mkaeros:
    [reduced_freqs.append(i) for i in mki.reduced_freqs]
reduced_freqs = np.array(reduced_freqs)
# print(len(reduced_freqs))
# print(num_matrices)
assert len(reduced_freqs) == num_matrices, "freqs and number of gafs not equal"
print("writing GAFs and reduced freqs. to {}".format(model_path / "matlab_gafs.mat"))
mdic = {"QHH": QHH, "reduced_freqs": reduced_freqs,
        "natural_frequencies": natural_frequencies}
savemat(model_path / "matlab_gafs.mat", mdic)
