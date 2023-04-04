import numpy as np
import pandas as pd
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import src.nastran_extraction as nastran_ex
import src.write_modes
import argparse
import src.filters

def filter_sigmoid(x):
    a = 1.
    b = 12.#6. #5
    c = 2.5 #1.85 # 2.
    return src.filters.filter_sigmoid(x, a, b, c)

###
parser = argparse.ArgumentParser()
parser.add_argument("-m","--num_modes", nargs='?', const=4, type=int)
parser.add_argument("-s","--scaling", nargs='?', const=1., type=float)
parser.add_argument("-f","--filtering", nargs='?', const=None, type=str)
parser.add_argument("-d","--dir2save", nargs='?', const=None, type=str)
parser.add_argument("-a","--markers_file", nargs='?', const=None, type=str)
parser.add_argument("-n","--nastran_file", nargs='?', const=None, type=str)

args = parser.parse_args()

if args.scaling is None:
    MODES_SCALING = 20.
else:
    MODES_SCALING = args.scaling
if args.num_modes is None:
    NUM_MODES = list(range(5))
else:
    NUM_MODES = list(range(args.num_modes))
if args.filtering is None:
    FILTERING = filter_sigmoid
else:
    FILTERING = locals()[args.filtering]
if args.dir2save is None:
    SAVE_DIR0 = f"./data/out/ONERA_fac{MODES_SCALING}/MeshDeformation/"
else:
    #SAVE_DIR0 = f"./data/out/ONERA_fac{MODES_SCALING}/{args.dir2save}/"
    SAVE_DIR0 = f"{args.dir2save}"
if args.markers_file is None:
    MARKERS_FILE = "./data/in/sbw_def0.txt"
    MARKERS_FILE = "./data/in/sbw_1901.txt"
else:
    #SAVE_DIR0 = f"./data/out/ONERA_fac{MODES_SCALING}/{args.dir2save}/"
    MARKERS_FILE = f"./data/in/{args.markers_file}"
if args.nastran_file is None:
    NASTRAN_FILE = "./data/in/SOL103/polimi-103cam.bdf"
    NASTRAN_FILE_OP2 = "./data/in/SOL103/polimi-103cam.op2"
else:
    NASTRAN_FILE = f"./data/in/{args.nastran_file}.bdf"
    NASTRAN_FILE_OP2 = f"./data/in/{args.nastran_file}.op2"

print("Modes scaling: %s" %MODES_SCALING)
print("Modes : %s" %NUM_MODES)
print("Filtering : %s" %FILTERING)
print("Directory : %s" %SAVE_DIR0)
print("MARKERS_FILE : %s" %MARKERS_FILE)

##################################################
model = BDF()
#model.read_bdf("./models/nastran/")
model.read_bdf(NASTRAN_FILE)

print(model.get_bdf_stats())

file_name  = NASTRAN_FILE_OP2
op2 = OP2()
op2.read_op2(file_name)
eig1=op2.eigenvectors[1]
modes=eig1.data

eigenvector_keys = list(op2.eigenvectors.keys())
print("loadcases = %s" % eigenvector_keys)

# get subcase 1
eig1 = op2.eigenvectors[1]
natural_freqs = np.array(eig1.mode_cycles) * 2 * np.pi
np.save(SAVE_DIR0 + "/natural_freqs.npy", natural_freqs)
#modes = eig1.modes
times = eig1._times #  the generic version of modes
#print("modes = %s\n" % modes)
print("times = %s\n" % times)

node_ids = eig1.node_gridtype[:, 0]
node_types = eig1.node_gridtype[:, 1]
####################################################

rbe2_ids,rbe3_ids,rbe3_dependent,idp2dp = nastran_ex.extract_rbes(model)
rbe2_wing = [i for i in rbe2_ids if str(i)[:2] == '20']
rbe2_strut = [i for i in rbe2_ids if str(i)[:2] == '50']
rbe2_vtail = [i for i in rbe2_ids if str(i)[:2] == '30']
rbe2_htail = [i for i in rbe2_ids if str(i)[:2] == '40']

#####################################################
components = dict(wing=dict(rbe2_ids=rbe2_wing,
                            tolerance=0.2,
                            plane_vector=[0,0,1.],
                            symmetry_index=36
                            ),
                  strut=dict(rbe2_ids=rbe2_strut,
                             tolerance=0.2,
                             plane_vector=[0,0,1.],
                             symmetry_index=24
                             ),
                  # vtail=dict(rbe2_ids=rbe2_vtail,
                  #            tolerance=0.2,
                  #            plane_vector=[0,1,0.],
                  #            symmetry_index=None
                  #            ),
                  # htail=dict(rbe2_ids=rbe2_htail,
                  #            tolerance=0.2,
                  #            plane_vector=[0,0,1.],
                  #            symmetry_index=10
                  #            ),
                  )


####################################################

m2 = nastran_ex.Model(model, components= components, modes=modes,
                      modal_node_ids=node_ids, modes_scaling=MODES_SCALING) # 50 before
#panels_chord = [4.21, 4.21, 3.47, 2.41, 1.6]
#panels_y = [0, 2.04, 8.61, 18.06, 27.57]
#chords = (m1.wing.trailing_edge_coord - m1.wing.leading_edge_coord)[:, 0]
#y_wing = m1.wing.trailing_edge_coord[:36,1]
#interpolation = np.interp(y_wing, panels_y, panels_chord)
#factors = interpolation / chords[:36]

nchord = 5#model.caeros[1001].nchord
m2.wing.apply_shift([0,0,0.7])
m2.strut.apply_shift([0,0,0.7])
m2.wing.apply_transformation([1.55, np.hstack([np.linspace(3.5,2.5,36),
                                               np.linspace(3.5,2.5,35)])])
m2.strut.apply_transformation([0.7/0.29, 0.7/0.29])

#m1.wing.apply_transformation([1,1])
#m1.strut.apply_transformation([1,1])

m2.build_gridmesh(nchord, save_file='Gridcam_half',
                  build_symmetric=False, save_dir=SAVE_DIR0 + "/DLMgrid/")
m2.build_modalmesh(nchord, range(35), save_file='Mcam_half',
                   build_symmetric=False, save_dir=SAVE_DIR0 + "/DLMgrid/")

Xv = np.vstack([m2.wing._points, m2.strut._points])
Xm = [np.vstack([m2.wing._modalpoints[mi],
                 m2.strut._modalpoints[mi]]) for mi in NUM_MODES]
# Xv = m2.wing._points[0::10]
# Xm = [m2.wing._modalpoints[k][0::10] for k in modes]
# Xv = m2.wing._points
# Xm = [m2.wing._modalpoints[k] for k in NUM_MODES]
df0 = pd.read_csv(MARKERS_FILE,
                  names=['id', 'x', 'y', 'z'],
                  comment="#",
                  index_col=False, sep=' ')
#print(Xm[0] - Xv)
in_parts = True
full_mesh = False
if (MARKERS_FILE == "./data/in/sbw_def0.txt" or
    MARKERS_FILE == "./data/in/sbw_0901.txt"):
    index_parts = 284698
elif (MARKERS_FILE == "./data/in/sbw_1901.txt"):
    index_parts = 219467
elif (MARKERS_FILE == "./data/in/sbw_1901FULL.txt"):
    df_combined = df0.drop_duplicates().set_index('id')
    in_parts = False
elif (MARKERS_FILE == "./data/in/sbw_1901MESHFULL.txt"):
    df_combined = df0.drop_duplicates().set_index('id')
    in_parts = False
    full_mesh = True
if in_parts:
    df_wing = df0.loc[:index_parts-1]
    df_strut = df0.loc[index_parts:]
    df_combined = pd.concat([df_wing.set_index('id'),
                             df_strut.set_index('id')]).drop_duplicates()

# modal_displacements, modal_coord = src.write_modes.calculate_interpolated_modes(Xv, Xm,
#                                                                                 df_combined.to_numpy(),
#                                                                                 df_combined.index.to_numpy())
modal_displacements, modal_coord = src.write_modes.calculate_interpolated_modes(Xv, Xm,
                                                                                df_combined.to_numpy(),
                                                                                df_combined.index.to_numpy(),
                                                                                filtering=FILTERING)
if full_mesh:
    src.write_modes.save_interpolated_modes(modal_coord,
                                            folder_name=SAVE_DIR0 + "/SU2_fullmesh/M",
                                            file_name="sbw_fordef.dat")
else:
    src.write_modes.save_interpolated_modes(modal_coord,
                                            folder_name=SAVE_DIR0 + "/SU2_mesh/M",
                                            file_name="sbw_fordef.dat")
#########################################################
# displacements_wing, coord_wing = src.write_modes.calculate_interpolated_modes(Xv, Xm, Xwing)
# displacements_strut, coord_strut = src.write_modes.calculate_interpolated_modes(Xv, Xm, Xstrut)
if in_parts:
    Xwing = df_wing[['x', 'y', 'z']].to_numpy()
    Xstrut = df_strut[['x', 'y', 'z']].to_numpy()
    displacements_wing, coord_wing = src.write_modes.calculate_interpolated_modes(Xv, Xm, Xwing,
                                                                                  filtering=FILTERING)
    displacements_strut, coord_strut = src.write_modes.calculate_interpolated_modes(Xv, Xm, Xstrut,
                                                                                    filtering=FILTERING)
    #print("#################")
    #print(displacements_wing[0])
    src.write_modes.save_interpolated_modes_parts(SAVE_DIR0 + "/SU2_mesh/",
                                                  "sbw_forHB.dat",
                                                  displacements_wing,
                                                  displacements_strut)

    src.write_modes.save_grid_parts(SAVE_DIR0 + "/SU2_mesh/",
                                    coord_wing[0] - displacements_wing[0],
                                    coord_strut[0] - coord_strut[0])
