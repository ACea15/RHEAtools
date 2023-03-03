import numpy as np
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import nastran_extraction as nastran_ex

##################################################
model = BDF()
#model.read_bdf("./models/nastran/")
model.read_bdf("./SOL103/polimi-103cam.bdf")

print(model.get_bdf_stats())

file_name  = "./SOL103/polimi-103cam.op2"
file_name  = "./modify_pbeamsL15_0/sol145.op2"
op2 = OP2()
# op2.set_additional_matrices_to_read({b'OPHP': False, b'OPHKS':False})
op2.read_op2(file_name)
eig1=op2.eigenvectors[1]
modes=eig1.data

eigenvector_keys = list(op2.eigenvectors.keys())
print("loadcases = %s" % eigenvector_keys)

# get subcase 1
eig1 = op2.eigenvectors[1]

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
                  vtail=dict(rbe2_ids=rbe2_vtail,
                             tolerance=0.2,
                             plane_vector=[0,1,0.],
                             symmetry_index=None
                             ),
                  htail=dict(rbe2_ids=rbe2_htail,
                             tolerance=0.2,
                             plane_vector=[0,0,1.],
                             symmetry_index=10
                             ),
                  )


m1 = nastran_ex.Model(model, components= components, modes=modes,
                      modal_node_ids=node_ids, modes_scaling=50)
#panels_chord = [4.21, 4.21, 3.47, 2.41, 1.6]
#panels_y = [0, 2.04, 8.61, 18.06, 27.57]
#chords = (m1.wing.trailing_edge_coord - m1.wing.leading_edge_coord)[:, 0]
#y_wing = m1.wing.trailing_edge_coord[:36,1]
#interpolation = np.interp(y_wing, panels_y, panels_chord)
#factors = interpolation / chords[:36]

nchord = 9#model.caeros[1001].nchord
#m1.wing.apply_shift([0,0,0.7])
#m1.strut.apply_shift([0,0,0.7])
#m1.wing.apply_transformation([1.55, np.hstack([np.linspace(3.5,2.5,36),
#                                                np.linspace(3.5,2.5,35)])])
#m1.strut.apply_transformation([0.7/0.29, 0.7/0.29])

#m1.wing.apply_transformation([1,1])
#m1.strut.apply_transformation([1,1])

m1.build_gridmesh(nchord,'Grid0')
m1.build_modalmesh(nchord, range(35),'M0cam_')

torsion = []
for mi in range(30):

    torsion.append(np.abs(m1.wing.beam_disp[mi][35][4]))

torsion = np.array(torsion)
####################################################

m2 = nastran_ex.Model(model, components= components, modes=modes,
                      modal_node_ids=node_ids, modes_scaling=50)
#panels_chord = [4.21, 4.21, 3.47, 2.41, 1.6]
#panels_y = [0, 2.04, 8.61, 18.06, 27.57]
#chords = (m1.wing.trailing_edge_coord - m1.wing.leading_edge_coord)[:, 0]
#y_wing = m1.wing.trailing_edge_coord[:36,1]
#interpolation = np.interp(y_wing, panels_y, panels_chord)
#factors = interpolation / chords[:36]

nchord = 9#model.caeros[1001].nchord
m2.wing.apply_shift([0,0,0.7])
m2.strut.apply_shift([0,0,0.7])
m2.wing.apply_transformation([1.55, np.hstack([np.linspace(3.5,2.5,36),
                                                np.linspace(3.5,2.5,35)])])
m2.strut.apply_transformation([0.7/0.29, 0.7/0.29])

#m1.wing.apply_transformation([1,1])
#m1.strut.apply_transformation([1,1])

m2.build_gridmesh(nchord,'Grid_cam')
m2.build_modalmesh(nchord, range(35),'Mcam_')
