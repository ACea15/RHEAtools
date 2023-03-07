import numpy as np
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
from pyNastran.f06 import parse_flutter as flut
import pathlib
import subprocess
from pyNastran.f06 import parse_flutter as flutter
import matplotlib.pyplot as plt
def shift_panels(model, caero_shift:dict):

    for k, v in caero_shift.items():
        model.caeros[k].p1 += np.array([model.caeros[k].x12*v, 0., 0.])
        model.caeros[k].p4 += np.array([model.caeros[k].x12*v, 0., 0.])

def shift_conm2s(model, chord, shift_chord, conm2_ids, conm2_symm):

    for k, v in conm2_symm.items():
        model.masses[conm2_ids[k]].X += np.array([chord * shift_chord[k], 0., 0.])
        model.masses[conm2_ids[v]].X += np.array([chord * shift_chord[k], 0., 0.])

def modify_pbeams(model, pbeam_ids, Afactor):
    for k in pbeam_ids:
        model.properties[k].A *= Afactor

def write_model(fileX, model, write_directly=False):
    if write_directly:
        model.write_bdf(f"{fileX}.bdf")
    else:
        with open(f"{fileX}.bdf", "w") as file1:
            executive_control = "$ EXECUTIVE CONTROL $"
            file1.write(executive_control)
            file1.write('\n')
            file1.write("NASTRAN QUARTICDLM = 1")
            file1.write('\n')        
            for line_i in model.executive_control_lines:
                file1.write(line_i)
                if line_i != '':
                    file1.write('\n')
            for line_i in model.case_control_deck.lines:
                file1.write(line_i)
                if line_i != '':
                    file1.write('\n')
            file1.write('BEGIN BULK')
            file1.write('\n')
            model.write_bulk_data(file1)

def run_nastran(fileX, file_path):
    result = subprocess.call([f"/msc/MSC_Nastran/2022.4/bin/nast20224 {fileX}.bdf batch=no"],
                             shell=True, executable='/bin/bash',cwd=file_path)
    # result = subprocess.run([f"/msc/MSC_Nastran/2022.4/bin/nast20224 {fileX}.bdf scr=yes old=no"],
    #                          shell=True, executable='/bin/bash',cwd=file_path)
    return result

def validate_interface(original_file, fileX):

    f0 = flutter.make_flutter_response(f"{original_file}.f06",
                                       {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
                                        'dynamic_pressure':'Pa'},
                                       {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
                                        'dynamic_pressure':'Pa'},
                                       )

    f1 = flutter.make_flutter_response(f"{fileX}.f06",
                                       {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
                                        'dynamic_pressure':'Pa'},
                                       {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
                                        'dynamic_pressure':'Pa'},
                                       )

    velocity = f1[1].results[:, :, f1[1].ivelocity]
    velocity_original = f0[1].results[:, :, f0[1].ivelocity]
    assert np.allclose(velocity,  velocity_original), "velocity not matching"
    freq = f1[1].results[:, :, f1[1].ifreq]
    freq_original = f0[1].results[:, :, f0[1].ifreq]
    assert np.allclose(freq, freq_original, 1e-3), "freq not matching"
    damping = f1[1].results[:, :, f1[1].idamping]
    damping_original = f0[1].results[:, :, f0[1].idamping]
    assert np.allclose(damping, damping_original, 5e-2), "damping not matching"

    # dict1 = dict(velocity=obj1.results[mi, :, obj1.ivelocity],
    #             eas=obj1.results[mi, :, obj1.ieas],
    #             kfreq=obj1.results[mi, :, obj1.ikfreq],
    #             freq=obj1.results[mi, :, obj1.ifreq],
    #             eigr=obj1.results[mi, :, obj1.ieigr],
    #             eifi=obj1.results[mi, :, obj1.ieigi],
    #             damping=obj1.results[mi, :, obj1.idamping]
    #             )

def stretch_strutchord(model, a_factor, pbeam_ids, caero_ids, **kwags):


    for k in pbeam_ids:
        model.properties[k].A *= a_factor**2
        model.properties[k].i1 *= a_factor**4
        model.properties[k].i2 *= a_factor**4
        
    for k in caero_ids:
        model.caeros[k].x12 *= a_factor
        model.caeros[k].x43 *= a_factor

    # for k, v in conm2_symm.items():
    #     model.masses[conm2_ids[k]].X += np.array([chord * shift_chord[k], 0., 0.])
    #     model.masses[conm2_ids[v]].X += np.array([chord * shift_chord[k], 0., 0.])


        
def parametric_factory(_range: list, parametric_function: callable,
                       original_file: str, folder_out, file_out, **kwargs):

    for si in _range:
        model = BDF()
        model.read_bdf(original_file + ".bdf")
        parametric_function(model, **kwargs)
        file_path = pathlib.Path(f"{folder_out}_{si}")
        file_path.mkdir(parents=True, exist_ok=True)
        fileX = file_path / file_out
        write_model(fileX, model)
        run_nastran(fileX, file_path)
    
def build_strut_shifting(shift_range, strut_panels, original_file_name,
                         file_name= "sol145", folder="/home/acea/runs/polimi/models/shift_panelsLM25"):

    for si in shift_range:
        model = BDF()
        model.read_bdf(original_file_name + ".bdf")
        shift_strut_dict = {k: si for k in strut_panels}
        shift_panels(model, shift_strut_dict)
        file_path = pathlib.Path(f"{folder}_{si}")
        file_path.mkdir(parents=True, exist_ok=True)
        fileX = file_path / file_name
        write_model(fileX, model)
        run_nastran(fileX, file_path)

def build_strut_conm2shifting(chord, shift_chord, conm2_ids, conm2_symm, original_file_name,
                              file_name= "sol145", folder="/home/acea/runs/polimi/models/shift_conm2sLM25"):

    for i, si in enumerate(shift_chord):
        model = BDF()
        model.read_bdf(original_file_name + ".bdf")
        shift_conm2s(model, chord, si, conm2_ids, conm2_symm)
        file_path = pathlib.Path(f"{folder}_{i}")
        file_path.mkdir(parents=True, exist_ok=True)
        fileX = file_path / file_name
        write_model(fileX, model)
        run_nastran(fileX, file_path)


def build_strut_pbeams(Afactors, pbeam_ids, original_file_name,
                       file_name= "sol145",
                       folder="/home/acea/runs/polimi/models/modify_pbeamsLM25"):

    
    for i, ai in enumerate(Afactors):
        model = BDF()
        model.read_bdf(original_file_name + ".bdf")
        modify_pbeams(model, pbeam_ids, ai)
        file_path = pathlib.Path(f"{folder}_{i}")
        file_path.mkdir(parents=True, exist_ok=True)
        fileX = file_path / file_name
        write_model(fileX, model)
        run_nastran(fileX, file_path)

def read_flutter(fileX):
    f1 = flutter.make_flutter_response(f"{fileX}.f06",
                                       {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
                                        'dynamic_pressure':'Pa'},
                                       {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
                                        'dynamic_pressure':'Pa'},
                                       )
    return f1[1]

def copy_raws(vector, dictionary):
    
    out = [{k: vi for k in dictionary.keys()} for vi in vector]
    return out

def conm2_coord(model, conm2s_ids):
    strut_conm2s_coord = []
    for k in conm2s_ids:
        X = model.masses[k].nid_ref.get_position()
        strut_conm2s_coord.append(X)
    strut_conm2s_coord = np.array(strut_conm2s_coord)
    return strut_conm2s_coord

def read145_f06(fileX):

    f1 = flutter.make_flutter_response(f"{fileX}.f06",
                                       {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
                                        'dynamic_pressure':'Pa'},
                                       {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
                                        'dynamic_pressure':'Pa'},
                                       )

    velocity = f1[1].results[:, :, f1[1].ivelocity]
    freq = f1[1].results[:, :, f1[1].ifreq]
    kfreq = f1[1].results[:, :, f1[1].ikfreq]
    damping = f1[1].results[:, :, f1[1].idamping]
    eigr = f1[1].results[:, :, f1[1].ieigr]
    eigi = f1[1].results[:, :, f1[1].ieigi]
    sol145 = type('sol145', (),
                  {'velocity': velocity,
                   'freq': freq,
                   'kfreq': kfreq,
                   'damping': damping,
                   'eigr': eigr,
                   'eigi': eigi,
                   'obj':f1[1]}
                  )
    return sol145

def fill_collector(local_vars, collector):

    for k in collector.keys():
        collector[k] = local_vars[k]

#fileX = "/home/acea/runs/polimi/models/shift_conm2s_1/sol145"
def calculate_flutter(fileX, Modes=None, collector=None):

    sol145 = read145_f06(fileX)
    if Modes is None:
        Modes = range(len(sol145.velocity))

    binary_damping = np.where(sol145.damping[Modes] > 0., 1, 0)
    flutter_indexes = [np.where(di == 1)[0] for di in binary_damping]
    flutter_modes = []
    for i, fi in enumerate(flutter_indexes):
        if len(fi) > 0:
            flutter_modes.append(fi[0])
        else:
            flutter_modes.append(np.inf)
    flutter_modes = np.array(flutter_modes)
    flutter_index = int(min(flutter_modes))
    flutter_mode = np.where(flutter_modes == flutter_index)[0]
    flutter_speeds = []
    for fi in flutter_mode:
        flutter_speed_i = (sol145.velocity[fi, flutter_index-1] + (sol145.velocity[fi, flutter_index]
                                                                   - sol145.velocity[fi, flutter_index - 1])/
                           (sol145.damping[fi, flutter_index] - sol145.damping[fi, flutter_index -1]) *
                           -sol145.damping[fi, flutter_index-1])
        flutter_speeds.append(flutter_speed_i)
    FlutterSpeed = min(flutter_speeds)
    FlutterMode = flutter_mode[flutter_speeds.index(FlutterSpeed)]

    if collector is not None:
        fill_collector(locals(), collector)
        collector["FlutterSpeed"] = FlutterSpeed
        collector["FlutterMode"] = FlutterMode
    else:
        return FlutterSpeed, FlutterMode

def build_flutter(main_folder, folders, file_name="sol145",  Modes=None, collector=None):

    results = dict()
    for fi in files:
        if collector is not None:
            collector_i = collector.copy()
        fileX = f"{main_folder}/{fi}/{file_name}"
        if collector is None:
            FlutterSpeed, FlutterMode = calculate_flutter(fileX, Modes)
            results[fi] = dict(FlutterSpeed=FlutterSpeed,
                               FlutterMode=FlutterMode)
        else:
            calculate_flutter(fileX, Modes, collector_i)
            results[fi] = collector_i
    return results

###########################################################
# Plotting
###########################################################

# main_folder = "/home/ac5015/pCloudDrive/Imperial/PostDoc/models_POLIMI/"
# files = [f"shift_conm2s2_{xi}" for xi in range(11)]
# files += [f"shift_panels2_{xi}" for xi in [-0.25, -0.2, -0.15, -0.1, 0.,  0.1, 0.15, 0.2, 0.25]]
# files += [f"shift_panels__{xi}" for xi in [0.1, 0.15, 0.2, 0.25]]

# collector_list = ['sol145']
# collector = {ci: None for ci in collector_list}
# results = build_flutter(main_folder, files, Modes=range(15), collector=collector)

# import plotly.express as px
# import pandas as pd

# df = pd.DataFrame({'flutter': [results["shift_conm2s2_%s" %i]["FlutterSpeed"] for i in range(11)],
#                    'flutter_mode': [results["shift_conm2s2_%s" %i]["FlutterMode"] for i in range(11)],
#                    'conm2_shifting': [0., -0.05, -0.1, -0.15, -0.2, -0.25, 0.05, 0.1, 0.15, 0.2, 0.25]
#                    })

# panels_shifting = [-0.25, -0.2, -0.15, -0.1, 0.,  0.1, 0.15, 0.2, 0.25]
# df2 = pd.DataFrame({'flutter': [results["shift_panels2_%s" %i]["FlutterSpeed"] for i in panels_shifting],
#                     'flutter_mode': [results["shift_panels2_%s" %i]["FlutterMode"] for i in panels_shifting],
#                     'panels_shifting': panels_shifting
#                    })

# panels_shifting = [0.1, 0.15, 0.2, 0.25]
# df3 = pd.DataFrame({'flutter': [results["shift_panels__%s" %i]["FlutterSpeed"] for i in panels_shifting],
#                     'flutter_mode': [results["shift_panels__%s" %i]["FlutterMode"] for i in panels_shifting],
#                     'panels_shifting': panels_shifting
#                    })

# fig = px.line(df2, x='panels_shifting', y='flutter')
# fig.show()

# results["shift_panels2_0.1"]['sol145'].obj.plot_vg_vf(modes=range(1, 15))
# results["shift_panels__0.1"]['sol145'].obj.plot_vg_vf(modes=range(1, 15))
# plt.show()

###########################################################
# Running
###########################################################

original_file_name  =  "/home/acea/runs/polimi/models/SOL145/polimi-145cam_078M"
validate_folder = "/home/acea/runs/polimi/models/validate_panels_shifting"
model0 = BDF()
model0.read_bdf(original_file_name + ".bdf")
strut_panels = [k for k in model0.caeros.keys() if k > 31000 and k < 37000]
VALIDATE = False
STRUT_SHIFTING_ANALYSIS = True
CONM2_SHIFTING = True
PBEAM = True
strut_conm2s = [k for k in model0.masses.keys() if model0.masses[k].eid in list(range(233,328+1))]
strut_pbeams = [k for k in model0.properties.keys() if model0.properties[k].Pid() in list(range(5000, 5022+1))] 
strut_conm2s_symmetric = {i: i+48 for i in range(2, 48)}
strut_conm2s_coord = conm2_coord(model0, strut_conm2s)
chord = model0.caeros[strut_panels[-1]].x12

if VALIDATE:
    shift_strut_dict = {k: 0. for k in strut_panels}
    shift_panels(model0, shift_strut_dict)
    file_path = pathlib.Path(validate_folder)
    file_path.mkdir(parents=True, exist_ok=True)
    fileX = file_path / "sol_145"
    write_model(fileX, model0)
    run_nastran(fileX, file_path)
    validate_interface(original_file_name, fileX)
if STRUT_SHIFTING_ANALYSIS:
    shift_range = [-0.25, -0.2, -0.15, -0.1, 0.,  0.1, 0.15, 0.2, 0.25]
    build_strut_shifting(shift_range, strut_panels, original_file_name, file_name= "sol145",
                         folder="/home/acea/runs/polimi/models/shift_panelsLM25")
if CONM2_SHIFTING:
    shift_chord = copy_raws([0., -0.05, -0.1, -0.15, -0.2, -0.25, 0.05, 0.1, 0.15, 0.2, 0.25],
                            strut_conm2s_symmetric)
    build_strut_conm2shifting(chord, shift_chord, strut_conm2s, strut_conm2s_symmetric,
                              original_file_name)
if PBEAM:
    pbeam_factors = [0.75, 0.9, 1., 1.1, 1.2]
    build_strut_pbeams(pbeam_factors, strut_pbeams, original_file_name)
#model = BDF()
#model.read_bdf("./models/nastran/")
#model.read_bdf(original_file_name + ".bdf")


# #f1 = flut.make_flutter_response("./SOL145/polimi-145cam.f06")

# f1[1].plot_root_locus()
# #f1[1].set_plot_options()
#f1[1].plot_vg(modes=[1,2,3, 4, 5])
# #f1[1].plot_vg_vf()

# f2 = flutter.make_flutter_response("/home/acea/runs/polimi/models/SOL145/polimi-145cam_078M.f06",
#                                    {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
#                                     'dynamic_pressure':'Pa'},
#                                    {'eas':'m/s', 'velocity':'m/s', 'density':'kg/m^3', 'altitude':'m',
#                                     'dynamic_pressure':'Pa'},
#                                    )

# #f1 = flut.make_flutter_response("./SOL145/polimi-145cam.f06")

# f1[1].plot_root_locus()
# #f1[1].set_plot_options()
# f2[1].plot_vg(modes=[1,2,3, 4, 5])



# import plotly.express as px
# import plotly.graph_objects as go
# def plot_vg(modes, df_m):
#     fig = px.line(df_m[0], x='velocity', y='damping')
#     for mi in modes[1:]:
#         fig.add_line
        
        
