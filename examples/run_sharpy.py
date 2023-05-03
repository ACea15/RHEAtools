import numpy as np
import sharpy.cases.models_generator.gen_main as gm
import pickle
import pdb

with open("./model_settings.pkl", "rb") as f:
    model_settings = pickle.load(f)
with open("./sharpy_component.pkl", "rb") as f:
    component_settings = pickle.load(f)


u_inf = 1.
c_ref = 3.
bound_panels = component_settings['wing_r1']['aero']['surface_m']
AoA = 0. * np.pi /180
sol_dict = {
    'sharpy': {'simulation_input': None,
               'default_module': 'sharpy.routines.static',
               'default_solution': 'sol_112',
               'default_solution_vars': {
                   'u_inf': u_inf,
                   'rho': 0.6,
                   'gravity_on': True,
                   'dt': c_ref / bound_panels / u_inf,
                   'panels_wake': bound_panels * 10,
                   'rotationA': [0., AoA, 0.],
                   'horseshoe': True,
                   'fsi_maxiter': 100,
                   'fsi_tolerance': 1e-6,
                   'fsi_relaxation': 0.1,
                   'fsi_load_steps': 5,
                   's_maxiter': 100,
                   's_tolerance': 1e-5,
                   's_relaxation': 0.2,
                   's_load_steps': 1,
                   's_delta_curved': 1e-4,
                   #'add2_flow': [['AerogridLoader', ['plot']]],
                   'add2_flow': [['StaticCoupled', ['plot']]]
               },
               'default_sharpy': {},
               'model_route': None
               }
    }
g1 = gm.Model('sharpy',['sharpy'], model_dict=model_settings,
              components_dict=component_settings,
              simulation_dict=sol_dict)
sol = g1.run()
