#! /usr/bin/env python3
import h5py as h5
import numpy as np
import os
import sharpy.utils.algebra as algebra

import copy
import json
import sharpy.utils.solver_interface as solver_interface
import sharpy.linear.utils.ss_interface as ss_interface
import sharpy.utils.rom_interface as rom_interface
import sharpy.rom.balanced as balanced

from            scipy.interpolate       import  interp1d


# FUNCTIONS-------------------------------------------------------------

   
#================================================================================================================================================================================#   
#================================================================================================================================================================================#   
#================================================================================================================================================================================#
#SOLUTION FUNCTIONS   
#================================================================================================================================================================================#   
#================================================================================================================================================================================#   
#================================================================================================================================================================================#   
class Sharpy_Setup:
    def __init__(self,case_route,case_name):
        self.case_route=case_route
        self.case_name=case_name
        pass

    def update_dic(self, dic, dic_new):
        """ 
        Updates dic with dic_new in a recursive manner 
        """
        
        for k, v in dic_new.items():
            if not isinstance(v,dict):
                    dic.update({k:v})
            else:
                if k in dic.keys():
                    if isinstance(dic[k],dict):
                        update_dic(dic[k],v)
                    else:
                        dic[k] = v
                else:
                    dic[k] = v
        return dic
        
    def write_sharpy(self, settings, flow):

        import configobj    
        config = configobj.ConfigObj()
        file_name = self.case_route + '/' + self.case_name + '.sharpy'
        config.filename = file_name
        
        settings_base = dict()
        settings_base['SHARPy'] = {'case':self.case_name,
                              'route': self.case_route,
                              'flow': flow,
                              'write_screen': True,
                              'write_log': True,
                              'log_folder': self.case_route,
                              'log_file': self.case_name + '.log',
                              'save_settings': True}

        for k, v in settings_base.items():
            config[k] = v                     
        for k, v in settings.items():
            config[k] = v
        
        config.write()      
                
    def sol_103(self, 
                num_modes,
                FoRA=[0.,0.,0.], 
                beam_orientation=[1., 0, 0, 0],
                rigid_body_modes=False, 
                rigid_modes_cg=False,
                use_undamped_modes=True,
                flow=[], **settings):
        """
        Modal solution (stiffness and mass matrices, and natural frequencies)
        in the reference configuration
        """

        settings_new = dict()
        if flow == []:
            flow = ['BeamLoader','Modal']
     
        for k in flow:
            settings_new[k] = {}
                 
        settings_new['BeamLoader']['orientation'] = beam_orientation
        settings_new['BeamLoader']['unsteady'] = False
        
        settings_new['Modal']['NumLambda'] = num_modes
        settings_new['Modal']['rigid_body_modes'] = rigid_body_modes
        settings_new['Modal']['rigid_modes_cg'] = rigid_modes_cg
        settings_new['Modal']['use_undamped_modes'] = use_undamped_modes
        settings_new['Modal']['print_matrices'] = True
        
        settings_new = self.update_dic(settings_new, settings)        
        return flow, settings_new
        
    def sol_103_aero(self, 
                num_modes,
                FoRA=[0.,0.,0.], 
                beam_orientation=[1., 0, 0, 0],
                rigid_body_modes=False,
                rigid_modes_cg=False,
                use_undamped_modes=True,
                write_modes_vtk=True,
                tstep_factor = 1.,
                panels_wake=40,
                flow=[], **settings):
        """
        Modal solution (stiffness and mass matrices, and natural frequencies)
        in the reference configuration with modes displayed
        """

        settings_new = dict()
        if flow == []:
            flow = ['BeamLoader','AerogridLoader','Modal','BeamPlot','AerogridPlot']
     
        for k in flow:
            settings_new[k] = {}
                 
        settings_new['BeamLoader']['orientation'] = beam_orientation
        settings_new['BeamLoader']['unsteady'] = False
        
        settings_new['AerogridLoader']['unsteady'] = False
        settings_new['AerogridLoader']['mstar'] = panels_wake
        settings_new['AerogridLoader']['wake_shape_generator'] = 'StraightWake'
        settings_new['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': 100.,
                                                                        'u_inf_direction': [1.,0.,0.],
                                                                        'dt': 0.1}

        settings_new['Modal']['NumLambda'] = num_modes
        settings_new['Modal']['rigid_body_modes'] = rigid_body_modes
        settings_new['Modal']['rigid_modes_cg'] = rigid_modes_cg
        settings_new['Modal']['use_undamped_modes'] = use_undamped_modes
        settings_new['Modal']['print_matrices'] = True
        
        
        settings_new['BeamPlot']['folder'] = self.case_route + '/output/'
        settings_new['BeamPlot']['include_rbm'] = True
        settings_new['BeamPlot']['include_applied_forces'] = True
        settings_new['BeamPlot']['include_applied_moments'] = True
        settings_new['BeamPlot']['include_forward_motion'] = False
                                
        settings_new['AerogridPlot']['include_rbm'] = False
        settings_new['AerogridPlot']['include_applied_forces'] = False
        
        settings_new = self.update_dic(settings_new, settings)  
        
        return flow, settings_new   

    def Check_model (self, 
                    orientation,
                    FoRA,
                    u_inf,
                    u_inf_direction,
                    panels_wake,
                    case_route,
                    dt=0.1,
                    flow=[], **settings):

        
        """ Aeroelastic equilibrium"""

        settings_new = dict()
        if flow == []:
            flow = ['BeamLoader', 'AerogridLoader','BeamPlot','AerogridPlot']
        for k in flow:
            settings_new[k] = {}      

        settings_new['BeamLoader']['orientation'] = orientation
        settings_new['BeamLoader']['unsteady'] = False
        settings_new['AerogridLoader']['mstar'] = panels_wake
        settings_new['AerogridLoader']['unsteady'] = False
        settings_new['AerogridLoader']['freestream_dir']=u_inf_direction
        settings_new['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': u_inf,
                                                                        'u_inf_direction': u_inf_direction,
                                                                        'dt': dt}
                                                       
        settings_new['BeamPlot']['folder'] = self.case_route + '/output/'
        settings_new['BeamPlot']['include_rbm'] = True
        settings_new['BeamPlot']['include_applied_forces'] = True
        settings_new['BeamPlot']['include_applied_moments'] = True
        settings_new['BeamPlot']['include_forward_motion'] = False
                                
        settings_new['AerogridPlot']['include_rbm'] = False
        settings_new['AerogridPlot']['include_applied_forces'] = False
        
        settings_new = self.update_dic(settings_new, settings)        
        return flow, settings_new
        
    def sol_144(self, 
                u_inf,                        # Free stream velocity
                u_inf_direction,              # Free stream direction
                rho,                          # Air density 
                panels_wake,                  # Number of wake panels 
                orientation,                       # Initial angle of attack
                pitch0,                      # Number of     
                thrust0,                      # Number of     
                cs0,                          # Number of wake panels 
                thrust_nodes,                 # Nodes where thrust is applied
                cs_i,                         # Indices of control surfaces to be trimmed
                nz,                           # Gravity factor for manoeuvres     
                FoRA,                         # Node A coordinates 
                nstep,
                fx_tolerance,
                fz_tolerance,
                pitching_tolerance, 
                fsi_tolerance,           # FSI loop tolerance
                fsi_relaxation,           # FSI relaxation_factor
                Dcs0,                     # Initial control surface variation
                Dthrust0,                 # Initial thrust variation 
                trim_relaxation_factor,   # Relaxation factor 
                struct_tol,
                n_node,               

                horseshoe=False,              # Horseshoe solution 
                dt=0.05,                      # dt for uvlm 
                trim_max_iter=100,            # Mximum number of trim iterations
 

                fsi_maxiter=100,              # FSI maximum number of iterations
                flow=[], **settings):         # Flow and settings to modify the predifined solution
                
        """ Longitudinal aircraft trim"""

        settings_new = dict()
        if flow == []:
            flow = ['BeamLoader', 'AerogridLoader', 'StaticTrim','BeamLoads','BeamPlot','StallCheck','AerogridPlot','WriteVariablesTime']
        for k in flow:
            settings_new[k] = {}
                
        settings_new['BeamLoader']['orientation'] = orientation
        settings_new['BeamLoader']['unsteady'] = False
        
        settings_new['AerogridLoader']['mstar'] = panels_wake
        settings_new['AerogridLoader']['unsteady'] = False
        settings_new['AerogridLoader']['freestream_dir']=u_inf_direction
        settings_new['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': u_inf,
                                                                        'u_inf_direction': u_inf_direction,
                                                                        'dt': dt}
                                                                        
        settings_new['StaticTrim']['print_info'] = True
        settings_new['StaticTrim']['initial_alpha'] = pitch0
        settings_new['StaticTrim']['initial_deflection'] = cs0
        settings_new['StaticTrim']['initial_angle_eps'] = Dcs0
        settings_new['StaticTrim']['initial_thrust'] = thrust0
        settings_new['StaticTrim']['initial_thrust_eps'] = Dthrust0
        settings_new['StaticTrim']['thrust_nodes'] = thrust_nodes
        settings_new['StaticTrim']['tail_cs_index'] = cs_i
        settings_new['StaticTrim']['fx_tolerance'] = fx_tolerance
        settings_new['StaticTrim']['fz_tolerance'] = fz_tolerance
        settings_new['StaticTrim']['m_tolerance'] = pitching_tolerance
        settings_new['StaticTrim']['max_iter'] = trim_max_iter
        settings_new['StaticTrim']['relaxation_factor'] = trim_relaxation_factor
        settings_new['StaticTrim']['save_info'] = True
        settings_new['StaticTrim']['solver'] = 'StaticCoupled'
        settings_new['StaticTrim']['solver_settings'] = {'print_info': True,
                                     'structural_solver': 'NonLinearStatic',
                                     'structural_solver_settings': {'print_info': True,
                                                                    'max_iterations': 100,
                                                                    'num_load_steps': nstep,
                                                                    'delta_curved': 1e-1,
                                                                    'min_delta': struct_tol,
                                                                    'gravity_on': True,
                                                                    'gravity':nz*9.807,
                                                                    'initial_position':FoRA,
                                                                    'dt':dt
                                                                    },
                                     'aero_solver': 'StaticUvlm',
                                     'aero_solver_settings': {'print_info': True,
                                                              'rho':rho,
                                                              'horseshoe': horseshoe,
                                                              'num_cores': 1,
                                                              'n_rollup': int(1.2*panels_wake),
                                                              'rollup_dt': dt,
                                                              'rollup_aic_refresh': 1,
                                                              'rollup_tolerance': 1e-1,
                                                              'velocity_field_generator': \
                                                              'SteadyVelocityField',
                                                              'initial_position':FoRA,
                                                              'velocity_field_input': \
                                                              {'u_inf':u_inf,
                                                               'u_inf_direction': u_inf_direction
                                                              },
                                                              },
                                     'max_iter': fsi_maxiter,
                                     'n_load_steps': nstep,
                                     'tolerance': fsi_tolerance,
                                     'relaxation_factor': fsi_relaxation
                                                    }
                
        settings_new['BeamLoads']['csv_output'] = True
        settings_new['BeamLoads']['folder'] = self.case_route + '/output/'
        settings_new['BeamLoads']['output_file_name'] = 'Loads.csv'
        
        settings_new['BeamPlot']['folder'] = self.case_route + '/output/'
        settings_new['BeamPlot']['include_rbm'] = True
        settings_new['BeamPlot']['include_applied_forces'] = True
        settings_new['BeamPlot']['include_applied_moments'] = True
        settings_new['BeamPlot']['include_forward_motion'] = False
        settings_new['BeamPlot']['include_FoR'] = True
                                
        settings_new['StallCheck']['output_degrees'] = True
        settings_new['StallCheck']['print_info'] = True
        
        settings_new['AerogridPlot']['include_rbm'] = False
        settings_new['AerogridPlot']['include_applied_forces'] = True

        settings_new['WriteVariablesTime']['structure_variables'] = ['pos', 'psi']  # pos is displacement in A frame, psi is the CRV
        settings_new['WriteVariablesTime']['structure_nodes'] = list(range(0, n_node))  # a file for each node in the list
        settings_new['WriteVariablesTime']['cleanup_old_solution'] = True
        
        settings_new = self.update_dic(settings_new, settings)
        
        return flow, settings_new

    def sol_144_WT  (self, 
                    orientation,
                    u_inf,
                    u_inf_direction,
                    rho,
                    panels_wake,
                    FoRA,
                    nz,
                    nstep,
                    fsi_tolerance,
                    fsi_relaxation,
                    n_node,
                    horseshoe=False,
                    dt=0.01,
                    gravity_on=True,
                    fsi_maxiter=100,
                    flow=[], **settings):

        
        """ Aeroelastic equilibrium like Wind Tunnel Test+"""
           
        settings_new = dict()
        if flow == []:
            flow = ['BeamLoader', 'AerogridLoader','StaticCoupled','BeamLoads','BeamPlot','StallCheck','AerogridPlot','WriteVariablesTime']
        for k in flow:
            settings_new[k] = {} 

        settings_new['BeamLoader']['orientation'] = orientation
        settings_new['BeamLoader']['unsteady'] = False
        
        settings_new['AerogridLoader']['mstar'] = panels_wake
        settings_new['AerogridLoader']['unsteady'] = False
        settings_new['AerogridLoader']['freestream_dir']=u_inf_direction
        settings_new['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': u_inf,
                                                                        'u_inf_direction': u_inf_direction,
                                                                        'dt': dt}
        settings_new['StaticCoupled']['print_info'] = True
        settings_new['StaticCoupled']['n_load_steps'] = nstep
        settings_new['StaticCoupled']['max_iter'] = fsi_maxiter
        settings_new['StaticCoupled']['tolerance'] = fsi_tolerance
        settings_new['StaticCoupled']['relaxation_factor'] = fsi_relaxation
        settings_new['StaticCoupled']['aero_solver'] = 'StaticUvlm'
        settings_new['StaticCoupled']['aero_solver_settings'] = {'print_info': True,
                                                                 'rho':rho,
                                                                 'horseshoe': horseshoe,
                                                                 'num_cores': 1,
                                                                 'n_rollup': int(2.*panels_wake),
                                                                 'rollup_dt': dt,
                                                                 'rollup_aic_refresh': 1,
                                                                 'rollup_tolerance': 1e-1,
                                                                 'velocity_field_generator': 'SteadyVelocityField',
                                                                 'initial_position':FoRA,
                                                                 'velocity_field_input': \
                                                                 {'u_inf': u_inf,
                                                                  'u_inf_direction':u_inf_direction}
                                                                 }

        settings_new['StaticCoupled']['structural_solver'] = 'NonLinearStatic'
        settings_new['StaticCoupled']['structural_solver_settings'] = {'initial_position':FoRA,
                                                                       'dt': dt,
                                                                       'gravity_on':gravity_on,
                                                                       'gravity':nz*9.807
                                                                       }

        settings_new['BeamLoads']['csv_output'] = True
        settings_new['BeamLoads']['folder'] = self.case_route + '/output/'
        settings_new['BeamLoads']['output_file_name'] = 'Loads.csv'
        
        settings_new['BeamPlot']['folder'] = self.case_route + '/output/'
        settings_new['BeamPlot']['include_rbm'] = True
        settings_new['BeamPlot']['include_applied_forces'] = True
        settings_new['BeamPlot']['include_applied_moments'] = True
        settings_new['BeamPlot']['include_forward_motion'] = False
        settings_new['BeamPlot']['include_FoR'] = True
                                
        settings_new['StallCheck']['output_degrees'] = True
        settings_new['StallCheck']['print_info'] = True
        
        settings_new['AerogridPlot']['include_rbm'] = False
        settings_new['AerogridPlot']['include_applied_forces'] = True

        settings_new['WriteVariablesTime']['structure_variables'] = ['pos', 'psi']  # pos is displacement in A frame, psi is the CRV
        settings_new['WriteVariablesTime']['structure_nodes'] = list(range(0, n_node))  # a file for each node in the list
        settings_new['WriteVariablesTime']['cleanup_old_solution'] = True

        settings_new = self.update_dic(settings_new, settings)        
        return flow, settings_new
