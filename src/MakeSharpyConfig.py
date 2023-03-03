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

import  Model_Build    
import  Sharpy_Solutions    

import inspect
__file__ = inspect.getfile(lambda: None)
case_name = 'WoT'
case_route = os.path.dirname(os.path.realpath(__file__)) + '/'

MB=Model_Build.Model_Build(case_route,case_name)
Sol=Sharpy_Solutions.Sharpy_Setup(case_route,case_name)









#/projects/ATLAS/p/WOT/DW/NSA52m/Linear_Model/NELI/Studies/Stick_no_aero_cor_CALC-01/DOE_001/MAN/0000_Clean_0.529_0.0_CASE_07_WoT55C_52m_FAME_TO_GFEM_CONM2_FUEL_CRUISE_1.0
 
 
 

#########################################################################################
'''
AC  MODEL
uncomment the section below to run the AC  MODEL
'''
#########################################################################################
#import WoT_52m as model
#Component_selection_struc=[\
#                    "CWB_right"       ,\
#                    "CWB_left"        ,\
#                    "Wing_right"      ,\
#                    "Wing_left"       ,\
#                    "Right_HTP"       ,\
#                    "Left_HTP"        ,\
#                    "VTP"             ,\
#                    "Left_HTP_att"    ,\
#                    "Right_HTP_att"   ,\
#                    "Right_pylon"     ,\
#                    "Left_pylon"      ,\
#                    "Fuselage_front"  ,\
#                    "Fuselage_rear"   ,\
#                    "Right_LG"        ,\
#                    "Left_LG"         ,\
#                    "Nose_LG"         ,\
#                    ]
#
#Component_selection_aero=[\
#                    "Wing_right"      ,\
#                    "Wing_left"       ,\
#                    "Right_HTP"       ,\
#                    "Left_HTP"        ,\
#                    "VTP"             ,\
#                    ]

import simple_AC_sweep as model         
Component_selection_struc=[\
                    "CWB_right"       ,\
                    "CWB_left"        ,\
                    "Wing_right"      ,\
                    "Wing_left"       ,\
                    "Right_HTP"       ,\
                    "Left_HTP"        ,\
                    "VTP"             ,\
                    #"Left_HTP_att"    ,\
                    #"Right_HTP_att"   ,\
                    "Right_pylon"     ,\
                    "Left_pylon"      ,\
                    "Fuselage_front"  ,\
                    "Fuselage_rear"   ,\
                    #"Right_LG"        ,\
                    #"Left_LG"         ,\
                    #"Nose_LG"         ,\
                    ]

Component_selection_aero=[\
                    "Wing_right"      ,\
                    "Wing_left"       ,\
                    "Right_HTP"       ,\
                    "Left_HTP"        ,\
                    "VTP"             ,\
                    ]

                    
# MODEL GEOMETRY
#Define Model Stiffness
bc='clamped'
#bc='free'
Clamping_node=10


AC_Components_Stiffness=model.AC_Components_Stiffness
AC_Components_nodes=model.AC_Components_nodes
AC_Components_Aerodynamics=model.AC_Components_Aerodynamics
list_mass_file=model.list_mass_file

       
AC_Components_Stiffness_=dict()             
AC_Components_nodes_=dict() 
for component in  Component_selection_struc:
    AC_Components_Stiffness_[component]=AC_Components_Stiffness[component]
    AC_Components_nodes_[component]=AC_Components_nodes[component]
    
AC_Components_Aerodynamics_=dict()
for component in  Component_selection_aero:
    AC_Components_Aerodynamics_[component]=AC_Components_Aerodynamics[component]
 
        
        

MB.clean_test_files()

#dummy mass to be attached to the massless nodes
dummy_mass=10.

n_elem, n_node_elem,n_node,Beam_DB,Node_DB=MB.generate_fem(AC_Components_Stiffness_,AC_Components_nodes_,bc,Clamping_node,list_mass_file,dummy_mass)
MB.generate_aero_file(n_elem, n_node_elem,n_node,Beam_DB,AC_Components_Aerodynamics_)


Thrust_node_nastran=[10]
Thrust_node_sharpy=[]
for node in Node_DB:
    if node[1] in Thrust_node_nastran:
        Thrust_node_sharpy.append(node[0])
        
#for node in Node_DB:
#    if node[1] == Clamping_node:
#        FoRA=node[2] 
     
                  
pitch0 = 3.5*np.pi/180.      #0.0524      #2*np.pi/180.                       # Initial angle of attack                            
yaw0  = 0.                     # Initial angle of attack                            
roll0  = 0.                     # Initial angle of attack       
orientation = algebra.euler2quat(np.array([roll0,pitch0,yaw0]))

u_inf = 180.02                        # Free stream velocity            0.529 M @ 0 alt   
 
AOA = 0. 
u_inf_direction = [np.cos(AOA), 0., np.sin(AOA)]  
rho = 1.225                              # Air density                             

panels_wake=50                          # Number of wake panels                       
                                                     
thrust_nodes = Thrust_node_sharpy                 # Nodes where thrust is applied                          
cs_i = 0                                 # Indices of control surfaces to be trimmed                                                      
nz = 1                      
FoRA = [0., 0.,0.]    

#144WT
nstep=0                     
fsi_tolerance=1e-2
fsi_relaxation=.2

#144
trim_tolerance=1.e1
fx_tolerance=trim_tolerance   
fz_tolerance=trim_tolerance 
pitching_tolerance=trim_tolerance
struct_tol=1.e-2
thrust0 = 0.
Dthrust0=2000.                          
Elev0 = -0*4*np.pi/180.                       
Dcs0=-1.*np.pi/180.
trim_relaxation_factor=.2
#########################################################################################









#########################################################################################
'''
HALE MODELuncomment the section below to run the HALE
'''
#########################################################################################
 
#import hale as model
#
#model.clean_test_files()                   
#model.generate_aero_file()
#model.generate_fem()
#
#pitch0 = 2.*np.pi/180.   
#yaw0  = 0.      
#roll0  = 0.     
#orientation = algebra.euler2quat(np.array([roll0,pitch0,yaw0]))
#thrust_nodes=[0]
#thrust0 = 6.16
#Elev0 = -2.08*np.pi/180
#n_node=10
# 
#u_inf = 10.
#
#Dthrust0=0.1                                                
#Dcs0=0.01
#trim_relaxation_factor=.2
#struct_tol=1.e-0
#AOA = 0.*np.pi/180.
#u_inf_direction = [np.cos(AOA), 0., np.sin(AOA)]  
#fx_tolerance=1.e-0      
#fz_tolerance=1.e-0       
#pitching_tolerance=1.e-0 
#rho = 1.225   
#panels_wake= 50  
#cs_i=0 
#nz=1.0
#FoRA = [0., 0.,0.]
#nstep=10                    
#fsi_tolerance=1e-0
#fsi_relaxation=0.5
#########################################################################################
  
  
#flow, settings=Sol.sol_103_aero(case_route,20)
  
#flow, settings=Sol.Check_model  (orientation,
#                            FoRA,
#                            u_inf,
#                            u_inf_direction,
#                            panels_wake,
#                            case_route)


#flow, settings=Sol.sol_144_WT   (orientation,
#                             u_inf,
#                             u_inf_direction,
#                             rho,
#                             panels_wake,
#                             FoRA,
#                             nz,
#                             nstep,
#                             fsi_tolerance,
#                             fsi_relaxation,
#                             n_node)
                            
flow, settings=Sol.sol_144( u_inf,                    
                        u_inf_direction,         
                        rho,                     
                        panels_wake,             
                        orientation,                  
                        pitch0,                 
                        thrust0,                 
                        Elev0,                     
                        thrust_nodes,            
                        cs_i,
                        nz,
                        FoRA,
                        nstep,
                        fx_tolerance,
                        fz_tolerance,
                        pitching_tolerance,
                        fsi_tolerance,
                        fsi_relaxation,
                        Dcs0,
                        Dthrust0,
                        trim_relaxation_factor,
                        struct_tol,
                        n_node) 

   
Sol.write_sharpy(settings, flow) 





