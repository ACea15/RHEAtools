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
#MODEL FUNCTIONS   
#================================================================================================================================================================================#   
#================================================================================================================================================================================#   
#================================================================================================================================================================================#   
class Model_Build:
    def __init__(self,case_route,case_name):
        self.case_route=case_route
        self.case_name=case_name
        pass
        
    def clean_test_files(self):
        fem_file_name = self.case_route + '/' + self.case_name + '.fem.h5'
        if os.path.isfile(fem_file_name):
            os.remove(fem_file_name)

        dyn_file_name = self.case_route + '/' + self.case_name + '.dyn.h5'
        if os.path.isfile(dyn_file_name):
            os.remove(dyn_file_name)

        aero_file_name = self.case_route + '/' + self.case_name + '.aero.h5'
        if os.path.isfile(aero_file_name):
            os.remove(aero_file_name)

        solver_file_name = self.case_route + '/' + self.case_name + '.sharpy'
        if os.path.isfile(solver_file_name):
            os.remove(solver_file_name)

        flightcon_file_name = self.case_route + '/' + self.case_name + '.flightcon.txt'
        if os.path.isfile(flightcon_file_name):
            os.remove(flightcon_file_name)
            
    def generate_fem(self,AC_Components_Stiffness,AC_Components_nodes,bc,Clamping_node,list_mass_file,dummy_mass):
        #===========================================================================================================================================================#
        #Stiffness processing
        # beam processing
        '''
        node for each element, they are 3 due to the nonlinear formulation 
        '''
        n_node_elem = 3

        # total number of aicraft components
        n_AC_comp=len(AC_Components_Stiffness)

        '''
        make a database of all the elements and the related properties 
        NOTE the order of the beams and the related array is the same of BEAM DB
        '''
        Beam_DB=[]

        counter_beam=0
        counter_node=0

        nastran_node_list=[]
        sharpy_node_list=[]
        free_nodes_list=[]
        
        '''
        make a list of the end and beginning of beam nodes and in the middle
        '''
        extremes_node_list=[]
        middle_node_list=[]
        for component in AC_Components_Stiffness:
            matrix_nastran_node=AC_Components_nodes[component]["values"]     
            '''
            identify nodes in the middle and at the extremes
            '''
            nnode=len(matrix_nastran_node)
            for i in range(nnode):
                node=matrix_nastran_node[i][0]
                if (i==0 or i==nnode-1) and (node not in extremes_node_list):
                    extremes_node_list.append(node)
                else:
                    middle_node_list.append(node)
        '''
        define the list of the real free nodes
        '''
        for node in extremes_node_list:
            if node not in middle_node_list:
                free_nodes_list.append(node)
                                            
        for component in AC_Components_Stiffness:
            matrix_nastran_node=AC_Components_nodes[component]["values"]
            matrix_nastran_beam=AC_Components_Stiffness[component]["values"] 
            
            #loop over elements of the component
            counter_elem=0
            for beam in AC_Components_Stiffness[component]["values"]:
                nastran_node_a = beam[0]
                nastran_node_b = beam[1]

                #search for the node id position in the nastran_node matrix
                counter=0
                for nastran_node in matrix_nastran_node:
                    if nastran_node[0]==nastran_node_a:
                        nastran_node_a_index=counter
                    elif nastran_node[0]==nastran_node_b:
                        nastran_node_b_index=counter
                    counter+=1
                    
                    
                '''
                calculate node ab
                '''
                nastran_node_a_pos=matrix_nastran_node[nastran_node_a_index][1:4]
                nastran_node_b_pos=matrix_nastran_node[nastran_node_b_index][1:4]
                
                nastran_node_ab_pos=[]
                for i in range (3):
                    nastran_node_a_pos[i]=round(nastran_node_a_pos[i],3)
                    nastran_node_b_pos[i]=round(nastran_node_b_pos[i],3)
                    nastran_node_ab_pos.append((nastran_node_a_pos[i]+nastran_node_b_pos[i])/2)
       
                  

                '''
                define sharpy nodes id
                '''
                ###############################################################################
                ###############################################################################
                #qua devo fare un check su node a e nodo b e mettere l id dei nodi in
                ###############################################################################
                ###############################################################################
                
                if (nastran_node_a not in nastran_node_list) and (nastran_node_b not in nastran_node_list):
                    if (counter_node in sharpy_node_list):
                        counter_node+=1
                        
                    sharpy_node_a=counter_node
                    sharpy_node_b=counter_node+2
                    sharpy_node_ab=counter_node+1
                    nastran_node_list.append(nastran_node_a)
                    sharpy_node_list.append(sharpy_node_a)
                    nastran_node_list.append(nastran_node_b)
                    sharpy_node_list.append(sharpy_node_b)
                    #print('a',nastran_node_a,sharpy_node_a,nastran_node_b,sharpy_node_b,sharpy_node_ab)
                    
                elif (nastran_node_a not in nastran_node_list) and (nastran_node_b in nastran_node_list):
                    sharpy_node_a=counter_node+2
                    indx=nastran_node_list.index(nastran_node_b)              
                    sharpy_node_b=sharpy_node_list[indx]
                    sharpy_node_ab=counter_node+1
                    nastran_node_list.append(nastran_node_a)
                    sharpy_node_list.append(sharpy_node_a)
                    #print('b',nastran_node_a,sharpy_node_a,nastran_node_b,sharpy_node_b,sharpy_node_ab)

                elif (nastran_node_a in nastran_node_list) and (nastran_node_b not in nastran_node_list):
                    indx=nastran_node_list.index(nastran_node_a)              
                    sharpy_node_a=sharpy_node_list[indx]
                    sharpy_node_b=counter_node+2
                    sharpy_node_ab=counter_node+1
                    nastran_node_list.append(nastran_node_b)
                    sharpy_node_list.append(sharpy_node_b)
                    #print('c',nastran_node_a,sharpy_node_a,nastran_node_b,sharpy_node_b,sharpy_node_ab)
                    
                elif (nastran_node_a in nastran_node_list) and (nastran_node_b in nastran_node_list):
                    indx=nastran_node_list.index(nastran_node_a)              
                    sharpy_node_a=sharpy_node_list[indx]
                    indx=nastran_node_list.index(nastran_node_b)              
                    sharpy_node_b=sharpy_node_list[indx]
                    sharpy_node_ab=counter_node+1
                    counter_node+=-1
                    #print('d',nastran_node_a,sharpy_node_a,nastran_node_b,sharpy_node_b,sharpy_node_ab)
 
                counter_node+=2
 
                                          
                '''    
                calculate the rotation matrix for each beam
                this is defined in the sharpy frame of reference where 
                z is the vertical axys 
                y is in the wing plane
                x goes from node to node
                '''
                
                vec=np.zeros((2,3))
                loc_z=np.zeros((1,3))
                Rot_Mat=np.zeros((3,3))
                
                
                if AC_Components_nodes[component]['main_direction'] =='z':
                    loc_z[0,1]=-1.
                else:
                    loc_z[0,2]=1.
                if AC_Components_nodes[component]['main_direction']=='y':
                    if matrix_nastran_node[-1][1]<0.:
                        loc_z[0,2]=-1.
                for i in range (3):
                    vec[0,i]=matrix_nastran_node[nastran_node_a_index][i+1]
                    vec[1,i]=matrix_nastran_node[nastran_node_b_index][i+1]
                x_dir_cos=vec[1,:]-vec[0,:]
                x_dir_cos=x_dir_cos/np.linalg.norm(x_dir_cos) 
                z_proj=np.dot(loc_z,x_dir_cos)*x_dir_cos                     #projection of [0,0,1] on the beam long axys
                #z_proj[0]=0                                              #remove x component
                z_dir_cos=loc_z-z_proj
                z_dir_cos=z_dir_cos/np.linalg.norm(z_dir_cos) 
                z_dir_cos=z_dir_cos[0]
                y_dir_cos=np.cross(z_dir_cos,x_dir_cos)
                y_dir_cos=y_dir_cos/np.linalg.norm(y_dir_cos) 
            
                
                for i in range (3):
                    Rot_Mat[0,i]=round(x_dir_cos[i],3)
                    Rot_Mat[1,i]=round(y_dir_cos[i],3)
                    Rot_Mat[2,i]=round(z_dir_cos[i],3)   

                '''    
                calculate stiffness
                '''
                stiffness=beam[2:]
                
                beam_=[counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component]
                Beam_DB.append(beam_)

                counter_beam+=1
 
        '''
        make database with all the nodes and the related postions
        '''
        Node_DB=[]
        node_id_list=[]
        for beam in Beam_DB:
            counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component=beam
           
            if sharpy_node_a not in node_id_list:
                node_id_list.append(sharpy_node_a)
                value=[sharpy_node_a,nastran_node_a,nastran_node_a_pos]
                Node_DB.append(value)
                
            if sharpy_node_ab not in node_id_list:
                node_id_list.append(sharpy_node_ab)
                value=[sharpy_node_ab,'nan',nastran_node_ab_pos]
                Node_DB.append(value)
                
            if sharpy_node_b not in node_id_list:
                node_id_list.append(sharpy_node_b)
                value=[sharpy_node_b,nastran_node_b,nastran_node_b_pos]
                Node_DB.append(value)                 
        
        #Sort node DB based on first element of sublist
        def Sort(sub_li):
            sub_li.sort(key = lambda x: x[0])
            return sub_li
        Node_DB=Sort(Node_DB)
        
        # total number of elements
        n_elem=len(Beam_DB)
        # total number of nodes considering 3 node per beam    
        n_node=len(Node_DB)
        
        ''' 
        node_coordinates [num_node, 3]: coordinates of the nodes in body-attached FoR (A).
        A is the body reference system
        NOTE: the 
        a----c----b   node_ab is the one in the middle of the element
        coordinates as the nodes listed as acacacacacac where the second a is the b of the previous beam
        '''
        
        
        #Calculate the location of the reference node
        node_list=[]
        for node in Node_DB:
            node_list.append(node[1])
        ForA_indx=node_list.index(Clamping_node)
        ForA=Node_DB[ForA_indx][2][:]
        
        #express the nodes locattions with respect to the reference nodes
        node_coordinates = np.zeros((n_node,3))
        counter=0
        for node in Node_DB:
            for i in range (3):
                #node_coordinates[counter,i]=round(node[2][i]-ForA[i],4)
                node_coordinates[counter,i]=round(node[2][i],4)
            counter+=1
                 

        '''
        connectivities [num_elem, num_node_elem] : Beam element’s connectivities.
        '''
        connectivities = np.zeros((n_elem, n_node_elem), dtype=int) 
        counter=0
        for beam in Beam_DB:
            counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component=beam
            connectivities[counter,0]=sharpy_node_a
            connectivities[counter,1]=sharpy_node_b
            connectivities[counter,2]=sharpy_node_ab
            counter+=1


        '''
        frame_of_reference_delta [num_elem, num_node_elem, 3]: rotation vector to FoR B.
        basically contains the local frame of ref for each beam
        x goes from node to node
        y is "vertical"
        z is "horizontal"
        the matrix has dimentions n_elem, n_node_elem, 3
        the last column is referred to xyz so the element [0,0,:]
        contain the xzy of the delta vector of the first node of the first element
        ''' 
         
        frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
        counter=0
        for beam in Beam_DB:
            counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component=beam
            delta=np.zeros((3))
            #Rot_Mat[1,2]=0 
            #delta[:]=Rot_Mat[1,:] 
            if AC_Components_nodes[component]['main_direction']=='x':
                    delta[0]=0.
                    delta[1]=1.
                    delta[2]=0. 
            if AC_Components_nodes[component]['main_direction']=='y':
                if nastran_node_b_pos[1]>=0:
                    delta[0]=-1.
                    delta[1]=0.
                    delta[2]=0.
                elif nastran_node_b_pos[1]<0:
                    delta[0]=+1.
                    delta[1]=0.
                    delta[2]=0.
            if AC_Components_nodes[component]['main_direction']=='z':
                    delta[0]=-1.
                    delta[1]=0.
                    delta[2]=0. 
                    
            
            for inode in range(n_node_elem):
                frame_of_reference_delta[counter, inode, :] = delta[:]
            counter+=1
        


        '''
        elem_stiffness [num_elem] : array of indices (starting at 0).
        this is just an array where every element point to the element stiffness matrix
        in this case every element has its own matrix, 
        ''' 
        elem_stiffness = np.zeros((n_elem), dtype=int)
        for beam in Beam_DB:
            counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component=beam
            elem_stiffness[counter_beam]=counter_beam
     


        '''
        stiffness_db is a matrix containing the stiffness database, ie EI GJ for each beam. it is like a stiffness database
        stiffness_db [:, 6, 6]: database of stiffness matrices.
        The first dimension has as many elements as different stiffness matrices are in the model.
        '''
        stiffness_db = np.zeros((n_elem, 6, 6))
        for beam in Beam_DB:
            counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component=beam

            eiy=stiffness[0]
            eiz=stiffness[1]
            gj=stiffness[2]
            ea=stiffness[5]
            ky=stiffness[6]
            kz=stiffness[7]
            
            stiffness_elem=np.diag([ea, gj, gj, gj, eiz, eiy])

            #stiffness_elem=np.zeros((6,6))
            #stiffness_elem[0,0]=ea
            ##Note: I define ea both for the z and y shear stiffness because in Nastran the default value was defined 
            #stiffness_elem[1,1]=gj#ea#/2.6
            #stiffness_elem[2,2]=gj#ea#/2.6
            #stiffness_elem[3,3]=gj 
            ##Note: this is because the nastran local ref system differs from the sharpy ref, Nastran local z axysi is on the wing plane and y is vertical, the opposite in sharpy
            #stiffness_elem[4,4]=eiz   
            #stiffness_elem[5,5]=eiy
            
            stiffness_db[counter_beam, ...]=stiffness_elem

         
        '''
        beam_number [num_elem]: beam index.
        Is another array of integers. Usually you don’t need to modify its value. Leave it at 0.
        each numner seems to identify a component, wing 0 fuselage 1 .....
        '''
        beam_number = np.zeros((n_elem), dtype=int)
        #for beam in Beam_DB:
        #    counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component=beam
        #    beam_number[counter_beam]=counter_beam
 
        '''
        connectivites [num_elem, num_node_elem] : Beam element’s connectivities.
        '''    
        connectivities = np.zeros((n_elem, n_node_elem), dtype=int)
        for beam in Beam_DB:
            counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component=beam
            connectivities[counter_beam,0]=sharpy_node_a
            connectivities[counter_beam,1]=sharpy_node_b
            connectivities[counter_beam,2]=sharpy_node_ab


                          
        #===========================================================================================================================================================#
        #Boundary Condition processing 
        '''
        boundary_conditions [num_node]: boundary conditions.
        An array of integers (np.zeros((num_node, ), dtype=int)) and contains all 0 except for
        One node NEEDS to have a 1 , this is the reference node. Usually, the first node has 1 and is located in [0, 0, 0]. This makes things much easier.
        If the node is a tip of a beam (is not attached to 2 elements, but just 1), it needs to have a -1.
        '''
        boundary_conditions = np.zeros((n_node), dtype=int)
        
        counter = 0
        for node in Node_DB:
            if node[1] in free_nodes_list:
                boundary_conditions[counter]=-1
            else:
                boundary_conditions[counter]=0
            counter+=1
        
        indx=nastran_node_list.index(Clamping_node)
        sharpy_node_a=sharpy_node_list[indx]

        #if bc =='clamped':
        #    boundary_conditions[sharpy_node_a]=1
        #if bc =='free':
        #    boundary_conditions[sharpy_node_a]=0
        boundary_conditions[sharpy_node_a]=1
            
        #===========================================================================================================================================================#
        #Mass processing    
        '''
        this function read a CONM2 file and store the data into a database
        '''
        def Read_CONM2(File_name,nastran_node_list):
            def read_scientific(string):
                string=string.strip()
                if ('-' in string[1:]) and ('e' not in string and 'E' not in string):
                    string=string[0:(string[1:].find('-')+1)]+'e'+string[(string[1:].find('-')+1):]
                    
                if ('+' in string[1:]) and ('e' not in string and 'E' not in string):
                    string=string[0:(string[1:].find('+')+1)]+'e'+string[(string[1:].find('+')+1):]
                return float(string)
                
                
            mass_DB=[]
            with open(File_name) as f:
                content = f.readlines()

            nline=len(content)

            line=0
            while line < nline:
                string=content[line]
                line=line+1
                if len(string)>5:
                    if string[0:5]=='CONM2' :
                        node_id=int(string[16:24])
                        mass_ref=int(string[24:32])
                        mass=read_scientific(string[32:40])
                        mass_x=read_scientific(string[40:48])
                        mass_y=read_scientific(string[48:56])
                        mass_z=read_scientific(string[56:64])
                        string=content[line]
                        line=line+1
                        if string[0:5].strip()!='CONM2' and len(string.strip())>0:
                            Ixx=read_scientific(string[8:16])
                            Iyx=read_scientific(string[16:24])
                            Iyy=read_scientific(string[24:32])
                            Izx=read_scientific(string[32:40])
                            Izy=read_scientific(string[40:48])
                            Izz=read_scientific(string[48:56])
                        else:
                            Ixx=0.
                            Iyx=0.
                            Iyy=0.
                            Izx=0.
                            Izy=0.
                            Izz=0.
                        list=node_id,mass_ref,mass,mass_x,mass_y,mass_z,Ixx,Iyx,Iyy,Izx,Izy,Izz
                        if node_id in nastran_node_list:
                            mass_DB.append(list)
            return mass_DB
            
        mass_DB=[]
        nastran_node_list=[]
        for node in Node_DB:
            nastran_node_list.append(node[1])
        for file in list_mass_file:
            mass_DB+=Read_CONM2(file,nastran_node_list)
         
        n_lumped_mass=len(mass_DB)
        
        '''
        lumped_mass [:]: lumped masses. This array collects all the masses
        lumped_mass_nodes [:]: Lumped mass nodes. This array makes the connection between the massess and the nodes
        lumped_mass_inertia [:, 3, 3]: Lumped mass inertia. This array collects all the inertia
        lumped_mass_position [:, 3]: Lumped mass position. This array collects all the mass location with respect to the nodes
        '''
         
        lumped_mass = np.zeros((n_lumped_mass))
        lumped_mass_nodes = np.zeros((n_lumped_mass), dtype=int)
        lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
        lumped_mass_position = np.zeros((n_lumped_mass, 3))

        counter=0

        for mass_data in mass_DB:
            nastran_node_id,mass_ref,mass,mass_x,mass_y,mass_z,Ixx,Iyx,Iyy,Izx,Izy,Izz=mass_data
            
            #loop over the node_DB to associate the masses to the correct node
            check=True
            for node in Node_DB:
                if node[1]==nastran_node_id:
                
                    for beam in Beam_DB:
                        counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component=beam
      
                        if (nastran_node_id==nastran_node_a or nastran_node_id==nastran_node_b) and check:
                            #check=False
                        
                            #define mass
                            lumped_mass[counter]=mass
                            
                            #define inertia
                            mass_inertia=np.zeros((3,3))
                            mass_inertia[0,0]=Ixx
                            mass_inertia[1,1]=Iyy
                            mass_inertia[2,2]=Izz

                            mass_inertia[1,0]=-Iyx
                            mass_inertia[2,0]=-Izx
                            mass_inertia[2,1]=-Izy
                            
                            mass_inertia[0,1]=+Iyx
                            mass_inertia[0,2]=+Izx
                            mass_inertia[1,2]=+Izy
                             
                            
                            #project the mass_inertia in the local frame of ref
                            mass_inertia=np.dot(Rot_Mat,mass_inertia)
                            mass_inertia=np.dot(mass_inertia,Rot_Mat.T)
                            lumped_mass_inertia[counter,...]=mass_inertia

                            #calcualte the mass distance from the nodes in global axys
                            vec=np.zeros((3,1))
                            vec[0,0]=mass_x-node[2][0]
                            vec[1,0]=mass_y-node[2][1]
                            vec[2,0]=mass_z-node[2][2]
                            
                            #project the distance in the local frame of ref
                            vec=np.dot(Rot_Mat,vec)
                            
                            #define position in the local ref
                            lumped_mass_position[counter,0]=vec[0,0]
                            lumped_mass_position[counter,1]=vec[1,0]
                            lumped_mass_position[counter,2]=vec[2,0]
                            
                            #associate the mass to the correct Sharpy node
                            lumped_mass_nodes[counter]=node[0]

                            #quit the loop since the local ref is only the one of the first beam where the node is defined 
                            break         

            counter+=1
           
        '''
        create a fake distribuited mass to make sure the mass matrix is not singular
        '''    
        n_mass=1
        mass_db = np.zeros((n_mass, 6, 6))
        for i in range (6):
            #Note: this are small values to avoid a singular mass matrix
            mass_db[0,i,i]=dummy_mass
            
        elem_mass = np.zeros((n_elem), dtype=int)
        for i in range (n_elem):
            elem_mass[i]=0
          
        #===========================================================================================================================================================#
        # OTHER PLACEHOLDERS
        # beam
        app_forces = np.zeros((n_node, 6))
        structural_twist = np.zeros((n_elem, 3))

        '''
        print hdf5 file
        '''

        with h5.File(self.case_route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            coordinates = h5file.create_dataset('coordinates', data=node_coordinates)
            conectivities = h5file.create_dataset('connectivities', data=connectivities)
            num_nodes_elem_handle = h5file.create_dataset('num_node_elem', data=n_node_elem)
            num_nodes_handle = h5file.create_dataset('num_node', data=n_node)
            num_elem_handle = h5file.create_dataset('num_elem', data=n_elem)
            stiffness_db_handle = h5file.create_dataset('stiffness_db', data=stiffness_db)
            stiffness_handle = h5file.create_dataset('elem_stiffness', data=elem_stiffness)
            mass_db_handle = h5file.create_dataset('mass_db', data=mass_db)
            mass_handle = h5file.create_dataset('elem_mass', data=elem_mass)
            frame_of_reference_delta_handle = h5file.create_dataset('frame_of_reference_delta', data=frame_of_reference_delta)
            structural_twist_handle = h5file.create_dataset('structural_twist', data=structural_twist)
            bocos_handle = h5file.create_dataset('boundary_conditions', data=boundary_conditions)
            beam_handle = h5file.create_dataset('beam_number', data=beam_number)
            app_forces_handle = h5file.create_dataset('app_forces', data=app_forces)
            lumped_mass_nodes_handle = h5file.create_dataset('lumped_mass_nodes', data=lumped_mass_nodes)
            lumped_mass_handle = h5file.create_dataset('lumped_mass', data=lumped_mass)
            lumped_mass_inertia_handle = h5file.create_dataset('lumped_mass_inertia', data=lumped_mass_inertia)
            lumped_mass_position_handle = h5file.create_dataset('lumped_mass_position', data=lumped_mass_position)
     
        return n_elem, n_node_elem,n_node,Beam_DB,Node_DB
        
       
    def generate_aero_file(self,n_elem, n_node_elem,n_node,Beam_DB,AC_Components_Aerodynamics):

        n_surfaces = len(AC_Components_Aerodynamics)
        # chordiwse panels
        m_distribution = 'uniform'
        
        n_control_surfaces=0
        control_surface_id_list=[]
        for component_aero in AC_Components_Aerodynamics:
            if AC_Components_Aerodynamics[component_aero]['control_surface_id'] != -1:
                if AC_Components_Aerodynamics[component_aero]['control_surface_id'] not in control_surface_id_list:
                    control_surface_id_list.append(AC_Components_Aerodynamics[component_aero]['control_surface_id'])

        n_control_surfaces=len(control_surface_id_list)
        
        
        '''   
        chords [num_elem, num_node_elem]: Chord
        Is an array with the chords of every airfoil given in an element/node basis.
        basically collects the local chord for each node and element
        '''
        chord = np.zeros((n_elem, n_node_elem))
        
        '''   
        elastic_axis [num_elem, num_node_elem]: elastic axis.
        Indicates the elastic axis location with respect to the leading edge as 
        a fraction of the chord of that rib. Note that the elastic axis is already 
        determined, as the beam is fixed now, so this settings controls the location 
        of the lifting surface wrt the beam.
        ''' 
        elastic_axis = np.zeros((n_elem, n_node_elem))
        
        '''   
        twist [num_elem, num_node_elem]: Twist.
        Has the twist angle in radians. It is implemented as a rotation around the local x axis.
        ''' 
        twist = np.zeros((n_elem, n_node_elem))
        
        '''   
        sweep [num_elem, num_node_elem]: Sweep.
        Same here, just a rotation around z.
        ''' 
        sweep = np.zeros((n_elem, n_node_elem))
        
        '''
        surface_distribution_input [num_elem]: Surface integer array.
        It contains the index of the surface the element belongs to. 
        -1 if that section has no control surface associated to it
        Surfaces need to be continuous, so please note that if your beam numbering is not continuous, you need to make a surface per 
        continuous section.    

        this is a vector where 1 is the wing right 2 is the wig left .....
        '''
        surface_distribution = np.zeros((n_elem), dtype=int)
        #initialize to -1
        surface_distribution[:]=-1
        
        '''   
        airfoil_distribution_input [num_elem, num_node_elem]: Airfoil distribution.
        Contains the indices of the airfoils that you put previously in airfoils.
        This points to the airfoil data base
        ''' 
        airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)

        '''  
        surface_m [num_surfaces]: Chordwise panelling.
        Is an integer array with the number of chordwise panels for every surface.
        '''    
        surface_m = np.zeros((n_surfaces ), dtype=int)
        
        
        '''   
        aero_node_input [num_node]: Aerodynamic node definition.
        Is a boolean (True or False) array that indicates if that node has a lifting surface attached to it.
        '''
        aero_node = np.zeros((n_node), dtype=bool)
        aero_node[:] = False
        
     
        '''    
        control_surface [num_elem, num_node_elem]: Control surface.
        Is an integer array containing -1 if that section has no control surface associated to it, 
        and 0, 1, 2 ... if the section belongs to the control surface 0, 1, 2 ... respectively.
        '''
        control_surface = np.zeros((n_elem, n_node_elem), dtype=int)
        #initialize to -1
        control_surface[:,:]=-1

        '''    
        control_surface_type [num_control_surface]: Control Surface type.
        Contains 0 if the control surface deflection is static, and 1 is it is dynamic.
        '''
        control_surface_type = np.zeros((n_control_surfaces ), dtype=int)
        
        '''    
        control_surface_deflection
        Vector with contro surfaces deflection
        '''
        control_surface_deflection = np.zeros((n_control_surfaces ))
        
        '''
        control_surface_chord [num_control_surface]: Control surface chord.
        Is an INTEGER array with the number of panels belonging to the control surface. For example, if M = 4 and you want your control surface to be 0.25c, you need to put 1.
        '''
        control_surface_chord = np.zeros((n_control_surfaces ), dtype=int)
        
        '''   
        control_surface_hinge_coord [num_control_surface]: Control surface hinge coordinate.
        Only necessary for lifting surfaces that are deflected as a whole, like some horizontal tails in some aircraft. Leave it at 0 if you are not modelling this.
        '''
        control_surface_hinge_coord = np.zeros((n_control_surfaces ), dtype=float)
        
        '''
        define aerodynamic_mesh
        '''
        Aerodynamics_keys=[]
        for i in AC_Components_Aerodynamics.keys():
            Aerodynamics_keys.append(i)     
            
        for beam in Beam_DB:
            counter_beam,nastran_node_a,nastran_node_b,sharpy_node_a,sharpy_node_ab ,sharpy_node_b,Rot_Mat,nastran_node_a_pos,nastran_node_ab_pos,nastran_node_b_pos,stiffness,component_beam=beam

            if component_beam in Aerodynamics_keys:
                matrix_aero=AC_Components_Aerodynamics[component_beam]["values"]

                if AC_Components_Aerodynamics[component_beam]["main_direction"]=='y':
                    comp=1
                if AC_Components_Aerodynamics[component_beam]["main_direction"]=='z':
                    comp=2
                
                x=[]
                y=[]
                chord_list=[]
                for value in matrix_aero:
                    x.append(value[0])    
                    y.append(value[comp])
                    chord_list.append(value[3])

                chord_interp = interp1d(y, chord_list, kind='linear',fill_value='extrapolate')
                
                LE_interp = interp1d(y, x, kind='linear',fill_value='extrapolate')

                chord[counter_beam,0]=chord_interp(nastran_node_a_pos[comp])
                chord[counter_beam,1]=chord_interp(nastran_node_b_pos[comp])
                chord[counter_beam,2]=chord_interp(nastran_node_ab_pos[comp])

                elastic_axis[counter_beam,0]=(nastran_node_a_pos[0]   -  LE_interp(nastran_node_a_pos[comp])  ) / chord[counter_beam,0]
                elastic_axis[counter_beam,1]=(nastran_node_b_pos[0]   -  LE_interp(nastran_node_b_pos[comp])  ) / chord[counter_beam,1]
                elastic_axis[counter_beam,2]=(nastran_node_ab_pos[0]  -  LE_interp(nastran_node_ab_pos[comp]) ) / chord[counter_beam,2]
                
                for i in range (3):
                    chord[counter_beam,i]=round(chord[counter_beam,i],3)
                    elastic_axis[counter_beam,i]=round(elastic_axis[counter_beam,i],3)
                
                twist[counter_beam,:]=0. 
                sweep[counter_beam,:]=0.
                    
                aero_surface_indx=Aerodynamics_keys.index(component_beam)
                
                #associate the beam with the aerodynamic compenent
                surface_distribution[counter_beam]=aero_surface_indx 

                
                #define surface_m
                surface_m[aero_surface_indx]=AC_Components_Aerodynamics[component_beam]["npanels_chord"] 
                
                #define airfoil_distribution
                airfoil_distribution[counter_beam,:]=0
                
                #define aero_node
                aero_node[sharpy_node_a]  = True
                aero_node[sharpy_node_b]  = True
                aero_node[sharpy_node_ab] = True 
                aero_node[0] = True 
                
                #associate the control_surface_id to each beam
                control_surface[counter_beam,:]=AC_Components_Aerodynamics[component_beam]["control_surface_id"]
     
       
        '''
        define control_surfaces 
        '''
        control_surface_id_list=[]
        for component_aero in AC_Components_Aerodynamics:
            if AC_Components_Aerodynamics[component_aero]["control_surface"]:
                if AC_Components_Aerodynamics[component_aero]['control_surface_id'] not in control_surface_id_list:

                    id=AC_Components_Aerodynamics[component_aero]['control_surface_id']
                    control_surface_type[id]=AC_Components_Aerodynamics[component_aero]['control_surface_type']
                    control_surface_deflection[id]=AC_Components_Aerodynamics[component_aero]['control_surface_deflection']
                    control_surface_chord[id]=AC_Components_Aerodynamics[component_aero]['control_surface_chord']
                    control_surface_hinge_coord[id]=AC_Components_Aerodynamics[component_aero]['control_surface_hinge_coord']
                    
                    control_surface_id_list.append(id)

        '''       
        airfoils: Airfoil group.
        In the aero.h5 file, there is a Group called airfoils. 
        The airfoils are stored in this group (which acts as a folder) as a two-column 
        matrix with x/c and y/c in each column. They are named '0', '1' , and so on.
        '''        
        def generate_naca_camber(M=0, P=0):
            mm = M*1e-2
            p = P*1e-1

            def naca(x, mm, p):
                if x < 1e-6:
                    return 0.0
                elif x < p:
                    return mm/(p*p)*(2*p*x - x*x)
                elif x > p and x < 1+1e-6:
                    return mm/((1-p)*(1-p))*(1 - 2*p + 2*p*x - x*x)

            x_vec = np.linspace(0, 1, 1000)
            y_vec = np.array([naca(x, mm, p) for x in x_vec])
            return x_vec, y_vec
            
        naca_airfoil_main=generate_naca_camber(P=0, M=0)

        with h5.File(self.case_route + '/' + self.case_name + '.aero.h5', 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            # add one airfoil
            naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(naca_airfoil_main))
     
            
            # chord
            chord_input = h5file.create_dataset('chord', data=chord)
            dim_attr = chord_input .attrs['units'] = 'm'
            
            # twist
            twist_input = h5file.create_dataset('twist', data=twist)
            dim_attr = twist_input.attrs['units'] = 'rad'
            
            # sweep
            sweep_input = h5file.create_dataset('sweep', data=sweep)
            dim_attr = sweep_input.attrs['units'] = 'rad'
            
            # airfoil distribution
            airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)
            
            surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
            surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
            m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))
            #
            aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
            elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)
            #
            control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
            control_surface_deflection_input = h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
            control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
            control_surface_hinge_coord_input = h5file.create_dataset('control_surface_hinge_coord', data=control_surface_hinge_coord)
            control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)
       
      




































                                           
