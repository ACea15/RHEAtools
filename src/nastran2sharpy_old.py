import numpy as np
import pandas as pd
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import src.nastran_extraction as nastran_ex
import src.write_modes
from dataclasses import dataclass
import pprint
##################################################

def ids_condition(id_X, ids, ids_type):

    if ids is None:
        condition = True
    elif ids_type is None:
        condition = True if id_X in ids else False
    elif ids_type == "initial":
        condition = [str(id_X)[:len(id_i)] == str(id_i) for id_i in ids]
        condition = True in condition
    return condition


def beam_template(model:BDF, element_id):

    element = model.elements[element_id]
    prop = model.properties[element.pid]
    E = element.E()
    G = element.G()
    A = prop.A
    Iy = prop.i2
    Iz = prop.i1
    Izy = prop.i12
    J = prop.j
    elem_stiff = []
    elem_stiff.append(element.ga)
    elem_stiff.append(element.gb)
    elem_stiff.append(E * Iy)
    elem_stiff.append(E * Iz)
    elem_stiff.append(G * J)    
    elem_stiff.append(0.) # Yoff
    elem_stiff.append(0.) # Zoff
    elem_stiff.append(E * A)
    elem_stiff.append(0.) # K1y
    elem_stiff.append(0.) # K1z
    elem_stiff.append(E * Izy)
    return elem_stiff

def beam_template2(model:BDF, element_id):

    element = model.elements[element_id]
    prop = model.properties[element.pid]
    E = element.E()
    G = element.G()
    A = prop.A
    Iy = prop.i2
    Iz = prop.i1
    Izy = prop.i12
    J = prop.j
    elem = []
    elem.append(element.ga)
    elem.append(element.gb)
    elem_stiff = np.zeros((6, 6))
    elem_stiff[0, 0] = E * A
    elem_stiff[1, 1] = G * A
    elem_stiff[2, 2] = G * A
    elem_stiff[3, 3] = G * J
    elem_stiff[4, 4] = E * Iz
    elem_stiff[5, 5] = E * Iy
    elem.append(elem_stiff)
    
    return elem

def mass_template(model, node2masses: dict, node_ids: list):

    lumped_mass = []
    lumped_mass_nodes = []
    lumped_mass_inertia = []
    lumped_mass_position = []
    for i, ni in enumerate(node_ids):
        #import pdb; pdb.set_trace()
        for mi in node2masses[ni]:
            lumped_mass.append(model.masses[mi].mass)
            lumped_mass_nodes.append(i)
            lumped_mass_position.append(model.masses[mi].X)
            lumped_mass_inertia.append(np.array(model.masses[mi].Inertia))
    return lumped_mass, lumped_mass_nodes, lumped_mass_position, lumped_mass_inertia

def caeros_template(model:BDF, element_id):
    
    values = []
    element = model.caeros[element_id]
    values.append(np.hstack([element.p1, element.x12]))
    values.append(np.hstack([element.p4, element.x43]))
    return np.array(values)

def add2_listunique(element, list1):
    if element not in list1:
        list1.append(element)

def get_conm2s(model: BDF, node_ids):

    mass_nodes = dict()
    for k, v in model.masses.items():
        if model.masses[k].type == 'CONM2':
            if model.masses[k].nid in mass_nodes.keys():
                mass_nodes[model.masses[k].nid] += [model.masses[k].eid]
            else:
                mass_nodes[model.masses[k].nid] = [model.masses[k].eid]
    return mass_nodes

def extract_beams(model: BDF, stiffness_template:callable, mass_template:callable,
                  ids: list = None, ids_type: str = None) -> np.ndarray:

    beam_values = []
    list_nodes = []
    node_values = []
    for ei in model.element_ids:
        if ids_condition(ei, ids, ids_type):
            beam_values.append(stiffness_template(model, ei))
            add2_listunique(beam_values[-1][0], list_nodes)
            add2_listunique(beam_values[-1][1], list_nodes)
    for ni in list_nodes:
        node_values.append([ni, model.Node(ni).get_position()])
    node_ids = [ni[0] for ni in node_values]
    conm2_dict = get_conm2s(model, node_values)
    (lumped_mass, lumped_mass_nodes, lumped_mass_position,
     lumped_mass_inertia) = mass_template(model, conm2_dict, node_ids)
    
    return (node_values, beam_values, lumped_mass, lumped_mass_nodes,
            lumped_mass_position, lumped_mass_inertia)


def extract_caeros(model: BDF, template:callable,
                   ids: list = None, ids_type: str = None) -> np.ndarray:

    caeros_values = []
    for ei in model.caeros.keys():
        if ids_condition(ei, ids, ids_type):
            caeros_values.append(template(model, ei))
    return np.array(caeros_values)

def merge_caeros_list(caeros, tolerance=1e-3):

    merge_dict = dict()
    num_caeros = len(caeros)
    for i in range(num_caeros):
        caeros_other = list(range(num_caeros))
        caeros_other.remove(i)
        for j in caeros_other:
            if (np.linalg.norm(caeros[j][0,:3] - (caeros[i][0,:3] +
                                                  np.array([caeros[i][0,3], 0., 0.]))) +
                np.linalg.norm(caeros[j][1,:3] - (caeros[i][1,:3] +
                                                  np.array([caeros[i][1,3], 0., 0.])))
                < tolerance):
                if j in merge_dict.keys():
                    merge_dict[j].append(i)
                else:
                    merge_dict[i] = [j]
    return merge_dict


def merge_caeros(caeros: np.ndarray, merge_dict: dict):

    caeros_new = list()
    merged_caeros = list()
    for i, ci in enumerate(caeros):
        if i in merge_dict.keys():
            merged_caeros += merge_dict[i]
            sum_p1 = ci[0,3]
            sum_p3 = ci[1,3]
            for j in merge_dict[i]:
                sum_p1 += caeros[i][0,3]
                sum_p3 += caeros[i][1,3]
            caeros_new.append(np.array([np.array([ci[0,0], ci[0, 1], ci[0, 2], sum_p1]),
                                        np.array([ci[1,0], ci[1, 1], ci[1, 2], sum_p3])]))
        elif i in merged_caeros:
            pass
        else:
            caeros_new.append(ci)

    return caeros_new

# i = 6
# (caeros_new[i][1,:3] - caeros_new[i][0,:3]) / np.linalg.norm((caeros_new[i][1,:3] - caeros_new[i][0,:3]))


# v = (model.Nodes([5023])[0].get_position()- model.Nodes([5022])[0].get_position()) /(
#     np.linalg.norm(model.Nodes([5023])[0].get_position()- model.Nodes([5022])[0].get_position()))



@dataclass
class BeamElement:

    node_ids: list
    node_coord: list[np.ndarray]
    stiffness: object
    mass: object
    orientation_vector: np.ndarray
    Rab: np.ndarray = None
    x: np.ndarray = None
    y: np.ndarray = None
    z: np.ndarray = None

    def __post_init__(self):

        self.Rab = self.caculate_rotation()
        self.x = self.Rab[:, 0]
        self.y = self.Rab[:, 1]
        self.z = self.Rab[:, 2]

    def caculate_rotation(self):

        x = (self.node_coord[-1] - self.node_coord[0])
        x /= np.linalg.norm(x)
        y = self.orientation_vector - (self.orientation_vector.dot(x))*x
        y /= np.linalg.norm(y)
        z= np.cross(x, y)
        return np.array([x, y, z]).T

@dataclass
class LumpedMass:

    lumped_mass: float
    lumped_mass_node: np.ndarray
    lumped_mass_index: int
    lumped_mass_position: np.ndarray
    lumped_mass_inertia: np.ndarray
    element_index: int

@dataclass
class BeamComponent:

    _model: BDF
    element_ids: list
    element_ids_type: list = None
    dummy_mass: float = None
    elements: list[BeamElement] = None
    conm2s: list[LumpedMass] = None    
    _node_ids: list = None
    _node_coord: list[np.ndarray] = None
    ref_nodes: list =None
    mapper_conm2: object = None
    node2elem: dict = None
    name: str = "default"

    def __post_init__(self):

        self.extract_elements()
        self.mapper_conm2 = self.map_conm2s(self._model, self._node_ids)
        self.extract_conm2s()
        
    @property
    def node_ids(self):
        return _node_ids

    @property
    def node_coord(self):
        return _node_coord

    def _geometry_template(self, element_id: int) -> tuple:
        """Get the nodes, coordinates and cross-sectional vector of the element

        The orientation vector method in PyNastran may not be
        robust. Here a good explanation of its meaning
        http://mscnastrannovice.blogspot.com/2013/06/understanding-cbeam-and-cbar-beam.html.

        Parameters
        ----------
        element_id : int
            The Nastran id of the element

        """

        element = self._model.elements[element_id]
        node_ids = element.node_ids
        nodes_coord = []
        for ni in self._model.Nodes(node_ids):
            nodes_coord.append(ni.get_position())
        nodes_coord = np.array(nodes_coord)
        try:
            orientation_vector = element.get_orientation_vector('y')
        except: # default value
            orientation_vector = np.array([0., 0., 1.])
            
        return node_ids, nodes_coord, orientation_vector 
    
    def _stiffness_template(self, element_id):
        """Get the stiffness associated to the input element

        6x6 matrix of the beam in the local frame of the Nastran
        element

        Parameters
        ----------
        element_id : int
            The Nastran id of the element

        TODO: Deal with shear stiffness and non-diagonal terms

        """

        element = self._model.elements[element_id]
        prop = self._model.properties[element.pid]
        E = element.E()
        G = element.G()
        A = prop.A
        Iy = prop.i2  # element.I22() too...
        Iz = prop.i1
        Izy = prop.i12
        J = prop.j
        elem_stiff = np.zeros((6, 6))
        elem_stiff[0, 0] = E * A
        elem_stiff[1, 1] = G * A
        elem_stiff[2, 2] = G * A
        elem_stiff[3, 3] = G * J
        elem_stiff[4, 4] = E * Iy
        elem_stiff[5, 5] = E * Iz
        elem_stiff[4, 5] = elem_stiff[5, 4] = E * Izy

        return elem_stiff

    def _mass_template(self, element_id):
        
        element = self._model.elements[element_id]
        prop = self._model.properties[element.pid]
        if self.dummy_mass is not None:
            mass = self.dummy_mass
        else:
            mass = element.MassPerLength()
        #A = prop.A
        # Rho = prop.Rho()
        # mass = Rho * A #  But it does not include
        # non-structural-mass (NSM in PBAR)
        Iy = prop.i2
        Iz = prop.i1
        Izy = prop.i12
        J = prop.j
        elem_mass = np.zeros((6, 6))
        elem_mass[0, 0] = mass
        elem_mass[1, 1] = mass
        elem_mass[2, 2] = mass
        elem_mass[3, 3] = mass * J
        elem_mass[4, 4] = mass * Iy
        elem_mass[5, 5] = mass * Iz

        return elem_mass

    def extract_elements(self):
        """Extracts the elements geometry, stiffness and mass information.

        Builds the elements variable. It also saves a reference node for each element (Ga in
        Nastran) for local coordinates information and the global list
        of node ids in the component

        """

        list_nodes = []
        self.elements = []
        self.node2elem = dict()
        for i, ei in enumerate(self._model.element_ids): 
            # Not efficient but allows more flexible input of elements ids
            if self.ids_condition(ei, self.element_ids, self.element_ids_type):
                nodes, nodes_coord, orientation_vector = self._geometry_template(ei)
                stiffness = self._stiffness_template(ei)
                mass = self._mass_template(ei)
                self.elements.append(BeamElement(nodes, nodes_coord,
                                                 stiffness, mass, orientation_vector))
                for ni in nodes:
                    add2_listunique(ni, list_nodes)
                    if ni not in self.node2elem.keys():
                        self.node2elem[ni] = i

        self._node_ids = list_nodes

    def extract_conm2s(self):
        """Extracts the CONM2s associated to the component.

        Builds the conm2s variable. Importantly it saves a reference node that will be needed if
        the position or inertia of the lumped mass (given in the
        global system or in CID entry) is to be changed later on

        """
        self.conm2s = []
        for i, ki in enumerate(self.mapper_conm2.keys()): # {nid:[conm2s_ids]}
            #import pdb; pdb.set_trace()
            for mj in self.mapper_conm2[ki]:
                assert self._model.masses[mj].Cid() == 0, f"mass not defined with respect \
                to the global reference system (Id, Cid={mj, self._model.masses[mj].Cid()}),\
                this function does not take care of this yet"
                lumped_mass = self._model.masses[mj].mass
                lumped_mass_node = ki
                lumped_mass_index = i
                lumped_mass_position = self._model.masses[mj].X
                lumped_mass_inertia = np.array(self._model.masses[mj].Inertia())
                associated_node = self.node2elem[lumped_mass_node] # connects lumped masses
                # to structural model
                self.conm2s.append(LumpedMass(lumped_mass, lumped_mass_node, lumped_mass_index,
                                              lumped_mass_position, lumped_mass_inertia,
                                              associated_node))

    @staticmethod    
    def ids_condition(id_X, ids, ids_type):
        """Defines the condition as to whether id_X is in ids

        Parameters
        ----------
        id_X : int
            element id
        ids : None
            True list of ids or string representing the set of
            possible ids. E.g. 2011 in [2022, 2011] is True; 2011 in
            '20' is True, but 2011 in '21' is False
        ids_type : str
            triggers the string condition if equal to "initial"

        """

        if ids is None:
            condition = True
        elif ids_type is None:
            condition = True if id_X in ids else False
        elif ids_type == "initial":
            condition = [str(id_X)[:len(id_i)] == str(id_i) for id_i in ids]
            condition = True in condition
        return condition
    
    @staticmethod
    def map_conm2s(model: BDF, node_ids: list) -> dict:
        """Creates a dictionary mapping all the CONM2 associated to each node id

        Parameters
        ----------
        model : BDF
        node_ids : list
            List of node ids where CONM2 are attached to

        Returns
        -------
        dict
            Dictionary to map node ids to CONM2s ids

        """

        mass_nodes = dict()
        for k, v in model.masses.items():
            if (model.masses[k].type == 'CONM2' and
                model.masses[k].nid in node_ids):
                if model.masses[k].nid in mass_nodes.keys():
                    mass_nodes[model.masses[k].nid] += [model.masses[k].eid]
                else:
                    mass_nodes[model.masses[k].nid] = [model.masses[k].eid]
        return mass_nodes

    def do_pprint(self, **kwargs):

        pprint.pprint(self, **kwargs)



    
if (__name__ == "__main__"):
    model = BDF()
    #model.read_bdf("./models/nastran/")
    model.read_bdf("../data/in/SOL103/polimi-103cam.bdf")
    model = BDF()
    #model.read_bdf("./models/nastran/")
    model.read_bdf("../data/in/SOL145/polimi-145cam_078M.bdf")

    # print(model.get_bdf_stats())

    (nodes, beams, lumped_mass, lumped_mass_nodes, lumped_mass_position,
     lumped_mass_inertia) = extract_beams(model, beam_template2, mass_template)
    node_ids = [n[0] for n in nodes]
    
    caeros = extract_caeros(model, caeros_template)
    md1 = merge_caeros_list(caeros)
    caeros_new = merge_caeros(caeros, md1)




    # AC_Components_nodes  =  { 

    #     "CWB_right" : {
    #       "main_direction": 'y',
    #       "values":       [\
    #                       [1      ,21.  ,0.     ,0. ],\
    #                       [801    ,21.5 ,1.0    ,-1.],\
    #                       ]
    #     },    

    # AC_Components_Stiffness  =  { 

    #     "CWB_right" : {
    #       "values":       [\
    #                          #Nodes_A      Nodes_B             EIy           EIz               GJ                Yoff             Zoff             AE                   K1y                K1z               EIzy 
    #                       [    1       ,     801     ,       5.e+9     ,   1.e+9       ,     5.e+8       ,      0.          ,     0.         ,    5.e9      ,        0.         ,        0.         ,      0.     ],\
    #                       ] 
    #     },


    # AC_Components_Aerodynamics  =  {  



    #     "Wing_right" : {
    #       "main_direction": 'y',
    #       "npanels_chord": 15,
    #       "control_surface": False,
    #       "control_surface_id": -1,
    #       "CS_chord": 0,
    #       "values":       [\
    #                       #   x       y         z       chord
    #                       [  20. ,  0.    ,  -1.  ,  5.],\
    #                       [  22.5,  5.    ,  -1.  ,  5.],\
    #                       [  27.5,  15.   ,  -1.  ,  5.],\
    #                       [  33.5,  27.   ,  -1.  ,  5.],\
    #                       ]
    #     },
