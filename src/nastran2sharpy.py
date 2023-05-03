import numpy as np
import pandas as pd
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import src.nastran_extraction as nastran_ex
import src.write_modes
from dataclasses import dataclass
import pprint
##################################################


# i = 6
# (caeros_new[i][1,:3] - caeros_new[i][0,:3]) / np.linalg.norm((caeros_new[i][1,:3] - caeros_new[i][0,:3]))


# v = (model.Nodes([5023])[0].get_position()- model.Nodes([5022])[0].get_position()) /(
#     np.linalg.norm(model.Nodes([5023])[0].get_position()- model.Nodes([5022])[0].get_position()))

@dataclass
class AeroComponent:

    _model: BDF
    element_ids: list = None
    ids_type: str = None
    merged_tolerance: float = 1e-3
    perform_merging: bool = True
    _aero_panel = None
    name: str = "default"

    def __post_init__(self):
        self.caeros = None
        self.merge_dict = None
        self.caeros_merged = None
        self.extract_elements()

    def extract_elements(self):
        """Extracts CAEROS

        """
        self.extract_caeros()
        if self.perform_merging:
            self.merge_dict = self.merge_caeros_list(self.caeros,
                                                     self.merged_tolerance)
            self.merge_caeros()

    def caeros_template(self, element_id):

        values = []
        element = self._model.caeros[element_id]
        values.append(np.hstack([element.p1, element.x12]))
        values.append(np.hstack([element.p4, element.x43]))
        return np.array(values)

    def extract_caeros(self):

        caeros_values = []
        for ei in self.element_ids:
            caeros_values.append(self.caeros_template(ei))
        self.caeros = np.array(caeros_values)

    @staticmethod
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
                                                      np.array([caeros[i][1,3], 0., 0.]))) <
                    tolerance):
                    if j in merge_dict.keys():
                        merge_dict[j].append(i)
                    else:
                        merge_dict[i] = [j]
        return merge_dict

    def merge_caeros(self):

        self.caeros_merged = list()
        merged_caeros = list()
        for i, ci in enumerate(self.caeros):
            if i in self.merge_dict.keys():
                merged_caeros += self.merge_dict[i]
                sum_p1 = ci[0,3]
                sum_p3 = ci[1,3]
                for j in self.merge_dict[i]:
                    sum_p1 += self.caeros[j][0,3]
                    sum_p3 += self.caeros[j][1,3]
                self.caeros_merged.append(np.array([np.array([ci[0,0],
                                                              ci[0, 1],
                                                              ci[0, 2],
                                                              sum_p1]),
                                                    np.array([ci[1,0],
                                                              ci[1, 1],
                                                              ci[1, 2],
                                                              sum_p3])]))
            elif i in merged_caeros:
                pass
            else:
                self.caeros_merged.append(ci)
        self.caeros_merged = np.array(self.caeros_merged)

    @property
    def aero_panel(self):
        if self._aero_panel is None:
            if self.caeros_merged is None:
                return self.caeros[0]
            else:
                return self.caeros_merged[0]
        else:
            return self._aero_panel

    @aero_panel.setter
    def aero_panel(self, value):

        self._aero_panel = value
        
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
        y = self.orientation_vector - (self.orientation_vector.dot(x)) * x
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
    node_ids: list = None
    node_coord: np.ndarray = None
    mapper_conm2: object = None
    node2elem: dict = None
    name: str = "default"

    def __post_init__(self):

        self.extract_elements()
        self.mapper_conm2 = self.map_conm2s(self._model, self.node_ids)
        self.extract_conm2s()

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

    def add2_listunique(self, ni, ni_coord, list_nodes, list_coord):
        if ni not in list_nodes:
            list_nodes.append(ni)
            list_coord.append(ni_coord)


    def extract_elements(self):
        """Extracts the elements geometry, stiffness and mass information.

        Builds the elements variable. It also saves a reference node for each element (Ga in
        Nastran) for local coordinates information and the global list
        of node ids in the component

        """
        #import pdb; pdb.set_trace()
        list_nodes = []
        list_nodes_coord = []
        self.elements = []
        self.node2elem = dict()
        for i, ei in enumerate(self.element_ids):
            nodes, nodes_coord, orientation_vector = self._geometry_template(ei)
            stiffness = self._stiffness_template(ei)
            mass = self._mass_template(ei)
            self.elements.append(BeamElement(nodes, nodes_coord,
                                             stiffness, mass, orientation_vector))
            for j, nj in enumerate(nodes):
                self.add2_listunique(nj, nodes_coord[j],  list_nodes, list_nodes_coord) #
                if nj not in self.node2elem.keys(): #maps a node to an element
                    self.node2elem[nj] = i

        self.node_ids = list_nodes
        self.node_coord = np.array(list_nodes_coord)

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

@dataclass
class NastranModel:

    _model: BDF
    beam_components: dict
    aero_components: dict = None
    name: str = "default"

    def __post_init__(self):
        
        self.beam_names = self.beam_components.keys()
        self.beam = dict()
        self.aero = dict()
        if self.aero_components is not None:        
            self.aero_names = self.aero_components.keys()
        else:
            self.aero_names = None
        for k, v in self.beam_components.items():
            self.beam[k] = BeamComponent(self._model,
                                         **v)
        for k, v in self.aero_components.items():
            self.aero[k] = AeroComponent(self._model,
                                         **v)
        self.make_aero_panel()

    def check_aeros_plane(self):

        for k in self.aero_names:
            print('#############')
            print(f'# {k} #')
            print('#############')
            for ci in self.aero[k].caeros:
              v1 = ci[1, :3] - ci[0, :3]
              v2 = (np.array([ci[0, 3], 0., 0.]))
              v1n = v1 / np.linalg.norm(v1)
              v2n = v2 / np.linalg.norm(v2)
              print(np.cross(v1n, v2n))
    
    def make_aero_panel(self):

        for k in self.aero_names:
            if self.aero[k].caeros_merged is None:
                aero_panel = np.array([self.aero[k].caeros[0][0],
                                       self.aero[k].caeros[-1][1]])
            else:
                aero_panel = np.array([self.aero[k].caeros_merged[0][0],
                                       self.aero[k].caeros_merged[-1][1]])

            self.aero[k].aero_panel = aero_panel
            
class Sharpy_generator:

    def __init__(self, model: NastranModel,
                 settings,
                 assembly,
                 model_name: str = 'Model',
                 model_route: str = None,
                 extra_components: list = None
                 ):

        self.model = model
        self.settings = settings
        self.assembly = assembly
        self.model_name = model_name
        self.model_route = model_route
        self.beam_names = self.model.beam_names
        self.aero_names = self.model.aero_names
        self.component_settings = {k: dict(fem={}, aero={})
                                   for k in self.beam_names}
        self.component_Rab = {k: list()
                              for k in self.beam_names}
        self.component_Rbn = {k: list()
                              for k in self.beam_names}

        self._build_components()
        self._build_model_settings()
        if extra_components is not None:
            for di in extra_components:
                self._add_component(**di)

    def _add_component(self, name, **kwargs):
        self.component_settings[name] = dict(**kwargs)

    def _add_component_sett(self, component):

        ci = component
        if 'merge_surface' in self.settings[ci]:
            self.component_settings[ci]['aero']['merge_surface'] = \
                self.settings[ci]['merge_surface']
        else:
            self.component_settings[ci]['aero']['merge_surface'] = False
        if 'delta_frame' in self.settings[ci]:
            self.component_settings[ci]['fem']['frame_of_reference_delta'] = \
                self.settings[ci]['delta_frame']
        if 'surface_m' in self.settings[ci]:
            self.component_settings[ci]['aero']['surface_m'] = \
                self.settings[ci]['surface_m']

    def _build_model_settings(self):

        self.model_settings = dict()
        self.model_settings['model_name'] = self.model_name
        self.model_settings['model_route'] = self.model_route
        self.model_settings['assembly'] = self.assembly

    def _build_components(self):

        for i, k in enumerate(self.beam_names):
            self._build_geometry(k)
            self._buid_for(k)
            self._build_fem(k)
            self._build_lumpedmass(k, i)
            if self.aero_names is None:
                self.component_settings[k]['workflow'] = \
                    ['create_structure']
            elif k in self.aero_names:
                self._build_aero(k)
                self.component_settings[k]['workflow'] = \
                    ['create_structure', 'create_aero']
            else:
                self.component_settings[k]['workflow'] = \
                    ['create_structure', 'create_aero0']
            self._bring_panels2beam(k)
            if ('transverse_reverse' in self.settings[k] and
                self.settings[k]['transverse_reverse']):
                self._reverse_component(k)
            self._force_node0(k)
            self._add_component_sett(k)

    def _reverse_component(self, component):

        ci = component
        self.component_settings[ci]['fem']['coordinates'] = \
            np.flip(self.component_settings[ci]['fem']['coordinates'],
                    axis=0)
        self.component_settings[ci]['fem']['stiffness_db'] = \
            np.flip(self.component_settings[ci]['fem']['stiffness_db'],
                    axis=0)
        self.component_settings[ci]['fem']['mass_db'] = \
            np.flip(self.component_settings[ci]['fem']['mass_db'],
                    axis=0)
        if 'lumped_mass' in self.component_settings[ci]['fem'].keys():
            self.component_settings[ci]['fem']['lumped_mass'] = \
                np.flip(self.component_settings[ci]['fem']['lumped_mass'],
                        axis=0)
        if 'lumped_mass_inertia' in self.component_settings[ci]['fem'].keys():
            self.component_settings[ci]['fem']['lumped_mass_inertia'] = \
                np.flip(self.component_settings[ci]['fem']['lumped_mass_inertia'],
                        axis=0)
        if 'lumped_mass_position' in self.component_settings[ci]['fem'].keys():
            self.component_settings[ci]['fem']['lumped_mass_position'] = \
                np.flip(self.component_settings[ci]['fem']['lumped_mass_position'],
                        axis=0)
        self.component_settings[ci]['aero']['beam_origin'] = \
            self.model.beam[ci].node_coord[-1]
        #swap aero 4-corner points
        old_leading_edge1 = np.copy(
            self.component_settings[ci]['aero']['point_platform']['leading_edge1'])
        old_leading_edge2 = np.copy(
            self.component_settings[ci]['aero']['point_platform']['leading_edge2'])
        old_trailing_edge1 = np.copy(
            self.component_settings[ci]['aero']['point_platform']['trailing_edge1'])
        old_trailing_edge2 = np.copy(
            self.component_settings[ci]['aero']['point_platform']['trailing_edge2'])
        self.component_settings[ci]['aero']['point_platform']['leading_edge2'] = \
            old_leading_edge1
        self.component_settings[ci]['aero']['point_platform']['leading_edge1'] = \
            old_leading_edge2
        self.component_settings[ci]['aero']['point_platform']['trailing_edge2'] = \
            old_trailing_edge1
        self.component_settings[ci]['aero']['point_platform']['trailing_edge1'] = \
            old_trailing_edge2
        
    def _force_node0(self, component):
        self.component_settings[component]['fem']['coordinates'] -= \
            self.component_settings[component]['fem']['coordinates'][0]
        
    def _bring_panels2beam(self, component):
        """Brings the aero panels to the beam so that both lie in the same plane  

        Parameters
        ----------
        component : str
            Name of the component

        """
        
        ci = component
        node_first = self.model.beam[ci].node_coord[0]
        node_last = self.model.beam[ci].node_coord[-1]
        delta_ylfirst = (node_first[1] -
                        self.component_settings[ci]['aero']['point_platform']['leading_edge1'][1])
        delta_zlfirst = (node_first[2] -
                        self.component_settings[ci]['aero']['point_platform']['leading_edge1'][2])
        delta_yllast = (node_last[1] -
                        self.component_settings[ci]['aero']['point_platform']['leading_edge2'][1])
        delta_zllast = (node_last[2] -
                        self.component_settings[ci]['aero']['point_platform']['leading_edge2'][2])
        delta_ytfirst = (node_first[1] -
                        self.component_settings[ci]['aero']['point_platform']['trailing_edge1'][1])
        delta_ztfirst = (node_first[2] -
                        self.component_settings[ci]['aero']['point_platform']['trailing_edge1'][2])
        delta_ytlast = (node_last[1] -
                        self.component_settings[ci]['aero']['point_platform']['trailing_edge2'][1])
        delta_ztlast = (node_last[2] -
                        self.component_settings[ci]['aero']['point_platform']['trailing_edge2'][2])

        self.component_settings[ci]['aero']['point_platform']['leading_edge1'] += \
            np.array([0., delta_ylfirst, delta_zlfirst])        
        self.component_settings[ci]['aero']['point_platform']['trailing_edge1'] += \
            np.array([0., delta_ytfirst, delta_ztfirst])        
        self.component_settings[ci]['aero']['point_platform']['leading_edge2'] += \
            np.array([0., delta_yllast, delta_zllast])        
        self.component_settings[ci]['aero']['point_platform']['trailing_edge2'] += \
            np.array([0., delta_ytlast, delta_ztlast])        

    def _build_geometry(self, component: str):

        nastran_component = self.model.beam[component]
        num_coord = len(nastran_component.node_coord)
        coordinates = []
        for i in range(num_coord - 1):
            coordinates.append(nastran_component.node_coord[i])
            coordinates.append((nastran_component.node_coord[i +1] +
                                nastran_component.node_coord[i]) / 2)
        coordinates.append(nastran_component.node_coord[i + 1])
        self.component_settings[component]['fem']['coordinates'] = \
            np.array(coordinates)
        self._amend_coordinates(component)
        
    def _amend_coordinates(self, component: str):
        
        c1 = self.component_settings[component]['fem']['coordinates'][0]
        c2 = self.component_settings[component]['fem']['coordinates'][-1]
        c21 = (c2 - c1) / np.linalg.norm(c2 - c1)
        self.component_settings[component]['fem']['coordinates'] = \
            c1 + np.array([np.dot(c21, ci) *c21
                           for ci in self.component_settings[component]['fem']['coordinates']])

    def _buid_for(self, component: str):

        self.component_settings[component]['fem']['frame_of_reference_delta'] = \
            self.settings[component]['delta_frame']
        nastran_component = self.model.beam[component]
        for ei in nastran_component.elements:
            x = ei.Rab[:, 0]
            z = np.cross(x, self.settings[component]['delta_frame'])
            z /= np.linalg.norm(z)
            y = np.cross(z, x)
            Rba = np.array([x, y, z])
            self.component_Rab[component].append(Rba.T)
            Ran = ei.Rab
            self.component_Rbn[component].append(Rba.dot(Ran))

    def _build_fem(self, component: str):

        nastran_component = self.model.beam[component]
        self.component_settings[component]['fem']['stiffness_db'] = list()
        self.component_settings[component]['fem']['mass_db'] = list()
        for i, ei in enumerate(nastran_component.elements):
            Rbn = self.component_Rbn[component][i]
            Rbn6 = np.block([[Rbn, np.zeros((3,3))],
                             [np.zeros((3,3)), Rbn]])
            self.component_settings[component]['fem']['stiffness_db'].append(
                Rbn6.dot(ei.stiffness.dot(Rbn6.T)))
            self.component_settings[component]['fem']['mass_db'].append(
                Rbn6.dot(ei.mass.dot(Rbn6.T)))
        self.component_settings[component]['fem']['stiffness_db'] = np.array(
            self.component_settings[component]['fem']['stiffness_db'])
        self.component_settings[component]['fem']['mass_db'] = np.array(
            self.component_settings[component]['fem']['mass_db'])

    def _build_lumpedmass(self, component: str, number:int):

        nastran_component = self.model.beam[component]
        #node0 is needed to put the lumped mass of the first element correctly in the first
        # or second nodes
        masses_1element =[mi.lumped_mass_node for mi in nastran_component.conm2s if
                          mi.element_index == 0]
        node0 = [min(masses_1element), max(masses_1element)]
        masses_1node = [mi.lumped_mass_node for mi in nastran_component.conm2s]
        num_masses_1node = masses_1node.count(min(masses_1node))
        self.component_settings[component]['fem']['lumped_mass'] = list()
        self.component_settings[component]['fem']['lumped_mass_inertia'] = list()
        self.component_settings[component]['fem']['lumped_mass_nodes'] = list()
        self.component_settings[component]['fem']['lumped_mass_position'] = list()
        for i, mi in enumerate(nastran_component.conm2s):
            if number == 0 or i >= num_masses_1node: # condition to not repeat nodes at the junction
                # except for the first element
                self.component_settings[component]['fem']['lumped_mass'].append(
                    mi.lumped_mass)
                self.component_settings[component]['fem']['lumped_mass_inertia'].append(
                    mi.lumped_mass_inertia)
                self.component_settings[component]['fem']['lumped_mass_position'].append(
                    self.component_Rab[component][mi.element_index].T.dot(mi.lumped_mass_position))
                if mi.element_index == 0 and mi.lumped_mass_node == node0[0]:
                    self.component_settings[component]['fem']['lumped_mass_nodes'].append(0)
                elif mi.element_index == 0 and mi.lumped_mass_node == node0[1]:
                    self.component_settings[component]['fem']['lumped_mass_nodes'].append(2)
                else:
                    self.component_settings[component]['fem']['lumped_mass_nodes'].append(
                        2*(mi.element_index + 1))

    def _build_aero(self, component: str):

        nastran_component = self.model.aero[component]
        ci = component
        self.component_settings[ci]['aero']['point_platform'] = dict()
        self.component_settings[ci]['aero']['beam_origin'] = \
            self.model.beam[ci].node_coord[0]
        self.component_settings[ci]['aero']['point_platform']['leading_edge1'] = \
            nastran_component.aero_panel[0, :3]
        self.component_settings[ci]['aero']['point_platform']['leading_edge2'] = \
            nastran_component.aero_panel[1, :3]
        self.component_settings[ci]['aero']['point_platform']['trailing_edge1'] = \
            (nastran_component.aero_panel[0, :3] +
             np.array([nastran_component.aero_panel[0, 3], 0, 0]))
        self.component_settings[ci]['aero']['point_platform']['trailing_edge2'] = \
            (nastran_component.aero_panel[1, :3] +
             np.array([nastran_component.aero_panel[1, 3], 0, 0]))

if (__name__ == "__main__"):
    #model103 = BDF()
    #model.read_bdf("./models/nastran/")
    #model103.read_bdf("../data/in/SOL103tailless/polimi-103cam.bdf")
    model145 = BDF()
    #model.read_bdf("./models/nastran/")
    model145.read_bdf("../data/in/SOL145tailless/polimi-145cam_078M.bdf")
    # wing_r = BeamComponent(model145, list(range(2000, 2035)))
    # wing_ra = AeroComponent(model145, [1001, 3001, 4001,
    #                          7001, 8001, 11001, 12001])

    beam_components = dict(wing_r1=dict(
        element_ids=[2000, 2001]),
                           wing_r2=dict(
        element_ids=list(range(2002, 2035))),
                           strut2=dict(
        element_ids=list(range(5002, 5023))),
                           strut1=dict(
        element_ids=[5000, 5001]),
                           )

    aero_components = dict(wing_r1=dict(
        element_ids=[1001]),
                           wing_r2=dict(
        element_ids=[3001, 4001, 7001,
                     8001, 11001, 12001]),
                           strut2=dict(
        element_ids=[33001,35001]),                           
                           strut1=dict(
        element_ids=[31001])
                           )

    nas_model =  NastranModel(model145, beam_components, aero_components)
    ##############################################
    _settings = dict(wing_r1=dict(delta_frame=np.array([-1, 0, 0]),
                                  merge_surface=False,
                                  transverse_reverse=False,
                                  surface_m=9
                                  ),
                     wing_r2=dict(delta_frame=np.array([-1, 0, 0]),
                                  merge_surface=True,
                                  transverse_reverse=False,
                                  surface_m=9
                                  ),
                     strut2=dict(delta_frame=np.array([1, 0, 0]),
                                 merge_surface=False,
                                 transverse_reverse=True,
                                 surface_m=9
                                 ),                     
                     strut1=dict(delta_frame=np.array([1, 0, 0]),
                                 merge_surface=True,
                                 transverse_reverse=True,
                                 surface_m=9
                                 )
                     )
    join_direction = (nas_model.beam['strut1'].node_coord[0] - 
                      nas_model.beam['wing_r1'].node_coord[0])
    join_length = np.linalg.norm(join_direction)
    component_addition = [dict(name='join',
                              fem=dict(stiffness_db=np.array([1e11*np.eye(6)]),
                                       mass_db=np.array([0.1*np.eye(6)])
                                       ),
                              geometry=dict(direction=join_direction,
                                            length=join_length,
                                            num_node=3),
                              workflow=['create_structure', 'create_aero0'])]
    assembly = {'include_aero': 1,
                'default_settings': 1,  
                'wing_r1': {'upstream_component': '',
                            'node_in_upstream': 0},
                'wing_r2': {'keep_aero_node': 1,
                            'upstream_component': 'wing_r1',
                            'node_in_upstream': -1},
                'strut2': {'upstream_component': 'wing_r2',
                           'node_in_upstream': 20 + 20}, #CAREFUL! SHARPy has 3-noded beams
                'strut1': {'keep_aero_node': 1,
                           'upstream_component': 'strut2',
                           'node_in_upstream': -1},
                'join': {'upstream_component': 'wing_r1',
                         'node_in_upstream': 0,
                         'chained_component': ['strut1', -1]}
                }

    add_mirror = True
    if add_mirror:
        component_addition.append(dict(name="wing_l1",
                                       symmetric={'component': 'wing_r1'}))
        component_addition.append(dict(name="wing_l2",
                                       symmetric={'component': 'wing_r2'}))
        component_addition.append(dict(name="strut2l",
                                       symmetric={'component': 'strut2'}))
        component_addition.append(dict(name="strut1l",
                                       symmetric={'component': 'strut1'}))

        assembly2 = {'wing_l1': {'keep_aero_node': 1,
                                 'upstream_component': 'wing_r1',
                                 'node_in_upstream': 0},
                     'wing_l2': {'keep_aero_node': 1,
                                 'upstream_component': 'wing_l1',
                                 'node_in_upstream': -1},
                     'strut2l': {'keep_aero_node': 1,
                                 'upstream_component': 'wing_l2',
                                'node_in_upstream': 20 + 20}, #CAREFUL! SHARPy has 3-noded beams
                     'strut1l': {'keep_aero_node': 1,
                                'upstream_component': 'strut2l',
                                 'node_in_upstream': -1,
                                 'chained_component': ['strut1', -1]}
                     }
        assembly = assembly | assembly2
        
    sh_model = Sharpy_generator(nas_model, _settings, assembly,
                                extra_components=component_addition,
                                model_route="/home/ac5015/programs/RHEAtools/examples/")

    import pickle
    with open("../examples/sharpy_component.pkl", "wb") as f:
        pickle.dump(sh_model.component_settings, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../examples/model_settings.pkl", "wb") as f:
        pickle.dump(sh_model.model_settings, f, protocol=pickle.HIGHEST_PROTOCOL)

    
