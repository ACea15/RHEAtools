import numpy as np
import pathlib
import pyvista
from dataclasses import dataclass
from pyNastran.bdf.mesh_utils.mass_properties import mass_properties
from pyNastran.bdf.bdf import BDF

# function to build symmetry 
get_symmetry = lambda v, si: np.vstack([v[0,:3], v[si:, :3]])

# index of node id x in the list 
index_node = lambda x, modal_node_ids: np.where(modal_node_ids == x)[0][0]

def extract_rbes(model: BDF):
    """Extract rigid elements from Nastran model

    Parameters
    ----------
    model : BDF
        Instance of pyNastran BDF class

    
    """

    rbe2_ids=[];rbe3_ids=[]
    rbe3_dependent=[]
    rbe3_independent=[]
    idp2dp={}
    for i in model.rigid_elements.keys():

        if  model.rigid_elements[i].type == 'RBE3':
            rbe3_ids.append(i)
            dp=model.rigid_elements[i].dependent_nodes[0]
            idp=model.rigid_elements[i].independent_nodes
            rbe3_dependent.append(dp)
            rbe3_independent+=idp
            for j in idp:
                idp2dp[j]=[]
            for j in idp:
                idp2dp[j]+=[dp]
        elif  model.rigid_elements[i].type == 'RBE2':

            rbe2_ids.append(i)
    return rbe2_ids,rbe3_ids,rbe3_dependent,idp2dp

def extract_aero(model: BDF, rbe2s: list,
                 epsilon: float = 0.2, plane_vect: list = [0,0,1]):
    """Extract beam, leading-edge and trailing-edge aero points from RBE2s


    Parameters
    ----------
    model : BDF

    rbe2s : list
        List of RBE2s ids associated to the aero-elements
    epsilon : float
        tolerance to decide whether the RBE2 node is in the aero plane
    plane_vect : list
        vector out-of-plane aero-panels

    Returns
    --------


    """

    le_ = []
    te_ = []
    beam_ = []
    le__coord = []
    te__coord = []
    beam__coord = []
    epsilon = 0.2
    for rbe2i in rbe2s:

        node = model.rigid_elements[rbe2i].Gn()
        node_coord = model.Node(node).get_position()
        beam_.append(node)
        beam__coord.append(node_coord)
        for j in model.rigid_elements[rbe2i].Gmi:
            node_coord_j = model.Node(j).get_position()
            if abs(np.dot(node_coord_j - node_coord, plane_vect)
                   / np.linalg.norm((node_coord_j - node_coord))) < epsilon:
                if node_coord_j[0] < node_coord[0]:
                    le_.append(j)
                    le__coord.append(node_coord_j)
                elif node_coord_j[0] > node_coord[0]:
                    te_.append(j)
                    te__coord.append(node_coord_j)
                else:
                    pass
                    #import pdb;pdb.set_trace()
    return (beam_, np.array(beam__coord),
            le_,   np.array(le__coord),
            te_,   np.array(te__coord))

def line_points(n: int, p1: np.ndarray, p2:np.ndarray) -> np.ndarray:
    """Array with n+1 points from p1 to p2

    Parameters
    ----------
    n : int
        Number of points
    p1 : np.ndarray
        Point 1
    p2 : np.ndarray
        Point 2

    Returns
    -------
    np.ndarray
        (n+1, 3) array with location of points

    """

    if np.allclose(p1, p2):
        import pdb; pdb.set_trace()
    pn = (p2-p1)/np.linalg.norm(p2-p1)
    step = np.linalg.norm(p2-p1)/n
    px=[]
    for i in range(n):
      px.append(p1+step*i*pn) 
    px.append(p2)
    px=np.array(px)
    return px
    
def build_pyvista(le, te, beam, chordwise_points=10) -> tuple[np.ndarray, np.ndarray]:

    points = []
    cells = []
    len_i = len(le)
    len_j = chordwise_points + 1 
    for i in range(len_i - 1):
        line_ = line_points(chordwise_points, le[i], te[i])
        if i==0:
            points = line_
        else:
            points = np.vstack([points, line_])
        for j in range(chordwise_points - 1):
            cells.append([4, i*len_j + j, (i+1)*len_j + j, (i + 1)*len_j + j+1, i*len_j + j +1])
    line_ = line_points(chordwise_points, le[i+1], te[i+1])
    points = np.vstack([points, line_])        
    return points, np.array(cells)

def extract_modes(beam: np.ndarray, le: np.ndarray, te: np.ndarray,
                  modes: list, modal_node_ids: list,
                  factor: float=1.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_modes = len(modes)
    beam_disp = []
    le_disp = []
    te_disp = []
    for mi in range(num_modes):
        beam_disp_mi = []
        le_disp_mi = []
        te_disp_mi = []
        for i, beam_i in enumerate(beam):
            node_i = index_node(beam_i, modal_node_ids)
            beam_disp_mi.append(modes[mi][node_i])
            node_lei = index_node(le[i], modal_node_ids)
            le_disp_mi.append(modes[mi][node_lei])
            node_tei = index_node(te[i], modal_node_ids)
            te_disp_mi.append(modes[mi][node_tei])
        beam_disp.append(beam_disp_mi)
        le_disp.append(le_disp_mi)
        te_disp.append(te_disp_mi)
        
    return (factor * np.array(beam_disp),
            factor * np.array(le_disp),
            factor * np.array(te_disp))

def extract_masses(model):
    """Extract mass information about model

    Parameters
    ----------
    model : BDF

    """
    
    mass, cg, inertia = mass_properties(model)
    masses_ids = model.masses.keys()
    masses_nodes=[]
    for i in masses_ids:
        masses_nodes.append(model.masses[i].nid)

    return mass, cg, inertia, masses_ids, masses_nodes

@dataclass
class Component:

    name: str
    beam_ids: list
    beam_coord: np.ndarray
    leading_edge_ids: list
    leading_edge_coord: np.ndarray
    trailing_edge_ids: list
    trailing_edge_coord: np.ndarray
    symmetry_index: int
    beam_disp: np.ndarray = None
    le_disp: np.ndarray = None
    te_disp: np.ndarray = None
    modes:list = None
    _points: np.ndarray = None
    _cells: np.ndarray = None
    mesh:type = None
    _modalpoints: dict = None
    _modalcells: dict = None

    def set_modalindex(self, modes_index):
        self.modes_index= modes_index
        self._modalpoints = {i:None for i in modes_index}
        self._modalcells = {i:None for i in modes_index}
        
    def set_points(self, points, cells):
        self._points = points
        self._cells = cells
        
    def add_points(self, points, cells):
        self._points = np.vstack([self._points, points])
        self._cells =  np.vstack([self._cells, cells])

    def set_modalpoints(self, mode_i, points, cells):
        self._modalpoints[mode_i] = points
        self._modalcells[mode_i] = cells
        
    def add_modalpoints(self, mode_i, points, cells):
        self._modalpoints[mode_i] = np.vstack([self._modalpoints[mode_i],
                                               points])
        self._modalcells[mode_i] =  np.vstack([self._modalcells[mode_i],
                                               cells])

    def set_mesh(self, mesh):
        self.mesh = mesh
        
    def set_modaldisp(self, b_disp, le_disp, te_disp):
        self.beam_disp = b_disp
        self.le_disp = le_disp
        self.te_disp = te_disp
        self.num_modes = len(b_disp)
        
    def apply_transformation(self, factors:np.ndarray)->None:

        le_coord = self.beam_coord + ((self.leading_edge_coord - self.beam_coord).T * factors[0]).T
        te_coord = self.beam_coord + ((self.trailing_edge_coord - self.beam_coord).T * factors[1]).T
        self.leading_edge_coord = le_coord
        self.trailing_edge_coord = te_coord
        if self.beam_disp is not None:
            for mi in range(self.num_modes):
                self.le_disp[mi] = self.beam_disp[mi] +  ((self.le_disp[mi]
                                                          - self.beam_disp[mi]).T * factors[0]).T
                self.te_disp[mi] = self.beam_disp[mi] +  ((self.te_disp[mi]
                                                          - self.beam_disp[mi]).T * factors[1]).T
        self.mesh = None

    def apply_shift(self, point):

        self.beam_coord += point 
        self.leading_edge_coord += point
        self.trailing_edge_coord += point
        self.mesh = None

class Model:

    def __init__(self, model, Component_type=Component, components=None, modes=None,
                 modal_node_ids=None, modes_scaling=1.):
        self.model = model
        self.model_components = []
        self.components = components
        self.Component =Component_type
        self.modes = modes
        if components is not None:
            self.build_model(components)
        if modes is not None:
            self.get_modes(modes, modal_node_ids, modes_scaling)
            
    def build_component(self, name, rbe2_ids, tolerance,
                        plane_vector=[0.,0.,1.], symmetry_index=None, **kwargs):

        self.model_components.append(name)
        (beam_component, beam_component_coord, le_component, le_component_coord,
         te_component, te_component_coord) = extract_aero(self.model,
                                                          rbe2_ids,
                                                          tolerance,
                                                          plane_vector)

        setattr(self, name, self.Component(name, beam_component, beam_component_coord,
                                           le_component, le_component_coord, te_component,
                                           te_component_coord, symmetry_index))

    def build_model(self, components):

        self.components = components
        for k, v in components.items():
            self.build_component(k, **v)
            ci = getattr(self, k)
            if 'scale_factors' in v.keys():
                ci.apply_transformation(v['scale_factors'])            
            if 'shift_point' in v.keys():
                ci.apply_shift(v['shift_point'])
            
    def get_modes(self, modes, modal_node_ids, mode_scaling=1.):

        for ci in self.model_components:
            comp_i = getattr(self, ci)
            beam, le, te = extract_modes(comp_i.beam_ids,
                                         comp_i.leading_edge_ids,
                                         comp_i.trailing_edge_ids,
                                         modes, modal_node_ids, factor=mode_scaling)
            comp_i.set_modaldisp(beam, le, te)
            
    def build_gridmesh(self, nchord, save_file=None,
                       save_dir=None, build_symmetric=True):

        for i, ci in enumerate(self.model_components):
            comp_i = getattr(self, ci)
            if isinstance(nchord, int):
                comp_i.nchord = nchord
            elif isinstance(nchord, list):
                comp_i.nchord = nchord[i]
                
            _points, _cells = build_pyvista(comp_i.leading_edge_coord[:comp_i.symmetry_index],
                                            comp_i.trailing_edge_coord[:comp_i.symmetry_index],
                                            comp_i.beam_coord[:comp_i.symmetry_index],
                                            chordwise_points=nchord)
            comp_i.set_points(_points, _cells)
            mesh = pyvista.PolyData(_points, _cells)
            if comp_i.symmetry_index is not None and build_symmetric:
                _points, _cells = build_pyvista(get_symmetry(comp_i.leading_edge_coord,
                                                             comp_i.symmetry_index),
                                                get_symmetry(comp_i.trailing_edge_coord,
                                                             comp_i.symmetry_index),
                                                get_symmetry(comp_i.beam_coord,
                                                             comp_i.symmetry_index),
                                                chordwise_points=nchord)
                mesh2 = pyvista.PolyData(_points, _cells)
                mesh = mesh.merge(mesh2)
                _points, _cells = build_pyvista(comp_i.leading_edge_coord[comp_i.symmetry_index:],
                                                comp_i.trailing_edge_coord[comp_i.symmetry_index:],
                                                comp_i.beam_coord[comp_i.symmetry_index:],
                                                chordwise_points=nchord)            
                comp_i.add_points(_points, _cells)
            comp_i.set_mesh(mesh)
            if save_file is not None:
                if save_dir is not None:
                    self.file_path = pathlib.Path(save_dir)
                else:
                    self.file_path = pathlib.Path().cwd() / "modes_paraview"
                self.file_path.mkdir(parents=True, exist_ok=True)
                mesh.save(self.file_path / f"{save_file}_{comp_i.name}.ply",
                          binary=False)

    def build_modalmesh(self, nchord, modes_index, save_file=None,
                        save_dir=None, build_symmetric=True):

        for i, ci in enumerate(self.model_components):
            comp_i = getattr(self, ci)
            if isinstance(nchord, int):
                comp_i.nchord = nchord
            elif isinstance(nchord, list):
                comp_i.nchord = nchord[i]
                
            comp_i.set_modalindex(modes_index)
            for mi in modes_index:
                _points, _cells = build_pyvista(comp_i.leading_edge_coord[:comp_i.symmetry_index] +
                                                comp_i.le_disp[mi, :comp_i.symmetry_index, :3],
                                                comp_i.trailing_edge_coord[:comp_i.symmetry_index] +
                                                comp_i.te_disp[mi, :comp_i.symmetry_index, :3],
                                                comp_i.beam_coord[:comp_i.symmetry_index] +
                                                comp_i.beam_disp[mi, :comp_i.symmetry_index, :3],
                                                chordwise_points=nchord)

                comp_i.set_modalpoints(mi, _points, _cells)
                mesh = pyvista.PolyData(_points, _cells)
                # _points, _cells = build_pyvista(comp_i.le_disp[mi, :comp_i.symmetry_index, :3],
                #                                 comp_i.te_disp[mi, :comp_i.symmetry_index, :3],
                #                                 comp_i.beam_disp[mi, :comp_i.symmetry_index, :3],
                #                                 chordwise_points=nchord)
                if comp_i.symmetry_index is not None and build_symmetric:
                    _points, _cells = build_pyvista(get_symmetry(comp_i.leading_edge_coord,
                                                                 comp_i.symmetry_index) +
                                                    get_symmetry(comp_i.le_disp[mi],
                                                                 comp_i.symmetry_index),
                                                    get_symmetry(comp_i.trailing_edge_coord,
                                                                 comp_i.symmetry_index) +
                                                    get_symmetry(comp_i.te_disp[mi],
                                                                 comp_i.symmetry_index),
                                                    get_symmetry(comp_i.beam_coord,
                                                                 comp_i.symmetry_index) +
                                                    get_symmetry(comp_i.beam_disp[mi],
                                                                 comp_i.symmetry_index),
                                                    chordwise_points=nchord)
                    comp_i.add_modalpoints(mi, _points, _cells)
                    mesh2 = pyvista.PolyData(_points, _cells)
                    mesh = mesh.merge(mesh2)
                    # _points, _cells = build_pyvista(comp_i.le_disp[mi, comp_i.symmetry_index:, :3],
                    #                                 comp_i.te_disp[mi, comp_i.symmetry_index:, :3],
                    #                                 comp_i.beam_disp[mi, comp_i.symmetry_index:, :3],
                    #                                 chordwise_points=nchord)

                if save_file is not None:
                    if save_dir is not None:
                        self.file_path = pathlib.Path(save_dir)
                    else:
                        self.file_path = pathlib.Path().cwd() / "modes_paraview"
                    self.file_path.mkdir(parents=True, exist_ok=True)
                    mesh.save(self.file_path / f"{save_file}{mi}_{comp_i.name}.ply",
                              binary=False)
