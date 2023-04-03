import numpy as np
from dataclasses import dataclass

@dataclass
class MeshData:

    NDIME: int = 0
    NELEM: int = 0
    NPOIN: int = 0
    NMARK: int = 0
    MARKER_TAGS: list[str] = None
    MARKER_ELEMS: dict = None
    points: list = None
    elements: list = None
    markers: dict = None
    
    def __post_init__(self):

        if self.MARKER_TAGS is None:
            self.MARKER_TAGS = list()
        if self.MARKER_ELEMS is None:
            self.MARKER_ELEMS = dict()
        if self.points is None:
            self.points = list()
        if self.elements is None:
            self.elements = list()
        if self.markers is None:
            self.markers = dict()
    
class ReadMesh:

    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self.mesh_data = MeshData()
        self._markers_list = dict()
        self._markers_grid = dict()
        self.current_group = None
        self.current_reading_lines = []
        self.read_file()
        self.calculate_markers_grid()
        
    def set_line_type(self, line):
        if "=" in line:
            self.line_type = 1
        else:
            self.line_type = 0

    def set_group(self, line):

        line = line.replace(" ", "")
        line0, line1 = line.split("=")
        if line0.lower() == "ndime":
            self.mesh_data.NDIME = int(line1)
        elif line0.lower() == "nelem":
            self.mesh_data.NELEM = int(line1)
            self.current_group = "NELEM"
        elif line0.lower() == "npoin":
            self.mesh_data.NPOIN = int(line1)
            self.current_group = "NPOIN"
        elif line0.lower() == "nmark":
            self.mesh_data.NMARK = int(line1)
            self.current_group = "NMARK"
        elif line0.lower() == "marker_tag":
            self.mesh_data.MARKER_TAGS.append(line1)
            self.current_marker = line1
            self.mesh_data.markers[self.current_marker] = list()
            self._markers_list[self.current_marker] = list()
        elif line0.lower() == "marker_elems":
            self.mesh_data.MARKER_ELEMS[self.current_marker] = int(line1)
            
    def save_line(self, line):
        if self.current_group == "NPOIN":
            split_line = [float(li) for li in line.split(" ") if li != '']
            self.mesh_data.points.append(split_line)
        elif self.current_group == "NELEM":
            split_line = [int(li) for li in line.split(" ") if li != '']
            self.mesh_data.elements.append(split_line)            
        elif self.current_group == "NMARK":
            split_line = [int(li) for li in line.split(" ") if li != '']
            self.mesh_data.markers[self.current_marker].append(split_line)
            _ = [self._markers_list[self.current_marker].append(si) for si in split_line[1:]
                 if si not in self._markers_list[self.current_marker]]

    def read_file(self):
        with open(self.mesh_file, 'r') as fp:
            for i, line_i in enumerate(fp):
                self.set_line_type(line_i)
                if self.line_type == 0:
                    self.save_line(line_i.strip())
                elif self.line_type == 1:
                    self.set_group(line_i.strip())

    def calculate_markers_grid(self):

        for mi in self.mesh_data.MARKER_TAGS: # cycle through markers
            self._markers_grid[mi] = list()
            for si in self._markers_list[mi]: #cycle through nodes 
                node_coord = self.mesh_data.points[si][:3]
                self._markers_grid[mi].append(node_coord)
                
    def get_marker_grid(self, marker):

        return self._markers_grid[marker]
    
    def write_marker_grid(self, file_name: str, markers: list[str]):

        write_status = 'w'
        for i, mi in enumerate(self.mesh_data.MARKER_TAGS):
            for j, mj in enumerate(markers):
                if mi == mj:
                    with open(file_name, write_status) as fp:
                        fp.write(f"#{j}#{i}\n")
                        for k, node_k in enumerate(self._markers_list[mi]):
                            x, y, z = self._markers_grid[mj][k]
                            fp.write(f"{node_k} {x} {y} {z}\n")
                    write_status = 'a'
                        
        
m2 = ReadMesh("../data/in/1901_inv.su2")
m2.write_marker_grid("../data/in/sbw_1901FULL.txt", m2.mesh_data.MARKER_TAGS)
#m1.read_file()
#m1.calculate_markers_grid()
