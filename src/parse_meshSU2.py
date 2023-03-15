import numpy as np
from dataclasses import dataclass

@dataclass
class MeshData:

    NDIME: int = 0
    NELEM: list = None
    NPOIN: list = None
    NMARK: int = 0
    MARKER_TAGS: dict = None
    MARKER_ELEMS: dict = None
    points: list = None
    elements: list = None
    markers: dict = None
    
    def __post_init__(self):
        pass
    
class ReadMesh:

    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self.current_group = None
        self.current_reading_lines = []
        
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
        elif line0.lower() == "NPOIN":
            self.mesh_data.NPOIN = int(line1)
            self.current_group = "NPOIN"
        elif line0.lower() == "nmark":
            self.mesh_data.NMARK = int(line1)
            self.current_group = "NMARK"
        elif line0.lower() == "marker_tag":
            self.mesh_data.MARKER_TAGS.append(line1)
            self.current_marker = line1
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
            self.mesh_data.elements.append(split_line)

    def read_file(self):
        with open(self.mesh_file, 'r') as fp:
            for i, line_i in enumerate(fp):
                self.set_line_type(line_i)
                if self.line_type == 0:
                    self.save_line(line_i.strip())
                elif self.line_type == 1:
                    self.set_group(line_i.strip())
                    
                
            
