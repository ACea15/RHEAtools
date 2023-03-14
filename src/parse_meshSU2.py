import numpy as np
from dataclasses import dataclass

@dataclass
class MeshData:

    NDIME: int = 0
    NELEM: list = None
    NPOIN: list = None
    NMARK: int = 0
    MARKER_TAGS: dict = None

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
        
        
    def save_line(self, line):
        if self.current_group == "NPOIN":
            split_line = [float(li) for li in line.split(" ") if li != '']
        elif self.current_group == "NELEM":
            pass
        
    def read_file(self):
        with open(self.mesh_file, 'r') as fp:
            for i, line_i in enumerate(fp):
                self.set_line_type(line_i)
                if self.line_type == 0:
                    self.save_line(line_i.strip())
                elif self.line_type == 1:
                    self.set_group(line_i.strip)
                    
                
            
