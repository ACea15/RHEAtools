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
        
    def set_line_type(self, line_index):
        if line_index in self.current_reading_lines:
            self.line_type = 0
        else:
            self.line_type = 1

    def save_line(self, line):
        if self.current_group == "NPOIN":
            pass
        elif self.current_group == "NELEM":
            pass
    def read_file(self):
        with open(self.mesh_file, 'r') as fp:
            for i, line_i in enumerate(fp):
                self.set_line_type(i)
                if self.line_type == 0:
                    self.current_group
                
            
