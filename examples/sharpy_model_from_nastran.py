import numpy as np
import src.nastran2sharpy as n2s
from pyNastran.bdf.bdf import BDF

def get_sharpy_settings(components_dict, components_options,
                        return_total_mass=False, use_femwet_nodes=True):
    
    INCLUDE_LUMPEDMASS = True
    DO_DIAGONAL = False
    component_names = components_dict.keys()
    component_settings = dict()
    model_settings = dict()
    model_settings['assembly'] = {'include_aero': 1,
                                  'default_settings': 1
                                  }
    total_lumped_mass = 0.
    for ci in component_names:

        comp = components_dict[ci]  # type: WingComponent
        beam = comp.beam_model # type: BeamProperties

        ###########################
        # define beam coordinates #
        ###########################
        component_settings[ci] = {'geometry': {}, 'fem': {}, 'aero': {}}
        # Sharpy uses 3-noded elements, num_node per component must be at least 3 or 2n+1
        if ci != "strut" and use_femwet_nodes:
            bn = [np.linalg.norm(beam.nodes[i+1]-beam.nodes[i]) for
                 i in range(len(beam.nodes)-1)] # calculated ds between nodes
            bn2 = [[bi/2, bi/2] for bi in bn] # add a middle node
            component_settings[ci]['geometry']['ds'] = sum(bn2, [])
        else:
            component_settings[ci]['geometry']['num_node'] = beam.nb_nodes + beam.nb_elements
            component_settings[ci]['geometry']['length'] = np.linalg.norm(beam.nodes[-1] -
                                                                          beam.nodes[0])

        if ci == "strut":
            component_settings[ci]['geometry']['direction'] = beam.nodes[0] - beam.nodes[-1]
        else:
            component_settings[ci]['geometry']['direction'] = beam.nodes[-1] - beam.nodes[0]

        ##############
        # define fem #
        ##############
        
        if ci == "strut":            
            component_settings[ci]['fem']['stiffness_db'] = 1.*np.flip(beam.elastic_moduli, axis=0)
            component_settings[ci]['fem']['mass_db'] = np.flip(beam.mass_properties, axis=0)
        else:
            component_settings[ci]['fem'][
                'stiffness_db'] = 1.*beam.elastic_moduli
            component_settings[ci]['fem'][
                'mass_db'] = beam.mass_properties
        if DO_DIAGONAL:
            component_settings[ci]['fem']['stiffness_db'] = \
                np.array([np.diag(np.diag(Emat)) for Emat in
                          component_settings[ci]['fem']['stiffness_db']])
            # component_settings[ci]['fem']['mass_db'] = \
            #     np.array([np.diag(np.diag(Emat)) for Emat in
            #               component_settings[ci]['fem']['mass_db']])
        # the following assumes there is one stiffness matrix per element and they are \
            # arranged in increasing order
        component_settings[ci]['fem']['elem_stiffness'] = range(len(
            component_settings[ci]['fem']['stiffness_db']))

        component_settings[ci]['fem']['elem_mass'] = range(len(
            component_settings[ci]['fem']['mass_db']))

        try:
            component_settings[ci]['fem']['frame_of_reference_delta'] = components_options[ci]["for_delta"]
        except AttributeError: # generic delta_frame for right-wing components
            component_settings[ci]['fem']['frame_of_reference_delta'] =  \
            np.array([-1, 0, 0])
            
        if components_dict[ci].lumped_masses_values.size > 0:
            total_lumped_mass += sum(comp.lumped_masses_values)
            if INCLUDE_LUMPEDMASS:
                component_settings[ci]['fem']['lumped_mass'] = comp.lumped_masses_values
                component_settings[ci]['fem']['lumped_mass_inertia'] = comp.lumped_inertias
                component_settings[ci]['fem']['lumped_mass_nodes'] = \
                   comp.lumped_masses_attachment
                component_settings[ci]['fem']['lumped_mass_position'] = \
                   comp.lumped_masses_coordinates
            
        ###############
        # define aero #
        ###############
        # TODO: Areo object (not done yet)
        if components_dict[ci].panel_corners.size > 0:
            component_settings[ci]['workflow'] = ['create_structure','create_aero']
            component_settings[ci]['aero']['surface_m'] = \
                components_options[ci]["surface_m"]      # Meaning ? nb panels chordwise
            # except AttributeError:
            #     component_settings[ci]['aero']['surface_m'] = \
            #         kwargs['surface_m'][ci]


            if ci == "strut":
                
                component_settings[ci]['aero']['beam_origin'] = beam.nodes[-1]

                component_settings[ci]['aero']['point_platform'] = dict()
                component_settings[ci]['aero']['point_platform']['leading_edge1'] = comp.leading_points[-1]
                component_settings[ci]['aero']['point_platform']['leading_edge2'] = comp.leading_points[0]
                component_settings[ci]['aero']['point_platform']['trailing_edge1'] = comp.trailing_points[-1]
                component_settings[ci]['aero']['point_platform']['trailing_edge2'] = comp.trailing_points[0]
            else:
                component_settings[ci]['aero']['beam_origin'] = beam.nodes[0]

                component_settings[ci]['aero']['point_platform'] = dict()
                component_settings[ci]['aero']['point_platform']['leading_edge1'] = \
                    comp.leading_points[0]
                component_settings[ci]['aero']['point_platform']['leading_edge2'] = \
                    comp.leading_points[-1]
                component_settings[ci]['aero']['point_platform']['trailing_edge1'] = \
                    comp.trailing_points[0]
                component_settings[ci]['aero']['point_platform']['trailing_edge2'] = \
                    comp.trailing_points[-1]

            if hasattr(comp, 'twists'):
                component_settings[ci]['aero']['twist'] = -comp.twists
            if hasattr(components_dict[ci], 'polars'):
                component_settings[ci]['aero']['airfoil_distribution'] = \
                    components_dict[ci].airfoil_distribution
                
                component_settings[ci]['aero']['polars'] = components_dict[ci].polars
            if hasattr(components_dict[ci], 'airfoils'):
                component_settings[ci]['aero']['airfoil_distribution'] = \
                    components_dict[ci].airfoil_distribution
                component_settings[ci]['aero']['airfoils'] = components_dict[ci].airfoils
        else:
            component_settings[ci]['workflow'] = ['create_structure','create_aero0']

        ############
        # assembly #
        ############
        
        if comp.upstream_component:
            upstream_component = comp.upstream_component.name
            node_in_upstream = comp.upstream_attachment_node
        elif comp.downstream_component:
            upstream_component = comp.downstream_component.name
            node_in_upstream = comp.downstream_attachment_node
        else:
            upstream_component = ''
            node_in_upstream = 0       
        assemble_dict = {'upstream_component': upstream_component,  # !! downstream for strut
                         'node_in_upstream': node_in_upstream}

        model_settings['assembly'].update({ci: assemble_dict})
    if return_total_mass:
        return component_settings, model_settings, total_lumped_mass
    else:
        return component_settings, model_settings

components = dict(wing=dict(cbars=list(range(2000, 2035))),
                  strut=dict(cbars=list(range(5000, 5023))))

model = BDF()
#model.read_bdf("./models/nastran/")
model.read_bdf("../data/in/SOL103/polimi-103cam.bdf")
model = BDF()
#model.read_bdf("./models/nastran/")
model.read_bdf("../data/in/SOL145/polimi-145cam_078M.bdf")

wing = n2s.BeamComponent(model, components['wing']['cbars'])
strut = n2s.BeamComponent(model, components['strut']['cbars'])
