import uproot
import numpy as np
import awkward as ak
from typing import Tuple
from mc_plotter import MCPlotter

class MCReader:
    def __init__(self, dis_file: uproot, branch: dict, name: str, rootfile):
        self.dis_file = dis_file
        self.branch = branch
        self.name = name
        self.rootfile = rootfile
        self.plotter = MCPlotter(rootfile, name)

    def get_mc_info(
            self, 
            verbose: bool = False, 
            plot_verbose: bool = False
        ) -> Tuple[ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray, np.ndarray, ak.Array, ak.Array]:
        """
        Retrieves Monte Carlo (MC) information, including momenta and derived quantities(charge, PDGID).

        Args:
            name (str): Name for plotting output files.
            verbose (bool): Flag for printing debug information.
            plot_verbose (bool): Flag for generating plots.

        Returns:
            Tuple[ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray, np.ndarray, ak.Array, ak.Array]: MC momenta and related properties.
        """
        print('Start getting MC info')

        mc_px = self.dis_file[self.branch['mc_branch'][12]].array(library='ak')
        mc_py = self.dis_file[self.branch['mc_branch'][13]].array(library='ak')
        mc_pz = self.dis_file[self.branch['mc_branch'][14]].array(library='ak')

        mc_p = np.sqrt(mc_px**2 + mc_py**2 + mc_pz**2)
        mc_p_theta = np.where(mc_p != 0, np.arccos(mc_pz / mc_p), 0)
        mc_p_phi = np.arctan2(mc_py, mc_px)
        mc_PDG_ID = self.dis_file[self.branch['mc_branch'][0]].array(library='ak')
        mc_charge = self.dis_file[self.branch['mc_branch'][3]].array(library='ak')
        mc_generator_status = self.dis_file[self.branch['mc_branch'][1]].array(library='ak')
        mc_vertex_x = self.dis_file[self.branch['mc_branch'][6]].array(library='ak')
        mc_vertex_y = self.dis_file[self.branch['mc_branch'][7]].array(library='ak')
        mc_vertex_z = self.dis_file[self.branch['mc_branch'][8]].array(library='ak')

        if verbose:
            print(f'Number of mc events px: {len(mc_px)}')

        if plot_verbose:
            self.plotter.plot_mc_info(
                mc_px, mc_py, mc_pz, mc_p, mc_p_theta, mc_p_phi, mc_PDG_ID, mc_vertex_x, mc_vertex_y, mc_vertex_z
            )
        
        print('End getting MC info')

        return mc_px, mc_py, mc_pz, mc_p, mc_p_theta, mc_p_phi, mc_PDG_ID, mc_charge, mc_generator_status, mc_vertex_x, mc_vertex_y, mc_vertex_z