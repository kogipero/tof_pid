import numpy as np
import uproot
import awkward as ak

from tof_plotter import TOFPlotter
from typing import Tuple

class TOFReader:
    def __init__(self, dis_file: uproot.TTree, branch: dict, name: str, rootfile):
        self.name = name
        self.rootfile = rootfile
        self.branch = branch
        self.dis_file = dis_file
        self.plotter = TOFPlotter(rootfile, name)

    def get_tof_info(self, name: str, SELECTED_EVENTS: int, rootfile: uproot.TTree, verbose: bool = False, plot_verbose: bool = False) -> Tuple[dict, dict]:
        """
        Retrieves TOF (Time-of-Flight) hit information for barrel and endcap detectors.

        Args:
            name (str): Name for plotting output files.
            SELECTED_EVENTS (int): Number of events to process.
            verbose (bool): Flag for printing debug information.
            plot_verbose (bool): Flag for generating plots.

        Returns:
            Tuple[dict, dict]: Dictionaries containing phi, theta, and time for BToF and EToF hits.
        """

        btof_pos_x = self.dis_file[self.branch['btof_branch'][1]].array(library='ak')[:SELECTED_EVENTS]
        btof_pos_y = self.dis_file[self.branch['btof_branch'][2]].array(library='ak')[:SELECTED_EVENTS]
        btof_pos_z = self.dis_file[self.branch['btof_branch'][3]].array(library='ak')[:SELECTED_EVENTS]
        btof_r = np.sqrt(btof_pos_x**2 + btof_pos_y**2)
        btof_time = self.dis_file[self.branch['btof_branch'][7]].array(library='ak')[:SELECTED_EVENTS]

        ectof_pos_x = self.dis_file[self.branch['etof_branch'][1]].array(library='ak')[:SELECTED_EVENTS]
        ectof_pos_y = self.dis_file[self.branch['etof_branch'][2]].array(library='ak')[:SELECTED_EVENTS]
        ectof_pos_z = self.dis_file[self.branch['etof_branch'][3]].array(library='ak')[:SELECTED_EVENTS]
        ectof_time = self.dis_file[self.branch['etof_branch'][7]].array(library='ak')[:SELECTED_EVENTS]

        btof_phi = np.arctan2(btof_pos_y, btof_pos_x)
        btof_theta = np.arctan2(
            np.sqrt(btof_pos_x**2 + btof_pos_y**2),
            btof_pos_z
        )

        ectof_phi = np.arctan2(ectof_pos_y, ectof_pos_x)
        ectof_theta = np.arctan2(
            np.sqrt(ectof_pos_x**2 + ectof_pos_y**2),
            ectof_pos_z
        )

        btof_phi_theta = {
            'phi': btof_phi,
            'theta': btof_theta,
            'time': btof_time
        }

        ectof_phi_theta = {
            'phi': ectof_phi,
            'theta': ectof_theta,
            'time': ectof_time
        }

        if verbose:
            print(f"Number of BToF hits: {ak.num(btof_phi_theta['phi'])}")
            print(f"Number of EToF hits: {ak.num(ectof_phi_theta['phi'])}")

        if plot_verbose:
            self.plotter.plot_tof_info(
              btof_pos_x, btof_pos_y, btof_pos_z, btof_phi_theta, btof_r
            )

        return btof_phi_theta, ectof_phi_theta

