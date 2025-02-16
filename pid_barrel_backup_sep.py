import numpy as np
import os
import uproot
import awkward as ak
import vector
import mplhep as hep
import matplotlib.pyplot as plt
import sys
import yaml
import helper_functions as myfunc
from typing import List, Tuple, Dict
import ROOT as r
from scipy.optimize import curve_fit
from scipy.stats import norm

vector.register_awkward()
hep.style.use(hep.style.ROOT)

# Utility functions
def load_yaml_config(file_path: str) -> dict:
    """
    Loads a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML configuration as a dictionary.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_tree_file(filename: str) -> uproot.TTree:
    """
    Loads a ROOT file and retrieves the 'events' tree.

    Args:
        filename (str): Path to the ROOT file.

    Returns:
        uproot.TTree: The 'events' tree from the ROOT file.
    """
    if not os.path.exists(filename):
        print(f'File {filename} does not exist')
        sys.exit()
    print(f'File {filename} opened')
    file = uproot.open(filename)

    return file['events']

def make_directory(directory_name: str):
    """
    Creates a directory if it does not exist.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

# Define the angular distance function manually
def angular_distance(phi1, theta1, phi2, theta2):   
    delta_phi = phi1 - phi2

    return np.arccos(
        np.sin(theta1) * np.sin(theta2) +
        np.cos(theta1) * np.cos(theta2) * np.cos(delta_phi)
    )

def gaussian(x, A, mu, sigma):
    return A * np.exp( - (x - mu)**2 / (2 * sigma**2) )

class Track:
    def __init__(self, tree: uproot.TTree, config: dict, branch: dict, name: str, dis_file: uproot.TTree):
        self.tree = tree
        self.config = config
        self.branch = branch
        self.name = name
        self.dis_file = dis_file

    def get_track_segments_pos(
            self, 
            name: str,
            rootfile: uproot.TTree, 
            verbose: bool = False, 
            plot_verbose: bool = False
        ) -> Tuple[ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray]:
        """
        Retrieves track segment positions.

        Args:
            name (str): Name for plotting output files.
            verbose (bool): Flag for printing debug information.
            plot_verbose (bool): Flag for generating plots.

        Returns:
            ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray: Track segment positions and derived quantities.
        """
        print('Start getting track segments')
        track_segments_pos_x = self.dis_file[self.branch['track_branch'][2]].array(library='ak')
        track_segments_pos_y = self.dis_file[self.branch['track_branch'][3]].array(library='ak')
        track_segments_pos_z = self.dis_file[self.branch['track_branch'][4]].array(library='ak')
        track_segments_pos_d = np.sqrt(track_segments_pos_x**2 + track_segments_pos_y**2 + track_segments_pos_z**2)
        track_segments_pos_r = np.sqrt(track_segments_pos_x**2 + track_segments_pos_y**2)

        if verbose:
            print(f'Number of track segments x: {len(track_segments_pos_x)}')
            print(f'Number of track segments y: {len(track_segments_pos_y)}')
            print(f'Number of track segments z: {len(track_segments_pos_z)}') 
            print(f'Number of track segments d: {len(track_segments_pos_d)}')
            print(f'Number of track segments r: {len(track_segments_pos_r)}')

        if plot_verbose:
            myfunc.make_histogram_root(ak.flatten(track_segments_pos_x),
                            nbins=100,
                            hist_range=[-1000, 1000],
                            title='Track segments x',
                            xlabel='x [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_pos_x',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_pos_y),
                            100,
                            hist_range=[-1000, 1000],
                            title='Track segments y',
                            xlabel='y [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_pos_y',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_pos_z),
                            100,
                            hist_range=[-1000, 1000],
                            title='Track segments z',
                            xlabel='z [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_pos_z',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_pos_d),
                            300,
                            hist_range=[0, 3000],
                            title='Track segments d',
                            xlabel='d [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_pos_d',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_pos_r),
                            300,
                            hist_range=[0, 3000],
                            title='Track segments r',
                            xlabel='r [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_pos_r',
                            rootfile=rootfile
                            )
            
            flatten_track_segments_pos_x_part = ak.flatten(track_segments_pos_x[:5])
            flatten_track_segments_pos_y_part = ak.flatten(track_segments_pos_y[:5])
            np_track_pos_x = np.array(flatten_track_segments_pos_x_part, dtype=np.float64)
            np_track_pos_y = np.array(flatten_track_segments_pos_y_part, dtype=np.float64)

            myfunc.make_TGraph(
                        np_track_pos_x,
                        np_track_pos_y,
                        title='Track segments xy 5 events',
                        xlabel='x [mm]',
                        ylabel='y [mm]',
                        outputname=f'{name}/track_segments_xy',
                        rangex=1000,
                        rangey=1000,
                        rootfile=rootfile
                        )
                        
        print('End getting track segments')

        return track_segments_pos_x, track_segments_pos_y, track_segments_pos_z, track_segments_pos_d, track_segments_pos_r

    def get_track_segments_momentum(
            self, 
            name: str, 
            rootfile: uproot.TTree,
            verbose: bool = False, 
            plot_verbose: bool = False
        ) -> Tuple[ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray, np.ndarray, ak.Array]:
        """
        Retrieves track segment momentum and computes derived quantities(pathlength).

        Args:
            name (str): Name for plotting output files.
            verbose (bool): Flag for printing debug information.
            plot_verbose (bool): Flag for generating plots.

        Returns:
            Tuple[ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray, np.ndarray, ak.Array]: Track segment momenta and derived quantities.
        """
        print('Start getting track segments momentum')

        track_segments_px = self.dis_file[self.branch['track_branch'][11]].array(library='ak')
        track_segments_py = self.dis_file[self.branch['track_branch'][12]].array(library='ak')
        track_segments_pz = self.dis_file[self.branch['track_branch'][13]].array(library='ak')
        track_segments_p = np.sqrt(track_segments_px**2 + track_segments_py**2 + track_segments_pz**2)
        track_segments_pt = np.sqrt(track_segments_px**2 + track_segments_py**2)

        track_segments_p_theta = np.where(track_segments_p != 0, np.arccos(track_segments_pz / track_segments_p), 0)
        track_segments_p_phi = np.arctan2(track_segments_py, track_segments_px)

        track_segment_pathlength = self.dis_file[self.branch['track_branch'][27]].array(library='ak')

        if verbose:
            print(f'Number of track events px: {len(track_segments_px)}')
            print(f'Number of track events py: {len(track_segments_py)}')
            print(f'Number of track events pz: {len(track_segments_pz)}')
            print(f'Number of track events p: {len(track_segments_p)}')
            print(f'Number of track events p_theta: {len(track_segments_p_theta)}')
            print(f'Number of track events p_phi: {len(track_segments_p_phi)}')
            print(f'Number of track events pathlength: {len(track_segment_pathlength)}')

        if plot_verbose:
            myfunc.make_histogram_root(ak.flatten(track_segments_px),
                            30,
                            hist_range=[0, 30],
                            title='Track segments px',
                            xlabel='px [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_px',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_py),
                            30,
                            hist_range=[0, 30],
                            title='Track segments py',
                            xlabel='py [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_py',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_pz),
                            30,
                            hist_range=[0, 30],
                            title='Track segments pz',
                            xlabel='pz [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_pz',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_p),
                            100,
                            hist_range=[0, 20],
                            title='Track segments p',
                            xlabel='p [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_p',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_pt),
                            100,
                            hist_range=[0, 20],
                            title='Track segments pt',
                            xlabel='pt [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_pt',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_p_theta),
                            100,
                            hist_range=[0, 3.2],
                            title='Track segments theta',
                            xlabel='theta [rad]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_p_theta',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segments_p_phi),
                            100,
                            hist_range=[-3.2, 3.2],
                            title='Track segments phi',
                            xlabel='phi [rad]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segments_p_phi',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(track_segment_pathlength),
                            300,
                            hist_range=[0, 3000],
                            title='Track segment pathlength',
                            xlabel='Pathlength [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/track_segment_pathlength',
                            rootfile=rootfile
                            )
            
        print('End getting track segments momentum')

        return track_segments_px, track_segments_py, track_segments_pz, track_segments_p, track_segments_pt, track_segments_p_theta, track_segments_p_phi, track_segment_pathlength

    def split_track_segments(
            self, 
            x_positions: ak.Array, 
            y_positions: ak.Array, 
            z_positions: ak.Array,
            px_momenta: ak.Array, 
            py_momenta: ak.Array, 
            pz_momenta: ak.Array, 
            track_segment_pathlength: ak.Array, 
            margin_theta: float, 
            margin_phi: float, 
            rootfile: uproot.TTree,
            verbose: bool = False, 
            plot_verbose: bool = False, 
            SELECTED_EVENTS: int = 50000
        ) -> List[List[List[Tuple]]]:
        """
        Splits track segments into individual tracks based on angular separation.

        Args:
            x_positions (ak.Array): X positions of track segments.
            y_positions (ak.Array): Y positions of track segments.
            z_positions (ak.Array): Z positions of track segments.
            px_momentum (ak.Array): Px components of track momentum.
            py_momentum (ak.Array): Py components of track momentum.
            pz_momentum (ak.Array): Pz components of track momentum.
            track_segment_pathlength (ak.Array): Path lengths of track segments.
            margin_theta (float): Angular margin for splitting tracks in theta.
            margin_phi (float): Angular margin for splitting tracks in phi.
            verbose (bool): Flag for printing debug information.
            plot_verbose (bool): Flag for generating plots.
            SELECTED_EVENTS (int): Number of events to process.

        Returns:
            List[List[List[Tuple]]]: Nested list of tracks for each event.
        """
        print('Start splitting track segments')

        all_tracks = []
        selected_events = min(SELECTED_EVENTS, len(x_positions))

        for event in range(selected_events):
            if len(x_positions[event]) == 0:
                print(f"Skipping empty event at index {event}")
                continue

            event_tracks = []
            r = np.sqrt(x_positions[event][0]**2 + y_positions[event][0]**2)
            current_track = [(event, 0, x_positions[event][0], y_positions[event][0], z_positions[event][0],
                              px_momenta[event][0], py_momenta[event][0], pz_momenta[event][0], r, track_segment_pathlength[event][0])]

            for i in range(1, len(x_positions[event])):
                px1, py1, pz1 = px_momenta[event][i - 1], py_momenta[event][i - 1], pz_momenta[event][i - 1]
                px2, py2, pz2 = px_momenta[event][i], py_momenta[event][i], pz_momenta[event][i]

                theta1, phi1 = np.arctan2(np.sqrt(px1**2 + py1**2), pz1), np.arctan2(py1, px1)
                theta2, phi2 = np.arctan2(np.sqrt(px2**2 + py2**2), pz2), np.arctan2(py2, px2)

                if abs(theta2 - theta1) < margin_theta and abs(phi2 - phi1) < margin_phi:
                    r = np.sqrt(x_positions[event][i]**2 + y_positions[event][i]**2)
                    z = z_positions[event][i]

                    current_track.append((event, len(event_tracks), x_positions[event][i], y_positions[event][i], z_positions[event][i], px_momenta[event][i], py_momenta[event][i], pz_momenta[event][i], r, track_segment_pathlength[event][i]))

                else:
                    if current_track:
                        event_tracks.append(current_track)

                    r = np.sqrt(x_positions[event][i]**2 + y_positions[event][i]**2)

                    current_track = [(event, len(event_tracks), x_positions[event][i], y_positions[event][i], z_positions[event][i], px_momenta[event][i], py_momenta[event][i], pz_momenta[event][i], r, track_segment_pathlength[event][i])]

            if current_track:
                event_tracks.append(current_track)
            all_tracks.append(event_tracks)

            if event % 1000 == 0:
                print(f'Processed event: {event} / {selected_events}')

        if verbose:
            for event_idx, event_tracks in enumerate(all_tracks[:5]):
                print(f'Event {event_idx+1} has {len(event_tracks)} tracks')
                for track_idx, track in enumerate(event_tracks):
                    print(f"  Track {track_idx+1}:")

        if plot_verbose:
            for event_idx, event_tracks in enumerate(all_tracks[:5]):
                for track_idx, track in enumerate(event_tracks):
                    x_pos_per_track = np.array([segment[2] for segment in track])
                    y_pos_per_track = np.array([segment[3] for segment in track])

                    myfunc.make_TGraph(
                                x_pos_per_track, 
                                y_pos_per_track,
                                title=f'Track segments_{track_idx}',
                                xlabel='x [mm]',
                                ylabel='y [mm]',
                                outputname=f'{self.name}/track_{track_idx}',
                                rangex=1000,
                                rangey=1000,
                                rootfile=rootfile
                                )
      
        print('End splitting track segments')

        return all_tracks

class MC:
    def __init__(self, tree: uproot.TTree, config: dict, branch: dict, name: str, dis_file: uproot.TTree):
        self.tree = tree
        self.config = config
        self.branch = branch
        self.name = name
        self.dis_file = dis_file

    def get_mc_info(
            self, 
            name: str,
            rootfile: uproot.TTree, 
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
            myfunc.make_histogram_root(ak.flatten(mc_px),
                            100,
                            hist_range=[-20, 20],
                            title='MC px',
                            xlabel='px [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_px',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(mc_py),
                            100,
                            hist_range=[-20, 20],
                            title='MC py',
                            xlabel='py [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_py',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(mc_pz),
                            100,
                            hist_range=[-200, 400],
                            title='MC pz',
                            xlabel='pz [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_pz',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(mc_p),
                            50,
                            hist_range=[0, 5],
                            title='MC p',
                            xlabel='p [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_p',
                            rootfile=rootfile
                            )

            myfunc.make_histogram_root(ak.flatten(mc_p_theta),
                            50,
                            hist_range=[0, 3.2],
                            title='MC theta',
                            xlabel='theta [rad]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_theta',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(mc_p_phi),
                            100,
                            hist_range=[-3.2, 3.2],
                            title='MC phi',
                            xlabel='phi [rad]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_phi',
                            rootfile=rootfile
                            )

            myfunc.make_histogram_root(ak.flatten(mc_PDG_ID),
                            500,
                            hist_range=[-250, 250],
                            title='MC PDG ID',
                            xlabel='PDG ID',
                            ylabel='Entries',
                            outputname=f'{name}/mc_PDG_ID',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(mc_vertex_x),
                            100,
                            hist_range=[-200, 200],
                            title='MC vertex x',
                            xlabel='x [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_vertex_x',
                            rootfile=rootfile
                            )

            myfunc.make_histogram_root(ak.flatten(mc_vertex_y),
                            100,
                            hist_range=[-200, 200],
                            title='MC vertex y',
                            xlabel='y [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_vertex_y',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(mc_vertex_z),
                            100,
                            hist_range=[-200, 200],
                            title='MC vertex z',
                            xlabel='z [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_vertex_z',
                            rootfile=rootfile
                            )
        
        print('End getting MC info')

        return mc_px, mc_py, mc_pz, mc_p, mc_p_theta, mc_p_phi, mc_PDG_ID, mc_charge, mc_generator_status, mc_vertex_x, mc_vertex_y, mc_vertex_z



class MatchingMCAndTrack:
    def __init__(self, track: Track, mc: MC):
        self.track = track
        self.mc = mc

    def get_segments_nearest_impact_point(
            self, 
            all_tracks: List[List[List[Tuple]]], 
            rootfile: uproot.TTree,
            verbose: bool = False, 
            plot_verbose: bool = False
        ) -> Tuple[List[List[float]], List[List[int]], List[List[int]]]:
        """
        Identifies the segments closest to the impact point for each track.

        Args:
            all_tracks (List[List[List[Tuple]]]): Nested list of tracks and their segments.

        Returns:
            Tuple[List[List[float]], List[List[int]], List[List[int]]]: Closest distances, indices, and TOF hits for each track.
        """
        print('Start getting nearest impact point')
        self.all_tracks = all_tracks

        r_min_track_index = []
        r_min_tracks = []

        for event_tracks in self.all_tracks:
            r_min = []
            r_min_index = []
            for track in event_tracks:
                if not track:
                    continue

                min_track = min([segment[8] if len(segment) > 8 and segment[8] is not None else float('inf') for segment in track])
                min_index = [segment[8] if len(segment) > 8 and segment[8] is not None else float('inf') for segment in track].index(min_track)

                if min_track > 50: 
                    print(f"Skipping track with large minimum distance: {min_track}")
                    continue

                r_min.append(min_track)
                r_min_index.append(min_index)

            r_min_tracks.append(r_min)
            r_min_track_index.append(r_min_index)

        if verbose:
            print(f'minimum track distances: {r_min_tracks}')
            print(f'minimum track indices: {r_min_track_index}')
            print(f'Number of events: {len(r_min_tracks)}')

        if plot_verbose:
            myfunc.make_histogram_root(ak.flatten(r_min_tracks),
                            100,
                            hist_range=[0, 500],
                            title='Minimum track distances',
                            xlabel='Distance [mm]',
                            ylabel='Entries',
                            outputname=f'{self.track.name}/min_track_distances',
                            rootfile=rootfile
                            )
        print('End getting nearest impact point')
            
        return r_min_tracks, r_min_track_index

    def match_track_to_mc(
            self, 
            name: str,
            track_momentum: ak.Array,
            track_momentum_x: ak.Array,
            track_momentum_y: ak.Array,
            track_momentum_z: ak.Array,
            track_momentum_transverse: ak.Array,
            track_momentum_theta: ak.Array,
            track_momentum_phi: ak.Array,
            track_pos_x: ak.Array,
            track_pos_y: ak.Array,
            track_pos_z: ak.Array,
            track_pathlength: ak.Array,
            mc_momentum_theta: ak.Array,
            mc_momentum_phi: ak.Array,
            mc_momentum: ak.Array,
            mc_pdg_ID: ak.Array,
            mc_generator_status: ak.Array,
            mc_charge: ak.Array,
            mc_vertex_x: ak.Array,
            mc_vertex_y: ak.Array,
            mc_vertex_z: ak.Array,
            r_min_track_index: List[List[int]],
            threshold: float,
            rootfile: uproot.TTree,
            output_txt: str = 'matching_result.txt',
            vertex_z_min: float = -5,
            vertex_z_max: float = 5,
            verbose: bool = False,
            plot_verbose: bool = False
        ) -> Tuple[Dict[str, List], List[float], Dict[str, List], Dict[str, List]]:
        """
        Matches tracks to MC particles based on angular distance.

        Args:
            name (str): Name for plotting output files.
            track_momentum_theta (ak.Array): Theta angles of track momenta.
            track_momentum_phi (ak.Array): Phi angles of track momenta.
            mc_momentum_theta (ak.Array): Theta angles of MC momenta.
            mc_momentum_phi (ak.Array): Phi angles of MC momenta.
            mc_pdg_ID (ak.Array): PDG IDs of MC particles.
            mc_momentum (ak.Array): Momenta of MC particles.
            track_pathlength (ak.Array): Path lengths of tracks.
            r_min_track_index (List[List[int]]): Closest track segment indices.
            r_min_track_on_tof (List[List[int]]): TOF hits for the closest segments.
            verbose (bool): Flag for printing debug information.
            plot_verbose (bool): Flag for generating plots.

        Returns:
            Tuple[Dict[str, List], List[float]]: Matched pairs and their angular distances.
        """
        print('Start matching track to MC')
        min_delta_angles_all_tracks = []
        delta_angles_all = []
        matched_pairs = {
            "event_idx": [],
            "track_idx": [],
            "track_p": [],
            "track_pt": [],
            "track_p_theta": [],
            "track_p_phi": [],
            "track_pos_theta": [],
            "track_pos_phi": [],
            "track_pos_x": [],
            "track_pos_y": [],
            "track_pos_z": [],
            "mc_theta": [],
            "mc_phi": [],
            "min_delta_angle": [],
            "mc_pdg": [],
            "mc_momentum": [],
            "mc_vertex_x": [],
            "mc_vertex_y": [],
            "mc_vertex_z": [],
            "track_pathlength": [],
        }

        all_matched_pairs = {
            "event_idx": [],
            "track_idx": [],
            "track_pos_theta": [],
            "track_pos_phi": [],
        }

        matched_pairs_on_btof = {
            "event_idx": [],
            "track_idx": [],
            "track_momentum_on_btof": [],
            "track_momentum_transverse_on_btof": [],
            "track_pos_theta_on_btof": [],
            "track_pos_phi_on_btof": [],
            "track_pos_x_on_btof": [],
            "track_pos_y_on_btof": [],
            "track_pos_z_on_btof": [],
            "mc_pdg": [],
            "mc_momentum": [],
            "mc_vertex_x": [],
            "mc_vertex_y": [],
            "mc_vertex_z": [],
            "mc_vertex_d": [],
            "track_pathlength": [],
        }

        matched_pairs_on_etof = {
            "event_idx": [],
            "track_idx": [],
            "track_momentum_on_etof": [],
            "track_momentum_transverse_on_etof": [],
            "track_pos_theta_on_etof": [],
            "track_pos_phi_on_etof": [],
            "track_pos_x_on_etof": [],
            "track_pos_y_on_etof": [],
            "track_pos_z_on_etof": [],
            "mc_pdg": [],
            "mc_momentum": [],
            "mc_vertex_x": [],
            "mc_vertex_y": [],
            "mc_vertex_z": [],
            "track_pathlength": [],
        }

        for event_idx, min_index_list in enumerate(r_min_track_index):
            if len(track_momentum_theta[event_idx]) == 0 or len(mc_momentum_theta[event_idx]) == 0:
                print(f"Skipping empty event at index {event_idx}")
                continue

            mc_momentum_event = np.array(ak.to_list(mc_momentum[event_idx]))  
            mc_theta_event = np.array(ak.to_list(mc_momentum_theta[event_idx]))
            mc_phi_event = np.array(ak.to_list(mc_momentum_phi[event_idx]))
            mc_pdg_event = np.array(ak.to_list(mc_pdg_ID[event_idx]))
            mc_generator_status_event = np.array(ak.to_list(mc_generator_status[event_idx]))
            mc_charge_event = np.array(ak.to_list(mc_charge[event_idx]))
            mc_vertex_x_event = np.array(ak.to_list(mc_vertex_x[event_idx]))
            mc_vertex_y_event = np.array(ak.to_list(mc_vertex_y[event_idx]))
            mc_vertex_z_event = np.array(ak.to_list(mc_vertex_z[event_idx]))
            mc_vertex_d = np.sqrt(mc_vertex_x_event**2 + mc_vertex_y_event**2 + mc_vertex_z_event**2)

            stable_indices = (mc_generator_status_event == 1) & (mc_charge_event != 0) & (mc_vertex_x_event > -100) & (mc_vertex_x_event < 100) & (mc_vertex_y_event > -100) & (mc_vertex_y_event < 100)
            vertex_z_indices = (mc_vertex_z_event > vertex_z_min) & (mc_vertex_z_event < vertex_z_max)
            final_indices = stable_indices & vertex_z_indices
            mc_momentum_event = mc_momentum_event[final_indices]
            mc_theta_event = mc_theta_event[final_indices]
            mc_phi_event = mc_phi_event[final_indices]
            mc_pdg_event = mc_pdg_event[final_indices]
            mc_vertex_x_event = mc_vertex_x_event[final_indices] 
            mc_vertex_y_event = mc_vertex_y_event[final_indices]
            mc_vertex_z_event = mc_vertex_z_event[final_indices]
            mc_vertex_d_event = mc_vertex_d[final_indices]
            track_pos_x_event = np.array(ak.to_list(track_pos_x[event_idx]))
            track_pos_y_event = np.array(ak.to_list(track_pos_y[event_idx]))
            track_pos_z_event = np.array(ak.to_list(track_pos_z[event_idx]))

            track_pos_phi = np.arctan2(track_pos_y_event, track_pos_x_event)
            track_pos_theta = np.arctan2(
                np.sqrt(track_pos_x_event**2 + track_pos_y_event**2),
                track_pos_z_event
            )
            track_p_theta_event = np.array(ak.to_list(track_momentum_theta[event_idx]))
            track_p_phi_event = np.array(ak.to_list(track_momentum_phi[event_idx]))
            track_p_event = np.array(ak.to_list(track_momentum[event_idx]))
            track_pt_event = np.array(ak.to_list(track_momentum_transverse[event_idx]))

            if verbose:
                print(f"Event {event_idx}: Number of tracks = {len(min_index_list)}")
                print(f"Number of MC particles = {len(mc_theta_event)}")
                print(f"Number of tracks = {len(min_index_list)}")

            for track_idx, min_index in enumerate(min_index_list):

                if min_index >= len(track_p_theta_event):
                    print(f"Skipping invalid index: min_index={min_index}")
                    continue

                track_p_theta = track_p_theta_event[min_index]
                track_p_phi = track_p_phi_event[min_index]
                track_p = track_p_event[min_index]
                track_pt = track_pt_event[min_index]
                track_pos_r_event = np.sqrt(track_pos_x_event**2 + track_pos_y_event**2)

                delta_angles = angular_distance(
                    phi1=track_p_phi,
                    theta1=track_p_theta,
                    phi2=mc_phi_event,
                    theta2=mc_theta_event
                )
                if delta_angles.size == 0:
                    print(f"Warning: delta_angles is empty for event {event_idx}, track {track_idx}. Skipping this track.")
                    continue

                imin_mc = np.argmin(delta_angles)
                min_delta_angle = delta_angles[imin_mc]
                delta_angles_all.append(delta_angles)
                all_matched_pairs["event_idx"].append(event_idx)
                all_matched_pairs["track_idx"].append(track_idx)
                all_matched_pairs["track_pos_theta"].append(track_pos_theta[min_index])
                all_matched_pairs["track_pos_phi"].append(track_pos_phi[min_index])

                if min_delta_angle > threshold: ## kokokaeta 2/8
                    print(f"Skipping track {track_idx} in event {event_idx} due to large delta angle: {min_delta_angle}")
                    continue

                if verbose:
                    print(f"Track: {track_idx}")
                    print(f"Track momentum: {track_p}")
                    print(f"Min index (closest to origin): {min_index}")
                    print(f"Min delta angle: {min_delta_angle}")
                    print(f"MC index: {imin_mc}")
                    print(f"MC theta: {mc_theta_event[imin_mc]}")
                    print(f"MC phi: {mc_phi_event[imin_mc]}")
                    print(f"MC PDG ID: {mc_pdg_event[imin_mc]}")
                    print(f"MC momentum: {mc_momentum_event[imin_mc]}")


                if track_pos_z_event[min_index] >= 1840 and track_pos_z_event[min_index] <= 1880 and track_pos_r_event[min_index] >= 0 and track_pos_r_event[min_index] <= 600:
                    if verbose:
                        print("ETOF HIT")
                    matched_pairs_on_etof["track_pos_theta_on_etof"].append(track_pos_theta[min_index])
                    matched_pairs_on_etof["track_pos_phi_on_etof"].append(track_pos_phi[min_index])
                    matched_pairs_on_etof["track_pos_x_on_etof"].append(track_pos_x_event[min_index])
                    matched_pairs_on_etof["track_pos_y_on_etof"].append(track_pos_y_event[min_index])
                    matched_pairs_on_etof["track_pos_z_on_etof"].append(track_pos_z_event[min_index])
                    matched_pairs_on_etof["track_momentum_on_etof"].append(track_p)
                    matched_pairs_on_etof["track_momentum_transverse_on_etof"].append(track_pt)
                    matched_pairs_on_etof["event_idx"].append(event_idx)
                    matched_pairs_on_etof["track_idx"].append(track_idx)
                    matched_pairs_on_etof["mc_pdg"].append(mc_pdg_event[imin_mc])
                    matched_pairs_on_etof["mc_momentum"].append(mc_momentum_event[imin_mc])
                    matched_pairs_on_etof["mc_vertex_x"].append(mc_vertex_x_event[imin_mc])
                    matched_pairs_on_etof["mc_vertex_y"].append(mc_vertex_y_event[imin_mc])
                    matched_pairs_on_etof["mc_vertex_z"].append(mc_vertex_z_event[imin_mc])
                    matched_pairs_on_etof["track_pathlength"].append(track_pathlength[event_idx][min_index])
                    
                if track_pos_z_event[min_index] >= -1500 and track_pos_z_event[min_index] <= 1840 and track_pos_r_event[min_index] >= 625 and track_pos_r_event[min_index] <= 642:
                    if verbose:
                        print("BTOF HIT")
                    matched_pairs_on_btof["track_pos_theta_on_btof"].append(track_pos_theta[min_index])
                    matched_pairs_on_btof["track_pos_phi_on_btof"].append(track_pos_phi[min_index])
                    matched_pairs_on_btof["track_pos_x_on_btof"].append(track_pos_x_event[min_index])
                    matched_pairs_on_btof["track_pos_y_on_btof"].append(track_pos_y_event[min_index])
                    matched_pairs_on_btof["track_pos_z_on_btof"].append(track_pos_z_event[min_index])
                    matched_pairs_on_btof["track_momentum_on_btof"].append(track_p)
                    matched_pairs_on_btof["track_momentum_transverse_on_btof"].append(track_pt)
                    matched_pairs_on_btof["event_idx"].append(event_idx)
                    matched_pairs_on_btof["track_idx"].append(track_idx)
                    matched_pairs_on_btof["mc_pdg"].append(mc_pdg_event[imin_mc])
                    matched_pairs_on_btof["mc_momentum"].append(mc_momentum_event[imin_mc])
                    matched_pairs_on_btof["mc_vertex_x"].append(mc_vertex_x_event[imin_mc])
                    matched_pairs_on_btof["mc_vertex_y"].append(mc_vertex_y_event[imin_mc])
                    matched_pairs_on_btof["mc_vertex_z"].append(mc_vertex_z_event[imin_mc])
                    matched_pairs_on_btof["track_pathlength"].append(track_pathlength[event_idx][min_index])

                matched_pairs["event_idx"].append(event_idx)
                matched_pairs["track_idx"].append(track_idx)
                matched_pairs["track_p"].append(track_p)
                matched_pairs["track_pt"].append(track_pt)
                matched_pairs["track_p_theta"].append(track_p_theta)
                matched_pairs["track_p_phi"].append(track_p_phi)
                matched_pairs["track_pos_theta"].append(track_pos_theta[min_index])
                matched_pairs["track_pos_phi"].append(track_pos_phi[min_index])
                matched_pairs["track_pos_x"].append(track_pos_x_event[min_index])
                matched_pairs["track_pos_y"].append(track_pos_y_event[min_index])
                matched_pairs["track_pos_z"].append(track_pos_z_event[min_index])
                matched_pairs["mc_theta"].append(mc_theta_event[imin_mc])
                matched_pairs["mc_phi"].append(mc_phi_event[imin_mc])
                matched_pairs["min_delta_angle"].append(min_delta_angle)
                matched_pairs["mc_pdg"].append(mc_pdg_event[imin_mc])
                matched_pairs["mc_momentum"].append(mc_momentum_event[imin_mc])
                matched_pairs["mc_vertex_x"].append(mc_vertex_x_event[imin_mc])
                matched_pairs["mc_vertex_y"].append(mc_vertex_y_event[imin_mc])
                matched_pairs["mc_vertex_z"].append(mc_vertex_z_event[imin_mc])
                matched_pairs["track_pathlength"].append(track_pathlength[event_idx][min_index])

                min_delta_angles_all_tracks.append(min_delta_angle)

        MatchingMCAndTrack.plot_efficiency(
                self=self,
                name=name,
                matched_pairs=matched_pairs,
                all_matched_pairs=all_matched_pairs,
                threshold=threshold,
                rootfile=rootfile,  
        )
    
        with open(output_txt, 'a') as f:
            f.write(f'{name}, all tracks: {len(ak.flatten(r_min_track_index))}\n')
            f.write(f'{name}, MC and Track matched tracks: {len(matched_pairs["track_idx"])}\n')
            f.write(f'{name}, MC and Track matched tracks on BTOF: {len(matched_pairs_on_btof["track_idx"])}\n')
            f.write(f'{name}, MC and Track matched tracks on ETOF: {len(matched_pairs_on_etof["track_idx"])}\n')
            f.write(f'{name}, MC and Track matching efficiency: {len(matched_pairs["track_idx"]) / len(ak.flatten(r_min_track_index))}\n')
            f.write(f'{name}, MC and Track matching Number of cut tracks: {len(ak.flatten(r_min_track_index)) - len(matched_pairs["track_idx"])}\n')
            f.write(f'{name}, MC and Track threshold cut efficiency: {len(matched_pairs["track_idx"]) / len(all_matched_pairs["track_idx"])}\n')

        if plot_verbose:
            myfunc.make_histogram_root(min_delta_angles_all_tracks,
                            100,
                            hist_range=[0, 3.2],
                            title='Minimum delta angles for all tracks matched to MC',
                            xlabel='Delta angle [rad]',
                            ylabel='Entries',
                            outputname=f'{name}/min_delta_angles',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(delta_angles_all),
                            100,
                            hist_range=[0, 3.2],
                            title='Delta angles for all tracks matched to MC',
                            xlabel='Delta angle [rad]',
                            ylabel='Entries',
                            outputname=f'{name}/delta_angles_all',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(matched_pairs["track_pathlength"],
                            100,
                            hist_range=[0, 7000],
                            title='Track pathlength matched to MC',
                            xlabel='Pathlength [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/track_pathlength',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(matched_pairs["track_pt"],
                            100,
                            hist_range=[0, 5],
                            title='Track pt matched to MC',
                            xlabel='pt [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/track_pt',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(matched_pairs["mc_momentum"],
                            100,
                            hist_range=[0, 5],
                            title='MC momentum matched to track',
                            xlabel='Momentum [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_momentum',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(matched_pairs["mc_pdg"],
                            100,
                            hist_range=[-250, 250],
                            title='MC PDG ID matched to track',
                            xlabel='PDG ID',
                            ylabel='Entries',
                            outputname=f'{name}/mc_pdg',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(matched_pairs_on_btof["mc_momentum"],
                            100,
                            hist_range=[0, 5],
                            title='MC momentum on BTOF matched to track',
                            xlabel='Momentum [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_momentum_on_btof',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(matched_pairs_on_btof["mc_pdg"],
                            100,
                            hist_range=[-250, 250],
                            title='MC PDG ID on BTOF matched to track',
                            xlabel='PDG ID',
                            ylabel='Entries',
                            outputname=f'{name}/mc_pdg_on_btof',
                            rootfile=rootfile
                            )
            
            
            myfunc.make_histogram_root(matched_pairs["mc_vertex_x"],
                            100,
                            hist_range=[-1000, 1000],
                            title='MC vertex x matched to track',
                            xlabel='x [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_vertex_x',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(matched_pairs["mc_vertex_y"],
                            100,
                            hist_range=[-1000, 1000],
                            title='MC vertex y matched to track',
                            xlabel='y [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_vertex_y',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(matched_pairs["mc_vertex_z"],
                            100,
                            hist_range=[-1000, 1000],
                            title='MC vertex z matched to track',
                            xlabel='z [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/mc_vertex_z',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(matched_pairs_on_btof["track_pathlength"],
                            100,
                            hist_range=[0, 3000],
                            title='Track pathlength on BTOF matched to MC',
                            xlabel='Pathlength [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/track_pathlength_on_btof',
                            rootfile=rootfile
                            )

            myfunc.make_2Dhistogram_root(matched_pairs_on_btof["track_pos_x_on_btof"],
                            100,
                            [-1000, 1000],
                            matched_pairs_on_btof["track_pos_y_on_btof"],
                            100,
                            [-1000, 1000],
                            title='Track pos xy on BTOF matched to MC',
                            xlabel='x [mm]',
                            ylabel='y [mm]',
                            outputname=f'{name}/track_pos_on_btof',
                            rootfile=rootfile
                            )
            
        print('End matching track to MC')

        return matched_pairs, min_delta_angles_all_tracks, matched_pairs_on_btof, matched_pairs_on_etof

    def plot_efficiency(self, name: str, matched_pairs: Dict[str, List], all_matched_pairs: Dict[str, List], threshold: float, rootfile: uproot.TTree) -> None:
        """
        Plots efficiency of track-matching based on a threshold for minimum delta angles, 
        but the final ratio is now plotted in a Root TH2D and stored in the rootfile.
        """

        track_phi_all = ak.to_numpy(all_matched_pairs['track_pos_phi'])
        track_theta_all = ak.to_numpy(all_matched_pairs['track_pos_theta'])

        track_phi_include_threshold = ak.to_numpy(matched_pairs['track_pos_phi'])
        track_theta_include_threshold = ak.to_numpy(matched_pairs['track_pos_theta'])

        # threshold = 0.5
        for threshold in [0.5]:

            myfunc.make_stacked_histogram_root([track_phi_all, track_phi_include_threshold],
                                        nbins=100,
                                        hist_range=[-3.5, 3.5],
                                        labels=['All', f'Include threshold {threshold}'],
                                        title='Track phi comparison with threshold',
                                        xlabel='phi [rad]',
                                        ylabel='Entries',
                                        outputname=f'{name}/track_phi_threshold_{threshold}',
                                        rootfile=rootfile
                                        )
            
            myfunc.make_stacked_histogram_root([track_theta_all, track_theta_include_threshold],
                                        nbins=100,
                                        hist_range=[0, 3.5],
                                        labels=['All', f'Include threshold {threshold}'],
                                        title='Track theta comparison with threshold',
                                        xlabel='theta [rad]',
                                        ylabel='Entries',
                                        outputname=f'{name}/track_theta_threshold_{threshold}',
                                        rootfile=rootfile
                                    )
            
            efficiency = len(track_phi_include_threshold) / len(track_phi_all)
            print(f'Efficiency for threshold {threshold}: {efficiency}')

            fig1, ax1, fill1, x_edge1, y_edge1 = myfunc.make_2Dhistogram(track_phi_all,
                                            32,
                                            [-3.2, 3.2],
                                            track_theta_all,
                                            32,
                                            [0, 3.2],
                                            title='Track phi vs theta',
                                            xlabel='phi [rad]',
                                            ylabel='theta [rad]',
                                            cmap='viridis')

            fill2, ax2, fill2, x_edge2, y_edge2 = myfunc.make_2Dhistogram(track_phi_include_threshold,
                                            32,
                                            [-3.2, 3.2],
                                            track_theta_include_threshold,
                                            32,
                                            [0, 3.2],
                                            title=f'Track phi vs theta include threshold {threshold}',
                                            xlabel='phi [rad]',
                                            ylabel='theta [rad]',
                                            cmap='viridis')

            ratio = np.divide(fill2, fill1, out=np.zeros_like(fill1), where=fill1 != 0)

            x_nbins = len(x_edge1)-1
            y_nbins = len(y_edge1)-1
            hist_eff_name = f"hist_mc_track_efficiency_map_threshold_{threshold}"
            hist_eff = r.TH2D(hist_eff_name,
                            f"Efficiency map (threshold={threshold});phi [rad];theta [rad]",
                            x_nbins, np.array(x_edge1, dtype='float64'),
                            y_nbins, np.array(y_edge1, dtype='float64'))

            for ix in range(x_nbins):
                for iy in range(y_nbins):
                    hist_eff.SetBinContent(ix+1, iy+1, ratio[iy, ix])

            c2 = r.TCanvas(f"c2_{threshold}","EfficiencyMap",800,600)
            hist_eff.SetMinimum(0)  
            hist_eff.SetMaximum(1)  
            hist_eff.Draw("COLZ")
            
            latex = r.TLatex()
            latex.SetNDC(True)
            latex.SetTextSize(0.03)
            latex.DrawLatex(0.15,0.92,f"Threshold={threshold}, Efficiency={efficiency:.3f}")
            c2.Update()

            if rootfile:
                rootfile.cd()
                hist_eff.Write()     
                c2.Write(f"canvas_efficiency_{threshold}")  

class MatchingTrackandToFHits:
    def __init__(self, dis_file, branch):
        self.branch = branch
        self.dis_file = dis_file

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
            myfunc.make_histogram_root(ak.flatten(btof_phi_theta['phi']),
                            100,
                            hist_range=[-3.2, 3.2],
                            title='BToF phi',
                            xlabel='phi [rad]',
                            ylabel='Entries',
                            outputname=f'{name}/btof_phi',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(btof_phi_theta['theta']),
                            100,
                            hist_range=[0, 3.2],
                            title='BToF theta',
                            xlabel='theta [rad]',
                            ylabel='Entries',
                            outputname=f'{name}/btof_theta',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(btof_pos_z),
                            100,
                            hist_range=[-2000, 2000],
                            title='BToF z',
                            xlabel='z [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/btof_z',
                            rootfile=rootfile
                            )
            
            myfunc.make_histogram_root(ak.flatten(btof_r),
                            100,
                            hist_range=[0, 1000],
                            title='BToF r',
                            xlabel='r [mm]',
                            ylabel='Entries',
                            outputname=f'{name}/btof_r',
                            rootfile=rootfile
                            )

        return btof_phi_theta, ectof_phi_theta

    def match_track_to_tof(
            self, 
            name: str, 
            matched_tracks_on_btof: dict, 
            matched_tracks_on_etof: dict, 
            btof_phi_theta: dict, 
            ectof_phi_theta: dict, 
            rootfile: uproot.TTree,
            output_txt: str = 'matching_result.txt',
            verbose: bool = False, 
            plot_verbose: bool = False
        ) -> Tuple[dict, dict]:
        """
        Matches tracks to TOF hits based on angular distances.

        Args:
            name (str): Name for plotting output files.
            matched_tracks_on_btof (dict): Dictionary of matched track information on BTOF.
            matched_tracks_on_etof (dict): Dictionary of matched track information on ETOF.
            btof_phi_theta (dict): Dictionary containing BToF phi, theta, and time.
            ectof_phi_theta (dict): Dictionary containing EToF phi, theta, and time.
            verbose (bool): Flag for printing debug information.
            plot_verbose (bool): Flag for generating plots.

        Returns:
            Tuple[dict, dict]: Dictionaries containing matched tracks with TOF phi, theta, time, and other related information.
        """

        btof_and_track_matched = {
            'event_idx': [],
            'track_idx': [],
            'track_p': [],
            'track_pt': [],
            'track_pos_phi': [],
            'track_pos_theta': [],
            'tof_pos_phi': [],
            'tof_pos_theta': [],
            'track_pos_x': [],
            'track_pos_y': [],
            'track_pos_z': [],
            'tof_time': [],
            'mc_pdg': [],
            'mc_momentum': [],
            'mc_vertex_z': [],
            'track_pathlength': [],
            'delta_angle': [],
        }

        ectof_and_track_matched = {
            'event_idx': [],
            'track_idx': [],
            'track_p': [],
            'track_pt': [],
            'track_pos_phi': [],
            'track_pos_theta': [],
            'track_pos_x': [],
            'track_pos_y': [],
            'track_pos_z': [],
            'tof_pos_phi': [],
            'tof_pos_theta': [],
            'tof_time': [],
            'mc_pdg': [],
            'mc_momentum': [],
            'mc_vertex_z': [],
            'track_pathlength': [],
            'delta_angle': [],
        }
        
        min_delta_angles_events = []
        delta_angles_all = []

        # angle_threshold = 0.013
        angle_threshold = 0.2

        print(f"Number of matched tracks on BTOF: {len(matched_tracks_on_btof['event_idx'])}")

        for i in range(len(matched_tracks_on_btof['event_idx'])):
            print(f"Event {i}")
            print(f"Number of matched tracks on BTOF: {len(matched_tracks_on_btof['event_idx'])}")
            print(f"Matched tracks on BTOF: {matched_tracks_on_btof['event_idx'][i]}")
            if len(matched_tracks_on_btof['event_idx']) == 0:
                if verbose:
                    print(f"No matched tracks on BTOF")
                continue

            event_idx = matched_tracks_on_btof['event_idx'][i]
            track_idx = matched_tracks_on_btof['track_idx'][i]
            track_p = matched_tracks_on_btof['track_momentum_on_btof'][i]
            track_pt = matched_tracks_on_btof['track_momentum_transverse_on_btof'][i]
            track_phi = matched_tracks_on_btof['track_pos_phi_on_btof'][i]
            track_theta = matched_tracks_on_btof['track_pos_theta_on_btof'][i]
            track_pos_x = matched_tracks_on_btof['track_pos_x_on_btof'][i]
            track_pos_y = matched_tracks_on_btof['track_pos_y_on_btof'][i]
            track_pos_z = matched_tracks_on_btof['track_pos_z_on_btof'][i]

            print(f"Event {event_idx}")
            print(f"Track {track_idx}")
            print(f"Track momentum: {track_p}")
            print(f"Track phi: {track_phi}")

            if event_idx >= len(btof_phi_theta['phi']):
                if verbose:
                    print(f"Event {event_idx} exceeds BTOF data hist_range")
                continue

            tof_phis = btof_phi_theta['phi'][event_idx]
            tof_thetas = btof_phi_theta['theta'][event_idx]
            tof_times = btof_phi_theta['time'][event_idx]

            if len(tof_phis) == 0:
                if verbose:
                    print(f"No BTOF hits in event {event_idx}")
                continue

            delta_angles = angular_distance(
                np.array([track_theta]*len(tof_thetas)),
                np.array([track_phi]*len(tof_phis)),
                np.array(tof_thetas),
                np.array(tof_phis)
            )

            delta_angles_all.append(delta_angles)
            min_idx = np.argmin(delta_angles)
            min_delta_angle = delta_angles[min_idx]

            if min_delta_angle < angle_threshold:
                btof_and_track_matched['event_idx'].append(event_idx)
                btof_and_track_matched['track_idx'].append(track_idx)
                btof_and_track_matched['track_p'].append(track_p)
                btof_and_track_matched['track_pt'].append(track_pt)
                btof_and_track_matched['track_pos_phi'].append(track_phi)
                btof_and_track_matched['track_pos_theta'].append(track_theta)
                btof_and_track_matched['track_pos_x'].append(track_pos_x)
                btof_and_track_matched['track_pos_y'].append(track_pos_y)
                btof_and_track_matched['track_pos_z'].append(track_pos_z)
                btof_and_track_matched['tof_pos_phi'].append(tof_phis[min_idx])
                btof_and_track_matched['tof_pos_theta'].append(tof_thetas[min_idx])
                btof_and_track_matched['tof_time'].append(tof_times[min_idx])
                btof_and_track_matched['mc_pdg'].append(matched_tracks_on_btof['mc_pdg'][i])
                btof_and_track_matched['mc_momentum'].append(matched_tracks_on_btof['mc_momentum'][i])
                btof_and_track_matched['mc_vertex_z'].append(matched_tracks_on_btof['mc_vertex_z'][i])
                btof_and_track_matched['track_pathlength'].append(matched_tracks_on_btof['track_pathlength'][i])
                btof_and_track_matched['delta_angle'].append(min_delta_angle)
                min_delta_angles_events.append(min_delta_angle)
                if verbose:
                    print(f"Matched track  in event {event_idx} with BTOF hit {min_idx}, delta_angle={min_delta_angle}")
            else:
                if verbose:
                    print(f"No BTOF match for track  in event {event_idx}, min_delta_angle={min_delta_angle}")

        for i in range(len(matched_tracks_on_etof['event_idx'])):
            if len(matched_tracks_on_etof['event_idx']) == 0:
                if verbose:
                    print(f"No matched tracks on ETOF")
                continue

            event_idx = matched_tracks_on_etof['event_idx'][i]
            track_idx = matched_tracks_on_etof['track_idx'][i]
            track_p = matched_tracks_on_etof['track_momentum_on_etof'][i]
            track_pt = matched_tracks_on_etof['track_momentum_transverse_on_etof'][i]
            track_phi = matched_tracks_on_etof['track_pos_phi_on_etof'][i]
            track_theta = matched_tracks_on_etof['track_pos_theta_on_etof'][i]
            track_pos_x = matched_tracks_on_etof['track_pos_x_on_etof'][i]
            track_pos_y = matched_tracks_on_etof['track_pos_y_on_etof'][i]
            track_pos_z = matched_tracks_on_etof['track_pos_z_on_etof'][i]

            if event_idx >= len(ectof_phi_theta['phi']):
                if verbose:
                    print(f"Event {event_idx} exceeds ETOF data hist_range")
                continue

            tof_phis = ectof_phi_theta['phi'][event_idx]
            tof_thetas = ectof_phi_theta['theta'][event_idx]
            tof_times = ectof_phi_theta['time'][event_idx]

            if len(tof_phis) == 0:
                if verbose:
                    print(f"No ETOF hits in event {event_idx}")
                continue

            delta_angles = angular_distance(
                np.array([track_theta]*len(tof_thetas)),
                np.array([track_phi]*len(tof_phis)),
                np.array(tof_thetas),
                np.array(tof_phis)
            )

            min_idx = np.argmin(delta_angles)
            min_delta_angle = delta_angles[min_idx]

            if min_delta_angle < angle_threshold:
                ectof_and_track_matched['event_idx'].append(event_idx)
                ectof_and_track_matched['track_idx'].append(track_idx)
                ectof_and_track_matched['track_p'].append(track_p)
                ectof_and_track_matched['track_pt'].append(track_pt)
                ectof_and_track_matched['track_pos_phi'].append(track_phi)
                ectof_and_track_matched['track_pos_theta'].append(track_theta)
                ectof_and_track_matched['tof_pos_phi'].append(tof_phis[min_idx])
                ectof_and_track_matched['tof_pos_theta'].append(tof_thetas[min_idx])
                ectof_and_track_matched['track_pos_x'].append(track_pos_x)
                ectof_and_track_matched['track_pos_y'].append(track_pos_y)
                ectof_and_track_matched['track_pos_z'].append(track_pos_z)
                ectof_and_track_matched['tof_time'].append(tof_times[min_idx])
                ectof_and_track_matched['mc_pdg'].append(matched_tracks_on_etof['mc_pdg'][i])
                ectof_and_track_matched['mc_momentum'].append(matched_tracks_on_etof['mc_momentum'][i])
                ectof_and_track_matched['mc_vertex_z'].append(matched_tracks_on_etof['mc_vertex_z'][i])
                ectof_and_track_matched['track_pathlength'].append(matched_tracks_on_etof['track_pathlength'][i])
                ectof_and_track_matched['delta_angle'].append(min_delta_angle)
                min_delta_angles_events.append(min_delta_angle)
                if verbose:
                    print(f"Matched track in event {event_idx} with ETOF hit {min_idx}, delta_angle={min_delta_angle}")
            else:
                if verbose:
                    print(f"No ETOF match for track in event {event_idx}, min_delta_angle={min_delta_angle}")

        with open(output_txt, 'a') as f:
            f.write(f'{name}, all tracks before bTOF and track matching: {len(matched_tracks_on_btof["event_idx"])}\n')
            f.write(f'{name}, all tracks before eTOF and track matching: {len(matched_tracks_on_etof["event_idx"])}\n')
            f.write(f'{name}, all tracks after bTOF and track matching: {len(btof_and_track_matched["event_idx"])}\n')
            f.write(f'{name}, all tracks after eTOF and track matching: {len(ectof_and_track_matched["event_idx"])}\n')
            f.write(f'{name}, efficiency bTOF: {len(btof_and_track_matched["event_idx"]) / len(matched_tracks_on_btof["event_idx"])}\n') if len(matched_tracks_on_btof["event_idx"]) > 0 else f.write(f'{name}, efficiency bTOF: 0\n')
            f.write(f'{name}, efficiency eTOF: {len(ectof_and_track_matched["event_idx"]) / len(matched_tracks_on_etof["event_idx"])}\n') if len(matched_tracks_on_etof["event_idx"]) > 0 else f.write(f'{name}, efficiency eTOF: 0\n')


        if verbose:
            print(f"Total matched tracks on BTOF a: {len(btof_and_track_matched['event_idx'])}")
            print(f"Total matched tracks on ETOF: {len(ectof_and_track_matched['event_idx'])}")

        if plot_verbose:
            myfunc.make_histogram_root(
                ak.flatten(delta_angles_all),
                100,
                hist_range=[0, angle_threshold],
                title='Delta angles for all tracks matched to TOF',
                xlabel='Delta angle [rad]',
                ylabel='Entries',
                outputname=f'{name}/delta_angles_match_track_to_tof',
                rootfile=rootfile
            )

            myfunc.make_histogram_root(
                min_delta_angles_events,
                100,
                hist_range=[0, angle_threshold],
                title='Minimum delta angles for all tracks matched to TOF',
                xlabel='Delta angle [rad]',
                ylabel='Entries',
                outputname=f'{name}/min_delta_angles_match_track_to_tof',
                rootfile=rootfile
            )

            if len(btof_and_track_matched['tof_time']) > 0:
                myfunc.make_histogram_root(
                    btof_and_track_matched['tof_time'],
                    100,
                    hist_range=[0, 10],
                    title='BTOF Time matched track to TOF',
                    xlabel='Time [ns]',
                    ylabel='Entries',
                    outputname=f'{name}/btof_time_match_track_to_tof',
                    rootfile=rootfile
                )

                myfunc.make_histogram_root(
                    btof_and_track_matched['track_p'],
                    100,
                    hist_range=[0, 5],
                    title='BTOF Track Momentum matched track to TOF',
                    xlabel='Momentum [GeV]',
                    ylabel='Entries',
                    outputname=f'{name}/btof_track_momentum_match_track_to_tof',
                    rootfile=rootfile
                )

                myfunc.make_histogram_root(
                    btof_and_track_matched['mc_pdg'],
                    100,
                    hist_range=[-250, 250],
                    title='BTOF MC PDG ID matched track to TOF',
                    xlabel='PDG ID',
                    ylabel='Entries',
                    outputname=f'{name}/btof_mc_pdg_match_track_to_tof',
                    rootfile=rootfile
                )

                myfunc.make_histogram_root(
                    btof_and_track_matched['track_pathlength'],
                    100,
                    hist_range=[0, 3000],
                    title='BTOF Track Pathlength matched track to TOF',
                    xlabel='Pathlength [mm]',
                    ylabel='Entries',
                    outputname=f'{name}/btof_track_pathlength_match_track_to_tof',
                    rootfile=rootfile
                )

        return btof_and_track_matched, ectof_and_track_matched

class MatchingEventDisplay:
    def __init__(self, dis_file, branch):
        self.branch = branch
        self.dis_file = dis_file

    def get_match_track_info(
            self, 
            name: str, 
            btof_track_matched: dict,
            ectof_track_matched: dict,
            SELECTED_EVENTS: int, 
            rootfile: uproot.TTree, 
            verbose: bool = False, 
            plot_verbose: bool = False
            ) -> Tuple[dict, dict]:
        """
        Retrieves matched track information for BTOF and ETOF detectors.

        Args:
            name (str): Name for plotting output files.
            SELECTED_EVENTS (int): Number of events to process.
            verbose (bool): Flag for printing debug information.
            plot_verbose (bool): Flag for generating plots.

        Returns:
            Tuple[dict, dict]: Dictionaries containing matched track information for BTOF and ETOF detectors.
        """

        track_posx_on_btof = btof_track_matched['track_pos_x']
        track_posy_on_btof = btof_track_matched['track_pos_y']
        track_posz_on_btof = btof_track_matched['track_pos_z']

        vertex_z = btof_track_matched['mc_vertex_z']

class ToFPIDPerformance:
    def __init__(self, dis_file, branch):
        self.branch = branch
        self.dis_file = dis_file

    def plot_pid_performance(self, 
                             name: str, 
                             btof_and_track_matched: dict, 
                             ectof_and_track_matched: dict, 
                             rootfile: uproot.TTree, 
                             MERGIN_PI: float = 100, 
                             MERGIN_K: float = 100, 
                             MERGIN_P: float = 100, 
                             LARGE_MERGIN_PI: float = 200, 
                             LARGE_MERGIN_K: float = 200, 
                             LARGE_MERGIN_P: float = 200, 
                             MOMENTUM_RANGE: float = 2.5,
                             output_txt_name: str = f'pid_result.txt',
                             ):
        """
        Calculates and returns PID performance metrics.

        Args:
            matched_tracks_and_tof_phi_theta (dict): Matched tracks and TOF information.

        Returns:
            Tuple: BTOF and ETOF metrics (momentums, beta inverses, and calculated masses).
        """

        btof_time = btof_and_track_matched['tof_time']
        btof_phi = btof_and_track_matched['tof_pos_phi']
        btof_theta = btof_and_track_matched['tof_pos_theta']
        track_momentum_on_btof = btof_and_track_matched['track_p']
        track_momentum_transverse_on_btof = btof_and_track_matched['track_pt']
        btof_pdg = btof_and_track_matched['mc_pdg']
        btof_vertex_z = btof_and_track_matched['mc_vertex_z']
        btof_pathlength = btof_and_track_matched['track_pathlength']
        
        ectof_time = ectof_and_track_matched['tof_time']
        ectof_phi = ectof_and_track_matched['tof_pos_phi']
        ectof_theta = ectof_and_track_matched['tof_pos_theta']
        track_momentum_on_ectof = ectof_and_track_matched['track_p']
        ectof_pdg = ectof_and_track_matched['mc_pdg']
        ectof_vertex_z = ectof_and_track_matched['mc_vertex_z']
        ectof_pathlength = ectof_and_track_matched['track_pathlength']

        track_momentums_on_btof = []
        track_momentums_transverse_on_btof = []
        track_momentums_on_ectof = []
        btof_beta_inversees = []
        etof_beta_inversees = []
        btof_calc_mass = []
        etof_calc_mass = []

        for i in range(len(btof_time)):
            current_time = btof_time[i]
            btof_beta = btof_pathlength[i] / current_time
            btof_beta_c = btof_beta / 299.792458  # Speed of light in mm/ns
            btof_beta_inverse = 1 / btof_beta_c
            calc_mass = 1000 * track_momentum_on_btof[i] * np.sqrt(1 - btof_beta_c**2) / btof_beta_c
            btof_beta_inversees.append(btof_beta_inverse)
            btof_calc_mass.append(calc_mass)
            track_momentums_on_btof.append(track_momentum_on_btof[i])
            track_momentums_transverse_on_btof.append(track_momentum_transverse_on_btof[i])

        for i in range(len(ectof_time)):
            current_time = ectof_time[i]
            etof_beta = ectof_pathlength[i] / current_time
            etof_beta_c = etof_beta / 299.792458  # Speed of light in mm/ns
            etof_beta_inverse = 1 / etof_beta_c
            calc_mass = 1000 * track_momentum_on_ectof[i] * np.sqrt(1 - etof_beta_c**2) / etof_beta_c
            etof_beta_inversees.append(etof_beta_inverse)
            etof_calc_mass.append(calc_mass)
            track_momentums_on_ectof.append(track_momentum_on_ectof[i])
    
        myfunc.make_histogram_root(
            track_momentums_on_btof,
                           100,
                           hist_range=[0, 5],
                        title='BTOF Momentum PID Performance',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_momentum_pid_performance',
                        rootfile=rootfile
        )
        
        myfunc.make_histogram_root(
            track_momentum_on_ectof,
                           100,
                           hist_range=[0, 5],
                        title='ETOF Momentum PID Performance',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{name}/etof_momentum_pid_performance',
                        rootfile=rootfile
        )

        myfunc.make_histogram_root(
            btof_beta_inversees,
                        100,
                        hist_range=[0.8, 1.8],
                        title='BTOF Beta Inverse PID Performance',
                        xlabel='Beta Inverse',
                        ylabel='Entries',
                        outputname=f'{name}/btof_beta_inverse_pid_performance',
                        rootfile=rootfile
        )
        
        myfunc.make_histogram_root(
            btof_calc_mass,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF Calculated Mass',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_pid_performance',
                        rootfile=rootfile
        )
                
        m_pi = 139 # MeV
        m_k = 493 # MeV
        m_p = 938 # MeV

        pi_calc_mass_on_btof = []
        k_calc_mass_on_btof = []
        p_calc_mass_on_btof = []
        pi_mass_count_btof = 0
        pi_mass_count_btof_large_mergin = 0
        pi_mass_count_btof_low_momentum = 0
        k_mass_count_btof = 0
        k_mass_count_btof_large_mergin = 0
        k_mass_count_btof_low_momentum = 0
        p_mass_count_btof = 0
        p_mass_count_btof_large_mergin = 0
        p_mass_count_btof_low_momentum = 0

        pi_low_momentum_btof = []
        pi_momentum_in_low_momentum_btof = []
        k_low_momentum_btof = []
        k_momentum_in_low_momentum_btof = []
        p_low_momentum_btof = []
        p_momentum_in_low_momentum_btof = []

        for i in range(len(btof_calc_mass)):
            if track_momentums_on_btof[i] < MOMENTUM_RANGE:
                if btof_pdg[i] == 211 or btof_pdg[i] == -211:
                    pi_low_momentum_btof.append(btof_calc_mass[i])
                    pi_momentum_in_low_momentum_btof.append(track_momentums_on_btof[i])

                if btof_pdg[i] == 321 or btof_pdg[i] == -321:
                    k_low_momentum_btof.append(btof_calc_mass[i])
                    k_momentum_in_low_momentum_btof.append(track_momentums_on_btof[i])

                if btof_pdg[i] == 2212 or btof_pdg[i] == -2212:
                    p_low_momentum_btof.append(btof_calc_mass[i])
                    p_momentum_in_low_momentum_btof.append(track_momentums_on_btof[i])

        for i in range(len(btof_calc_mass)):
            if btof_pdg[i] == 211 or btof_pdg[i] == -211:
                pi_calc_mass_on_btof.append(btof_calc_mass[i])
                if -MERGIN_PI < btof_calc_mass[i] - m_pi < MERGIN_PI:
                    pi_mass_count_btof += 1
                if -m_pi < btof_calc_mass[i] - m_pi < LARGE_MERGIN_PI:
                    pi_mass_count_btof_large_mergin += 1
                if track_momentums_on_btof[i] < MOMENTUM_RANGE:
                    if -MERGIN_PI < btof_calc_mass[i] - m_pi < MERGIN_PI:
                        pi_mass_count_btof_low_momentum += 1

            if btof_pdg[i] == 321 or btof_pdg[i] == -321:
                k_calc_mass_on_btof.append(btof_calc_mass[i])
                if -MERGIN_K < btof_calc_mass[i] - m_k < MERGIN_K:
                    k_mass_count_btof += 1
                if -LARGE_MERGIN_K < btof_calc_mass[i] - m_k < LARGE_MERGIN_K:
                    k_mass_count_btof_large_mergin += 1
                if track_momentums_on_btof[i] < MOMENTUM_RANGE:
                    if -MERGIN_K < btof_calc_mass[i] - m_k < MERGIN_K:
                        k_mass_count_btof_low_momentum += 1

            if btof_pdg[i] == 2212 or btof_pdg[i] == -2212:
                p_calc_mass_on_btof.append(btof_calc_mass[i])
                if -MERGIN_P < btof_calc_mass[i] - m_p < MERGIN_P:
                    p_mass_count_btof += 1
                if -LARGE_MERGIN_P < btof_calc_mass[i] - m_p < LARGE_MERGIN_P:
                    p_mass_count_btof_large_mergin += 1
                if track_momentums_on_btof[i] < MOMENTUM_RANGE:
                    if -MERGIN_P < btof_calc_mass[i] - m_p < MERGIN_P:
                        p_mass_count_btof_low_momentum += 1

        pi_eff_btof = pi_mass_count_btof / len(pi_calc_mass_on_btof) if len(pi_calc_mass_on_btof) > 0 else 0
        pi_eff_btof_low_momentum = pi_mass_count_btof_low_momentum / len(pi_low_momentum_btof) if len(pi_low_momentum_btof) > 0 else 0
        pi_eff_btof_large_mergin = pi_mass_count_btof_large_mergin / len(pi_calc_mass_on_btof) if len(pi_calc_mass_on_btof) > 0 else 0
        k_eff_btof = k_mass_count_btof / len(k_calc_mass_on_btof) if len(k_calc_mass_on_btof) > 0 else 0
        k_eff_btof_low_momentum = k_mass_count_btof_low_momentum / len(k_low_momentum_btof) if len(k_low_momentum_btof) > 0 else 0
        k_eff_btof_large_mergin = k_mass_count_btof_large_mergin / len(k_calc_mass_on_btof) if len(k_calc_mass_on_btof) > 0 else 0
        p_eff_btof = p_mass_count_btof / len(p_calc_mass_on_btof) if len(p_calc_mass_on_btof) > 0 else 0
        p_eff_btof_low_momentum = p_mass_count_btof_low_momentum / len(p_low_momentum_btof) if len(p_low_momentum_btof) > 0 else 0
        p_eff_btof_large_mergin = p_mass_count_btof_large_mergin / len(p_calc_mass_on_btof) if len(p_calc_mass_on_btof) > 0 else 0

        with open(output_txt_name, 'a') as f:
            f.write(f'BTOF PID Performance\n')
            f.write(f'Pi Efficiency: {pi_eff_btof}\n')
            f.write(f'K Efficiency: {k_eff_btof}\n')
            f.write(f'P Efficiency: {p_eff_btof}\n')
            f.write(f'BTOF Low Momentum PID Performance\n')
            f.write(f'Pi Efficiency: {pi_eff_btof_low_momentum}\n')
            f.write(f'K Efficiency: {k_eff_btof_low_momentum}\n')
            f.write(f'P Efficiency: {p_eff_btof_low_momentum}\n')
            f.write(f'BTOF Large Mergin PID Performance\n')
            f.write(f'Pi Efficiency: {pi_eff_btof_large_mergin}\n')
            f.write(f'K Efficiency: {k_eff_btof_large_mergin}\n')
            f.write(f'P Efficiency: {p_eff_btof_large_mergin}\n')

        myfunc.make_histogram_root(
            pi_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF Calculated Mass for Pi',
                        xlabel='Mass [GeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_pi_pid_performance',
                        rootfile=rootfile
        )
        
        myfunc.make_histogram_root(
            k_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF Calculated Mass for K',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_k_pid_performance',
                        rootfile=rootfile
        )
        
        myfunc.make_histogram_root(
            p_calc_mass_on_btof,
                        100,
                        hist_range=[200, 1200],
                        title='BTOF Calculated Mass for P',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_p_pid_performance',
                        rootfile=rootfile
        )
    
        myfunc.make_2Dhistogram_root(
            track_momentums_on_btof,
            100,
            [0, 3.5],
            btof_beta_inversees,
            100,
            [0.8, 3.5],
            title='BTOF Momentum vs Beta Inverse',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{name}/btof_momentum_vs_beta_inverse_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_on_btof,
            100,
            [0, 5],
            btof_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF Momentum vs Beta Inverse',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{name}/btof_momentum_vs_beta_inverse_pid_performance_diff_range',
            cmap='plasma',
            logscale=True,
            rootfile=rootfile
        )

        # ==================== π/K separation =====================
        # separation power = |mu_pi - mu_K| / sqrt(1/2 sigma_pi^2 + sigma_K^2 )

        pi_calc_mass_on_btof = np.array(pi_calc_mass_on_btof)
        k_calc_mass_on_btof  = np.array(k_calc_mass_on_btof)

        with open(output_txt_name, 'a') as f:
            f.write("\n----- Separation Power with TF1 Gaussian Fit -----\n")

        if len(pi_calc_mass_on_btof) > 10 and len(k_calc_mass_on_btof) > 10:

            pi_mean  = np.mean(pi_calc_mass_on_btof)
            pi_sigma = np.std(pi_calc_mass_on_btof)
            k_mean   = np.mean(k_calc_mass_on_btof)
            k_sigma  = np.std(k_calc_mass_on_btof)

            if pi_sigma>0 and k_sigma>0:
                sep_power_simple = abs(pi_mean - k_mean)/np.sqrt(1/2 * (pi_sigma**2 + k_sigma**2))
            else:
                sep_power_simple = -999

            print(f"[PID] Pi/K separation power (simple std calc) = {sep_power_simple:.3f}")
            with open(output_txt_name, 'a') as f:
                f.write(f"Pi/K separation power (simple std calc): {sep_power_simple}\n")


            hist_pi =r.TH1D("hist_pi","Pi reconstructed mass;Mass [MeV];Entries",100,0,1000)
            for val in pi_calc_mass_on_btof:
                hist_pi.Fill(val)

            hist_k = r.TH1D("hist_k","K reconstructed mass;Mass [MeV];Entries",100,0,1000)
            for val in k_calc_mass_on_btof:
                hist_k.Fill(val)

            # --- π Fit ---
            hist_pi.Fit("gaus","Q")  
            f1_pi = hist_pi.GetFunction("gaus")
            if f1_pi:
                A_pi     = f1_pi.GetParameter(0)
                mu_pi    = f1_pi.GetParameter(1)
                sigma_pi = f1_pi.GetParameter(2)
            else:
                mu_pi, sigma_pi = 0, 0

            # --- K Fit ---
            hist_k.Fit("gaus","Q")
            f1_k = hist_k.GetFunction("gaus")
            if f1_k:
                A_k     = f1_k.GetParameter(0)
                mu_k    = f1_k.GetParameter(1)
                sigma_k = f1_k.GetParameter(2)
            else:
                mu_k, sigma_k = 0, 0

            if sigma_pi>1e-7 and sigma_k>1e-7:
                sep_power_gaus = abs(mu_pi - mu_k)/np.sqrt(sigma_pi**2 + sigma_k**2)
            else:
                sep_power_gaus = -999

            print(f"[PID] Pi/K separation power (TF1 Gaussian fit) = {sep_power_gaus:.3f}")
            with open(output_txt_name, 'a') as f:
                f.write(f"Pi/K separation power (TF1 gaus): {sep_power_gaus}\n")
                f.write("========================================================\n")

            c1 = r.TCanvas("c1","Separation Power TF1",800,600)
            hist_pi.SetLineColor(r.kRed)
            hist_pi.SetLineWidth(2)
            hist_pi.Draw()

            hist_k.SetLineColor(r.kBlue)
            hist_k.SetLineWidth(2)
            hist_k.Draw("same")

            legend = r.TLegend(0.65,0.70,0.88,0.88)
            legend.AddEntry(hist_pi, f"Pi mass: #mu={mu_pi:.1f},#sigma={sigma_pi:.1f}", "l")
            legend.AddEntry(hist_k,  f"K mass:  #mu={mu_k:.1f},#sigma={sigma_k:.1f}", "l")
            legend.SetBorderSize(1)
            legend.SetFillColor(0)
            legend.Draw()

            latex = r.TLatex()
            latex.SetNDC(True)
            latex.SetTextSize(0.03)
            latex.DrawLatex(0.15, 0.85, f"SepPower={sep_power_gaus:.3f}")
            c1.Update()

            if rootfile:
                rootfile.cd()
                hist_pi.Write("hist_pi_mass")
                hist_k.Write("hist_k_mass")
                c1.Write("c1_sep_power_tf1")
        else:
            print("[PID] Not enough pi or K data to calculate separation power (TF1).")
            with open(output_txt_name, 'a') as f:
                f.write("Not enough pi/K data for TF1 gaus fit.\n")

        # ==================== Separation Power =====================

        myfunc.make_histogram_root(
            pi_low_momentum_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF Calculated Mass for Pi (Low Momentum)',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_pi_low_momentum_pid_performance',
                        rootfile=rootfile
        )

        myfunc.make_histogram_root(
            k_low_momentum_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF Calculated Mass for K (Low Momentum)',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_k_low_momentum_pid_performance',
                        rootfile=rootfile
        )

        myfunc.make_histogram_root(
            p_low_momentum_btof,
                        100,
                        hist_range=[200, 1200],
                        title='BTOF Calculated Mass for P (Low Momentum)',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_p_low_momentum_pid_performance',
                        rootfile=rootfile
        )

        btof_calc_mass = np.array(btof_calc_mass)
        btof_pdg       = np.array(btof_pdg)      
        track_momentums_on_btof = np.array(track_momentums_on_btof)
        track_momentums_transverse_on_btof = np.array(track_momentums_transverse_on_btof)

        return btof_calc_mass, btof_pdg, track_momentums_on_btof, track_momentums_transverse_on_btof

    def plot_separation_power_vs_momentum(
        self,
        btof_calc_mass: np.ndarray,
        btof_pdg: np.ndarray,
        track_momentums_on_btof: np.ndarray,
        track_momentums_transverse_on_btof: np.ndarray,
        name: str = "test",
        nbins: int = 35,
        momentum_range: tuple = (0, 3.5),
        rootfile=None
    ):
        """

        """

        pi_mask = (btof_pdg ==  211) | (btof_pdg == -211)
        k_mask  = (btof_pdg ==  321) | (btof_pdg == -321)
        p_mask  = (btof_pdg == 2212) | (btof_pdg == -2212)

        pi_mass_all = btof_calc_mass[pi_mask]
        pi_mom_all  = track_momentums_transverse_on_btof[pi_mask]
        k_mass_all  = btof_calc_mass[k_mask]
        k_mom_all   = track_momentums_on_btof[k_mask]

        p_bins      = np.linspace(momentum_range[0], momentum_range[1], nbins+1)
        bin_centers = 0.5 * (p_bins[:-1] + p_bins[1:])
        separation_list = []

        for i in range(nbins):
            p_low  = p_bins[i]
            p_high = p_bins[i+1]

            pi_in_bin = pi_mass_all[(pi_mom_all >= p_low) & (pi_mom_all < p_high)]
            k_in_bin  = k_mass_all[(k_mom_all  >= p_low) & (k_mom_all  < p_high)]


            if len(pi_in_bin) < 5 or len(k_in_bin) < 5:
                separation_list.append(None)
                continue

            hist_pi_name = f"hist_pi_bin_sep{i}"
            hist_pi = r.TH1F(hist_pi_name, ";Mass [MeV];Entries", 100, 0, 1000)
            for val in pi_in_bin:
                hist_pi.Fill(val)

            hist_pi.SetTitle(f"Pi Mass in {p_low:.2f} - {p_high:.2f} GeV")

            bin_max   = hist_pi.GetMaximumBin()
            x_max     = hist_pi.GetBinCenter(bin_max)  # peak position
            ampl      = hist_pi.GetBinContent(bin_max) # amplitude
            rms       = hist_pi.GetRMS()               # RMS

            f_pi = r.TF1("f_pi","[0]*exp(-0.5*((x-[1])/[2])**2)", 0, 1000)
            f_pi.SetParameters(ampl, x_max, rms)
            f_pi.SetParLimits(2, 1e-3, 200)  # limit sigma to be positive

            hist_pi.Fit(f_pi, "Q")
            A_pi    = f_pi.GetParameter(0)
            mu_pi   = f_pi.GetParameter(1)
            sigma_pi= f_pi.GetParameter(2)
            hist_k_name = f"hist_k_bin_sep{i}"
            hist_k = r.TH1F(hist_k_name, ";Mass [MeV];Entries", 100, 0, 1000)
            for val in k_in_bin:
                hist_k.Fill(val)

            hist_k.SetTitle(f"K Mass in {p_low:.2f} - {p_high:.2f} GeV")

            f_k = r.TF1("f_k","[0]*exp(-0.5*((x-[1])/[2])**2)", 0, 1000)
            f_k.SetParameters(hist_k.GetMaximum(), 490, 20) 
            hist_k.Fit(f_k, "Q")
            A_k     = f_k.GetParameter(0)
            mu_k    = f_k.GetParameter(1)
            sigma_k = f_k.GetParameter(2)

            # separation power
            sep_power = None
            if sigma_pi>1e-7 and sigma_k>1e-7:
                sep_power = abs(mu_pi - mu_k)/np.sqrt(1/2 * (sigma_pi**2 + sigma_k**2))

            separation_list.append(sep_power)

            if rootfile:
                hist_pi.Write()
                hist_k.Write()
                f_pi.Write()
                f_k.Write()

        separation_list = np.array(separation_list, dtype=object)
        valid_mask = (separation_list != None)
        valid_sep = separation_list[valid_mask].astype(float)
        valid_bin_center = bin_centers[valid_mask]

        gr = r.TGraph()
        gr.SetName("sep_power_vs_mom")
        gr.SetTitle("Separation Power vs Momentum;pt [GeV];Separation Power")
        idx = 0
        for bc, sep in zip(valid_bin_center, valid_sep):
            gr.SetPoint(idx, bc, sep)
            idx += 1

        if rootfile:
            gr.Write()  

        c1 = r.TCanvas("c1","Separation Power", 800,600)
        c1.SetLogy()

        gr.GetXaxis().SetLimits(0, 3.5)
        gr.GetYaxis().SetRangeUser(0, 50)
        gr.Draw("AP")  
        gr.SetMarkerStyle(20)
        gr.SetMarkerColor(r.kRed)
        gr.SetMarkerSize(1.3)

        sigma_line = r.TLine(0, 3, 3.5, 3)
        sigma_line.SetLineColor(r.kRed)
        sigma_line.SetLineStyle(2)
        sigma_line.Draw("same")

        c1.Update()

        if rootfile:
            c1.Write("canvas_sep_power_logy")  
            
        if rootfile:
            gr = r.TGraph()
            gr.SetName("sep_power_vs_mom")
            idx = 0
            for bc, sep in zip(valid_bin_center, valid_sep):
                gr.SetPoint(idx, bc, sep)
                idx+=1
            gr.Write()

        return valid_bin_center, valid_sep

    def plot_pid_performance_vs_momentum(
        self,
        btof_calc_mass: np.ndarray,
        btof_pdg: np.ndarray,
        track_momentums_on_btof: np.ndarray,
        track_momentums_transverse_on_btof: np.ndarray,
        name: str = "test",
        nbins: int = 35,
        momentum_range: tuple = (0, 3.5),
        MERGIN_PI: float = 100,
        MERGIN_K: float = 100,
        MERGIN_P: float = 100,
        rootfile=None
    ):
        """

        """

        pi_mask = (btof_pdg ==  211) | (btof_pdg == -211)
        k_mask  = (btof_pdg ==  321) | (btof_pdg == -321)
        p_mask = (btof_pdg ==  2212) | (btof_pdg == -2212)

        pi_mass_all = btof_calc_mass[pi_mask]
        pi_mom_all  = track_momentums_on_btof[pi_mask]
        k_mass_all  = btof_calc_mass[k_mask]
        k_mom_all   = track_momentums_on_btof[k_mask]

        p_mass_all = btof_calc_mass[p_mask]
        p_mom_all  = track_momentums_on_btof[p_mask]

        p_bins      = np.linspace(momentum_range[0], momentum_range[1], nbins+1)
        bin_centers = 0.5 * (p_bins[:-1] + p_bins[1:])

        pi_mass_count_list = []
        k_mass_count_list = []
        p_mass_count_list = []

        pi_mass_correct_count_list = []
        k_mass_correct_count_list = []
        p_mass_correct_count_list = []
    
        PI_MASS = 139.57039
        KAON_MASS = 493.677
        PROTON_MASS = 938.272

        for i in range(nbins):
            p_low  = p_bins[i]
            p_high = p_bins[i+1]

            pi_in_bin = pi_mass_all[(pi_mom_all >= p_low) & (pi_mom_all < p_high)]
            k_in_bin  = k_mass_all[(k_mom_all  >= p_low) & (k_mom_all  < p_high)]
            p_in_bin  = p_mass_all[(p_mom_all  >= p_low) & (p_mom_all  < p_high)]

            pi_mass_count_btof = 0
            k_mass_count_btof = 0
            p_mass_count_btof = 0
            pi_mass_correct_count_btof = 0
            k_mass_correct_count_btof = 0
            p_mass_correct_count_btof = 0

            hist_pi_name = f"hist_pi_bin{i}"
            hist_pi = r.TH1F(hist_pi_name, ";Mass [MeV];Entries", 100, 0, 1000)
            for val in pi_in_bin:
                hist_pi.Fill(val)

                pi_mass_count_btof += 1

                if -MERGIN_PI < val - PI_MASS < MERGIN_PI:
                    pi_mass_correct_count_btof += 1


            hist_pi.SetTitle(f"Pi Mass in {p_low:.2f} - {p_high:.2f} GeV")

            hist_k_name = f"hist_k_bin{i}"
            hist_k = r.TH1F(hist_k_name, ";Mass [MeV];Entries", 100, 0, 1000)

            for val in k_in_bin:
                hist_k.Fill(val)

                k_mass_count_btof += 1

                if -MERGIN_K < val - KAON_MASS < MERGIN_K:
                    k_mass_correct_count_btof += 1

            hist_k.SetTitle(f"K Mass in {p_low:.2f} - {p_high:.2f} GeV")

            hist_p_name = f"hist_p_bin{i}"
            hist_p = r.TH1F(hist_p_name, ";Mass [MeV];Entries", 100, 0, 1000)
            for val in p_in_bin:
                hist_p.Fill(val)

                p_mass_count_btof += 1

                if -MERGIN_P < val - PROTON_MASS < MERGIN_P:
                    p_mass_correct_count_btof += 1

            pi_mass_count_list.append(pi_mass_count_btof)
            k_mass_count_list.append(k_mass_count_btof)
            p_mass_count_list.append(p_mass_count_btof)

            pi_mass_correct_count_list.append(pi_mass_correct_count_btof)
            k_mass_correct_count_list.append(k_mass_correct_count_btof)
            p_mass_correct_count_list.append(p_mass_correct_count_btof)

            if rootfile:
                hist_pi.Write()
                hist_k.Write()
                hist_p.Write()

        pi_eff_btof = np.array(pi_mass_correct_count_list) / np.array(pi_mass_count_list) if len(pi_mass_count_list) > 0 else 0
        k_eff_btof = np.array(k_mass_correct_count_list) / np.array(k_mass_count_list) if len(k_mass_count_list) > 0 else 0
        p_eff_btof = np.array(p_mass_correct_count_list) / np.array(p_mass_count_list) if len(p_mass_count_list) > 0 else 0
    
        print(f"[PID] Pi Efficiency: {pi_eff_btof}")
        print(f"[PID] K Efficiency: {k_eff_btof}")
        print(f"[PID] P Efficiency: {p_eff_btof}")

        valid_mask_pi = (~np.isnan(pi_eff_btof)) & np.isfinite(pi_eff_btof) & (pi_eff_btof != 0)
        valid_mask_k = (~np.isnan(k_eff_btof)) & np.isfinite(k_eff_btof) & (k_eff_btof != 0)
        valid_mask_p = (~np.isnan(p_eff_btof)) & np.isfinite(p_eff_btof) & (p_eff_btof != 0)

        valid_pi_eff_btof = pi_eff_btof[valid_mask_pi]
        valid_k_eff_btof = k_eff_btof[valid_mask_k]
        valid_p_eff_btof = p_eff_btof[valid_mask_p]

        pi_eff_err_btof = np.sqrt(valid_pi_eff_btof * (1 - valid_pi_eff_btof) / np.array(pi_mass_count_list)[valid_mask_pi])
        k_eff_err_btof  = np.sqrt(valid_k_eff_btof * (1 - valid_k_eff_btof) / np.array(k_mass_count_list)[valid_mask_k])
        p_eff_err_btof  = np.sqrt(valid_p_eff_btof * (1 - valid_p_eff_btof) / np.array(p_mass_count_list)[valid_mask_p])

        valid_bin_centers_pi = bin_centers[valid_mask_pi]
        valid_bin_centers_k = bin_centers[valid_mask_k]
        valid_bin_centers_p = bin_centers[valid_mask_p]

        # Pi Efficiency Graph with Error Bars
        gr_pi = r.TGraphErrors()
        gr_pi.SetName("pi_eff_vs_mom")
        gr_pi.SetTitle("Pi Efficiency vs Momentum;p [GeV];Efficiency")
        gr_pi.GetXaxis().SetRangeUser(0, 3.5)  
        gr_pi.GetYaxis().SetRangeUser(0, 1)

        idx = 0
        for bc, eff, err in zip(valid_bin_centers_pi, valid_pi_eff_btof, pi_eff_err_btof):
            gr_pi.SetPoint(idx, bc, eff)
            gr_pi.SetPointError(idx, 0, err)  
            idx += 1

        gr_pi.SetMarkerStyle(20)
        gr_pi.SetMarkerColor(r.kRed)
        gr_pi.SetMarkerSize(1.3)
        gr_pi.SetLineWidth(2)  

        # K Efficiency Graph with Error Bars
        gr_k = r.TGraphErrors()
        gr_k.SetName("k_eff_vs_mom")
        gr_k.SetTitle("K Efficiency vs Momentum;p [GeV];Efficiency")
        gr_k.GetXaxis().SetRangeUser(0, 3.5)  
        gr_k.GetYaxis().SetRangeUser(0, 1)

        idx = 0
        for bc, eff, err in zip(valid_bin_centers_k, valid_k_eff_btof, k_eff_err_btof):
            gr_k.SetPoint(idx, bc, eff)
            gr_k.SetPointError(idx, 0, err)  
            idx += 1

        gr_k.SetMarkerStyle(20)
        gr_k.SetMarkerColor(r.kBlue)
        gr_k.SetMarkerSize(1.3)
        gr_k.SetLineWidth(2)

        # P Efficiency Graph with Error Bars
        gr_p = r.TGraphErrors()
        gr_p.SetName("p_eff_vs_mom")
        gr_p.SetTitle("P Efficiency vs Momentum;p [GeV];Efficiency")
        gr_p.GetXaxis().SetRangeUser(0, 3.5)  
        gr_p.GetYaxis().SetRangeUser(0, 1)

        idx = 0
        for bc, eff, err in zip(valid_bin_centers_p, valid_p_eff_btof, p_eff_err_btof):
            gr_p.SetPoint(idx, bc, eff)
            gr_p.SetPointError(idx, 0, err)  
            idx += 1

        gr_p.SetMarkerStyle(20)
        gr_p.SetMarkerColor(r.kGreen)
        gr_p.SetMarkerSize(1.3)
        gr_p.SetLineWidth(2)

        if rootfile:
            gr_pi.Write()
            gr_k.Write()
            gr_p.Write()

        # Pi Efficiency Canvas
        c1 = r.TCanvas("c1", "Pi Efficiency", 800, 600)
        c1.SetLogy()
        c1.SetLogx(False)  

        frame = c1.DrawFrame(0, 0, 3.5, 1)  
        frame.GetXaxis().SetTitle("p [GeV]")
        frame.GetYaxis().SetTitle("Efficiency")
        frame.GetXaxis().SetRangeUser(0, 3.5)

        gr_pi.Draw("AP SAME")
        c1.Update()

        # K Efficiency Canvas
        c2 = r.TCanvas("c2", "K Efficiency", 800, 600)
        c2.SetLogy()
        c2.SetLogx(False)  

        frame2 = c2.DrawFrame(0, 0, 3.5, 1)  
        frame2.GetXaxis().SetTitle("p [GeV]")
        frame2.GetYaxis().SetTitle("Efficiency")
        frame2.GetXaxis().SetRangeUser(0, 3.5)

        gr_k.Draw("AP SAME")        
        c2.Update()

        # P Efficiency Canvas
        c3 = r.TCanvas("c3", "P Efficiency", 800, 600)
        c3.SetLogy()
        c3.SetLogx(False)  

        frame3 = c3.DrawFrame(0, 0, 3.5, 1)  
        frame3.GetXaxis().SetTitle("p [GeV]")
        frame3.GetYaxis().SetTitle("Efficiency")
        frame3.GetXaxis().SetRangeUser(0, 3.5)

        gr_p.Draw("AP SAME")
        c3.Update()

        if rootfile:
            c1.Write("canvas_pi_eff_logy")
            c2.Write("canvas_k_eff_logy")
            c3.Write("canvas_p_eff_logy")

    def plot_pid_performance_vs_momentum_with_TEfficiency(
        self,
        btof_calc_mass: np.ndarray,
        btof_pdg: np.ndarray,
        track_momentums_on_btof: np.ndarray,
        track_momentums_transverse_on_btof: np.ndarray,
        name: str = "test",
        nbins: int = 35,
        momentum_range: tuple = (0, 3.5),
        MERGIN_PI: float = 100,
        MERGIN_K: float = 100,
        MERGIN_P: float = 100,
        rootfile=None
    ):
        """
        Clopper-Pearson (small sample size error bar) is used for TEfficiency.

        Parameters
        ----------
        btof_calc_mass : ndarray
            reconstructed mass from bTOF (MeV)
        btof_pdg       : ndarray
            truth PDG code of the particle
        track_momentums_on_btof : ndarray
            reconstructed track momentum on bTOF (GeV)
        MERGIN_?       : float
            mass margin for PID (MeV)
        nbins, momentum_range
            binning and momentum range for TEfficiency
        rootfile       : ROOT.TFile or None
            output ROOT file
        """

        PI_MASS     = 139.57039
        KAON_MASS   = 493.677
        PROTON_MASS = 938.272

        eff_pi = r.TEfficiency("eff_pi", "Pi Efficiency; p [GeV]; Efficiency", nbins, momentum_range[0], momentum_range[1])
        eff_k  = r.TEfficiency("eff_k",  "K  Efficiency; p [GeV]; Efficiency", nbins, momentum_range[0], momentum_range[1])
        eff_p  = r.TEfficiency("eff_p",  "P  Efficiency; p [GeV]; Efficiency", nbins, momentum_range[0], momentum_range[1])

        # Clopper-Pearson (small sample size error bar)
        eff_pi.SetStatisticOption(r.TEfficiency.kFCP)
        eff_k.SetStatisticOption(r.TEfficiency.kFCP)
        eff_p.SetStatisticOption(r.TEfficiency.kFCP)

        n_events = len(btof_calc_mass)
        for i in range(n_events):
            mass = btof_calc_mass[i]
            pdg  = btof_pdg[i]
            mom  = track_momentums_on_btof[i]

            if mom < momentum_range[0] or mom >= momentum_range[1]:
                continue

            if pdg == 211 or pdg == -211:
                pass_flag = (abs(mass - PI_MASS) < MERGIN_PI)
                eff_pi.Fill(pass_flag, mom)

            elif pdg == 321 or pdg == -321:
                pass_flag = (abs(mass - KAON_MASS) < MERGIN_K)
                eff_k.Fill(pass_flag, mom)

            elif pdg == 2212 or pdg == -2212:
                pass_flag = (abs(mass - PROTON_MASS) < MERGIN_P)
                eff_p.Fill(pass_flag, mom)

        gr_pi = eff_pi.CreateGraph()
        gr_k  = eff_k.CreateGraph()
        gr_p  = eff_p.CreateGraph()

        gr_pi.SetName("pi_eff_vs_mom")
        gr_pi.SetTitle("Pi Efficiency vs Momentum;p [GeV];Efficiency")
        gr_pi.SetMarkerStyle(20)
        gr_pi.SetMarkerColor(r.kRed)
        gr_pi.SetLineColor(r.kRed)

        gr_k.SetName("k_eff_vs_mom")
        gr_k.SetTitle("K Efficiency vs Momentum;p [GeV];Efficiency")
        gr_k.SetMarkerStyle(21)
        gr_k.SetMarkerColor(r.kBlue)
        gr_k.SetLineColor(r.kBlue)

        gr_p.SetName("p_eff_vs_mom")
        gr_p.SetTitle("P Efficiency vs Momentum;p [GeV];Efficiency")
        gr_p.SetMarkerStyle(22)
        gr_p.SetMarkerColor(r.kGreen+2)
        gr_p.SetLineColor(r.kGreen+2)

        c_pi = r.TCanvas("c_pi", "Pi Efficiency", 800, 600)
        c_pi.SetLogy()
        gr_pi.Draw("AP")  
        c_pi.Update()

        c_k = r.TCanvas("c_k", "K Efficiency", 800, 600)
        c_k.SetLogy()
        gr_k.Draw("AP")
        c_k.Update()

        c_p = r.TCanvas("c_p", "P Efficiency", 800, 600)
        c_p.SetLogy()
        gr_p.Draw("AP")
        c_p.Update()

        if rootfile:
            eff_pi.Write()  
            eff_k.Write()
            eff_p.Write()

            gr_pi.Write()
            gr_k.Write()
            gr_p.Write()

            c_pi.Write("canvas_pi_eff_logy")
            c_k.Write("canvas_k_eff_logy")
            c_p.Write("canvas_p_eff_logy")

        print("[PID] TEfficiency analysis complete.")

            
    def pid_performance_include_t0_effect(self, 
                                          name: str, 
                                          btof_and_track_matched: dict, 
                                          ectof_and_track_matched: dict, 
                                          rootfile: uproot.TTree ,
                                          t0_mean: float = 0, 
                                          t0_sigma_values: List[float] = [0e-12, 15e-12, 30e-12, 45e-12],
                                          SELECTED_EVENTS: int = 1000, 
                                          MERGIN_PI: float = 100, 
                                          MERGIN_K: float = 100, 
                                          MERGIN_P: float = 100, 
                                          LARGE_MERGIN_PI: float = 200, 
                                          LARGE_MERGIN_K: float = 200, 
                                          LARGE_MERGIN_P: float = 200, 
                                          MOMENTUM_RANGE: float = 2,
                                          output_txt_name: str = 'pid_result.txt'
                                          ):
        """
        Calculates and returns PID performance metrics including T0 effect.

        Args:
            matched_tracks_and_tof_phi_theta (dict): Matched tracks and TOF information.

        Returns:
            Tuple: BTOF and ETOF metrics (momentums, beta inverses, and calculated masses).
        """

        btof_time = btof_and_track_matched['tof_time']
        btof_phi = btof_and_track_matched['tof_pos_phi']
        btof_theta = btof_and_track_matched['tof_pos_theta']
        track_momentum_on_btof = btof_and_track_matched['track_p']
        btof_pdg = btof_and_track_matched['mc_pdg']
        btof_pathlength = btof_and_track_matched['track_pathlength']

        ectof_time = ectof_and_track_matched['tof_time']
        ectof_phi = ectof_and_track_matched['tof_pos_phi']
        ectof_theta = ectof_and_track_matched['tof_pos_theta']
        track_momentum_on_ectof = ectof_and_track_matched['track_p']
        ectof_pdg = ectof_and_track_matched['mc_pdg']
        ectof_pathlength = ectof_and_track_matched['track_pathlength']

        t0_timing_dict = {}  
        btof_t0_timing_dict = {}
        ectof_t0_timing_dict = {}

        for t0_sigma in t0_sigma_values:
            t0_timing_dict[t0_sigma] = []
            btof_t0_timing_dict[t0_sigma] = []
            ectof_t0_timing_dict[t0_sigma] = []
            
            for i in range(len(btof_time)):
                t0 = np.random.normal(t0_mean, t0_sigma)  
                t0_ps = t0 * 1e12  
                t0_ns = t0 * 1e9  
                t0_timing_dict[t0_sigma].append(t0_ps)
                btof_t0_timing_dict[t0_sigma].append(btof_time[i] - t0_ns)  

            for i in range(len(ectof_time)):
                t0 = np.random.normal(t0_mean, t0_sigma)
                t0_ps = t0 * 1e12
                t0_ns = t0 * 1e9
                t0_timing_dict[t0_sigma].append(t0_ps)
                ectof_t0_timing_dict[t0_sigma].append(ectof_time[i] - t0_ns)

        btof_beta_inversees_dict = {}
        btof_calc_mass_dict = {}
        track_momentums_on_btof_dict = {}

        etof_beta_inversees_dict = {}
        etof_calc_mass_dict = {}
        track_momentums_on_ectof_dict = {}

        for t0_sigma in t0_sigma_values:
            btof_beta_inversees_dict[t0_sigma] = []
            btof_calc_mass_dict[t0_sigma] = []
            track_momentums_on_btof_dict[t0_sigma] = []

            etof_beta_inversees_dict[t0_sigma] = []
            etof_calc_mass_dict[t0_sigma] = []
            track_momentums_on_ectof_dict[t0_sigma] = []

            for i in range(len(btof_t0_timing_dict[t0_sigma])):  
                current_time = btof_t0_timing_dict[t0_sigma][i]
                btof_beta = btof_pathlength[i] / current_time
                btof_beta_c = btof_beta / 299.792458

                btof_beta_inverse = 1 / btof_beta_c
                calc_mass = 1000 * track_momentum_on_btof[i] * np.sqrt(1 - btof_beta_c**2) / btof_beta_c
                btof_beta_inversees_dict[t0_sigma].append(btof_beta_inverse)
                btof_calc_mass_dict[t0_sigma].append(calc_mass)
                track_momentums_on_btof_dict[t0_sigma].append(track_momentum_on_btof[i])

            for i in range(len(ectof_t0_timing_dict[t0_sigma])):  
                current_time = ectof_t0_timing_dict[t0_sigma][i]
                etof_beta = ectof_pathlength[i] / current_time
                etof_beta_c = etof_beta / 299.792458  
                etof_beta_inverse = 1 / etof_beta_c
                calc_mass = 1000 * track_momentum_on_ectof[i] * np.sqrt(1 - etof_beta_c**2) / etof_beta_c
                etof_beta_inversees_dict[t0_sigma].append(etof_beta_inverse)
                etof_calc_mass_dict[t0_sigma].append(calc_mass)
                track_momentums_on_ectof_dict[t0_sigma].append(track_momentum_on_ectof[i])

            myfunc.make_histogram_root(
                btof_t0_timing_dict[t0_sigma],
                100,
                hist_range=[0, 10],
                title=f'BTOF T0 Timing (t0_sigma={t0_sigma*1e12:.0f} ps)',
                xlabel='Time [ns]',
                ylabel='Entries',
                outputname=f'{name}/btof_t0_timing_pid_performance_include_t0_sigma_{t0_sigma*1e12:.0f}ps',
                rootfile=rootfile
            )

            myfunc.make_histogram_root(
                track_momentums_on_btof_dict[t0_sigma],
                100,
                hist_range=[0, 5],
                title=f'BTOF Momentum (t0_sigma={t0_sigma*1e12:.0f} ps)',
                xlabel='Momentum [GeV]',
                ylabel='Entries',
                outputname=f'{name}/btof_momentum_pid_performance_include_t0_sigma_{t0_sigma*1e12:.0f}ps',
                rootfile=rootfile
            )

            myfunc.make_histogram_root(
                btof_beta_inversees_dict[t0_sigma],
                100,
                hist_range=[0.8, 1.8],
                title=f'BTOF Beta Inverse (t0_sigma={t0_sigma*1e12:.0f} ps)',
                xlabel='Beta Inverse',
                ylabel='Entries',
                outputname=f'{name}/btof_beta_inverse_pid_performance_include_t0_sigma_{t0_sigma*1e12:.0f}ps',
                rootfile=rootfile
            )

            myfunc.make_histogram_root(
                btof_calc_mass_dict[t0_sigma],
                100,
                hist_range=[0, 1000],
                title=f'BTOF Calculated Mass (t0_sigma={t0_sigma*1e12:.0f} ps)',
                xlabel='Mass [MeV]',
                ylabel='Entries',
                outputname=f'{name}/btof_mass_pid_performance_include_t0_sigma_{t0_sigma*1e12:.0f}ps',
                rootfile=rootfile
            )

            m_pi = 139 # MeV
            m_k = 493 # MeV
            m_p = 938 # MeV

            pi_calc_mass_on_btof = []
            k_calc_mass_on_btof = []
            p_calc_mass_on_btof = []
            pi_mass_count_btof = 0
            pi_mass_count_btof_large_mergin = 0
            pi_mass_count_btof_low_momentum = 0
            k_mass_count_btof = 0
            k_mass_count_btof_large_mergin = 0
            k_mass_count_btof_low_momentum = 0
            p_mass_count_btof = 0
            p_mass_count_btof_large_mergin = 0
            p_mass_count_btof_low_momentum = 0

            pi_low_momentum_btof = []
            k_low_momentum_btof = []
            p_low_momentum_btof = []

            for i in range(len(btof_calc_mass_dict[t0_sigma])):
                if track_momentums_on_btof_dict[t0_sigma][i] < MOMENTUM_RANGE:
                    if btof_pdg[i] == 211 or btof_pdg[i] == -211:
                        pi_low_momentum_btof.append(btof_calc_mass_dict[t0_sigma][i])

                    if btof_pdg[i] == 321 or btof_pdg[i] == -321:
                        k_low_momentum_btof.append(btof_calc_mass_dict[t0_sigma][i])

                    if btof_pdg[i] == 2212 or btof_pdg[i] == -2212:
                        p_low_momentum_btof.append(btof_calc_mass_dict[t0_sigma][i])

            for i in range(len(btof_calc_mass_dict[t0_sigma])):
                if btof_pdg[i] == 211 or btof_pdg[i] == -211:
                    pi_calc_mass_on_btof.append(btof_calc_mass_dict[t0_sigma][i])
                    if -MERGIN_PI < btof_calc_mass_dict[t0_sigma][i] - m_pi < MERGIN_PI:
                        pi_mass_count_btof += 1
                    if -m_pi < btof_calc_mass_dict[t0_sigma][i] - m_pi < m_pi:
                        pi_mass_count_btof_large_mergin += 1
                    if track_momentums_on_btof_dict[t0_sigma][i] < MOMENTUM_RANGE:
                        if -MERGIN_PI < btof_calc_mass_dict[t0_sigma][i] - m_pi < MERGIN_PI:
                            pi_mass_count_btof_low_momentum += 1
                if btof_pdg[i] == 321 or btof_pdg[i] == -321:
                    k_calc_mass_on_btof.append(btof_calc_mass_dict[t0_sigma][i])
                    if -MERGIN_K < btof_calc_mass_dict[t0_sigma][i] - m_k < MERGIN_K:
                        k_mass_count_btof += 1
                    if -LARGE_MERGIN_K < btof_calc_mass_dict[t0_sigma][i] - m_k < LARGE_MERGIN_K:
                        k_mass_count_btof_large_mergin += 1
                    if track_momentums_on_btof_dict[t0_sigma][i] < MOMENTUM_RANGE:
                        if -MERGIN_K < btof_calc_mass_dict[t0_sigma][i] - m_k < MERGIN_K:
                            k_mass_count_btof_low_momentum += 1
                if btof_pdg[i] == 2212 or btof_pdg[i] == -2212:
                    p_calc_mass_on_btof.append(btof_calc_mass_dict[t0_sigma][i])
                    if -MERGIN_P < btof_calc_mass_dict[t0_sigma][i] - m_p < MERGIN_P:
                        p_mass_count_btof += 1
                    if -LARGE_MERGIN_P < btof_calc_mass_dict[t0_sigma][i] - m_p < LARGE_MERGIN_P:
                        p_mass_count_btof_large_mergin += 1
                    if track_momentums_on_btof_dict[t0_sigma][i] < MOMENTUM_RANGE:
                        if -MERGIN_P < btof_calc_mass_dict[t0_sigma][i] - m_p < MERGIN_P:
                            p_mass_count_btof_low_momentum += 1

            pi_eff_btof = pi_mass_count_btof / len(pi_calc_mass_on_btof) if len(pi_calc_mass_on_btof) > 0 else 0
            pi_eff_btof_low_momentum = pi_mass_count_btof_low_momentum / len(pi_low_momentum_btof) if len(pi_low_momentum_btof) > 0 else 0
            pi_eff_btof_large_mergin = pi_mass_count_btof_large_mergin / len(pi_calc_mass_on_btof) if len(pi_calc_mass_on_btof) > 0 else 0
            k_eff_btof = k_mass_count_btof / len(k_calc_mass_on_btof) if len(k_calc_mass_on_btof) > 0 else 0
            k_eff_btof_low_momentum = k_mass_count_btof_low_momentum / len(k_low_momentum_btof) if len(k_low_momentum_btof) > 0 else 0
            k_eff_btof_large_mergin = k_mass_count_btof_large_mergin / len(k_calc_mass_on_btof) if len(k_calc_mass_on_btof) > 0 else 0
            p_eff_btof = p_mass_count_btof / len(p_calc_mass_on_btof) if len(p_calc_mass_on_btof) > 0 else 0
            p_eff_btof_low_momentum = p_mass_count_btof_low_momentum / len(p_low_momentum_btof) if len(p_low_momentum_btof) > 0 else 0
            p_eff_btof_large_mergin = p_mass_count_btof_large_mergin / len(p_calc_mass_on_btof) if len(p_calc_mass_on_btof) > 0 else 0

            with open(output_txt_name, 'a') as f:
                f.write(f'BTOF PID Performance_include_t0_effect_{t0_sigma*1e12:.0f}ps\n')
                f.write(f'Pi Efficiency: {pi_eff_btof}\n')
                f.write(f'K Efficiency: {k_eff_btof}\n')
                f.write(f'P Efficiency: {p_eff_btof}\n')

            myfunc.make_histogram_root(
                pi_calc_mass_on_btof,
                            100,
                            hist_range=[0, 1000],
                            title='BTOF Calculated Mass for Pi_include_t0',
                            xlabel='Mass [GeV]',
                            ylabel='Entries',
                            outputname=f'{name}/btof_mass_pi_pid_performance_include_t0',
                            rootfile=rootfile
            )
            
            myfunc.make_histogram_root(
                k_calc_mass_on_btof,
                            100,
                            hist_range=[0, 1000],
                            title='BTOF Calculated Mass for K_include_t0',
                            xlabel='Mass [MeV]',
                            ylabel='Entries',
                            outputname=f'{name}/btof_mass_k_pid_performance_include_t0',
                            rootfile=rootfile
            )
            
            myfunc.make_histogram_root(
                p_calc_mass_on_btof,
                            100,
                            hist_range=[200, 1200],
                            title='BTOF Calculated Mass for P_include_t0',
                            xlabel='Mass [MeV]',
                            ylabel='Entries',
                            outputname=f'{name}/btof_mass_p_pid_performance_include_t0',
                            rootfile=rootfile
            )

            myfunc.make_2Dhistogram_root(
                track_momentums_on_btof_dict[t0_sigma],
                100,
                [0, 5],
                btof_beta_inversees_dict[t0_sigma],
                100,
                [0.8, 1.8],
                title='BTOF Momentum vs Beta Inverse_include_t0',
                xlabel='Momentum [GeV]',
                ylabel='Beta Inverse',
                outputname=f'{name}/btof_momentum_vs_beta_inverse_pid_performance_include_t0',
                cmap='plasma',
                logscale=True,
                rootfile=rootfile
            )

        return btof_t0_timing_dict, track_momentums_on_btof_dict, btof_beta_inversees_dict, btof_calc_mass_dict

    def plot_pid_performance_include_tof_time_reso_and_t0_effect(self, 
                                                                 name: str, 
                                                                 btof_and_track_matched: dict, 
                                                                 ectof_and_track_matched: dict, 
                                                                 t0_sigma: float,
                                                                 btof_t0_timing: np.ndarray,
                                                                 ectof_t0_timing, 
                                                                 rootfile: uproot.TTree, 
                                                                 SELECTED_EVENTS: int = 1000, 
                                                                 MERGIN_PI: float = 100, 
                                                                 MERGIN_K: float = 100, 
                                                                 MERGIN_P: float = 100, 
                                                                 LARGE_MERGIN_PI: float = 200, 
                                                                 LARGE_MERGIN_K: float = 200, 
                                                                 LARGE_MERGIN_P: float = 200, 
                                                                 MOMENTUM_RANGE: float = 2, 
                                                                 btof_time_reso: float = 50e-12,
                                                                 ectof_time_reso: float = 50e-12,
                                                                 output_txt_name: str = 'pid_result.txt'
                                                                 ):
        """
        Calculates and returns PID performance metrics including T0 effect and TOF time resolution effect.

        Args:
            matched_tracks_and_tof_phi_theta (dict): Matched tracks and TOF information.

        Returns:
            Tuple: BTOF and ETOF metrics (momentums, beta inverses, and calculated masses).
        """

        btof_t0_timing = btof_t0_timing
        btof_time = btof_and_track_matched['tof_time']
        ectof_t0_timing = ectof_t0_timing
        ectof_time = ectof_and_track_matched['tof_time']

        track_momentum_on_btof = btof_and_track_matched['track_p']
        btof_pdg = btof_and_track_matched['mc_pdg']
        btof_pathlength = btof_and_track_matched['track_pathlength']

        track_momentum_on_ectof = ectof_and_track_matched['track_p']
        ectof_pdg = ectof_and_track_matched['mc_pdg']
        ectof_pathlength = ectof_and_track_matched['track_pathlength']

        btof_t0_timing = []
        t0_timing = []


        for i in range(len(btof_time)):
            t0 = np.random.normal(0, t0_sigma)  
            t0_ps = t0 * 1e12  
            t0_ns = t0 * 1e9  
            btof_t0_timing.append(btof_time[i] - t0_ns)
            t0_timing.append(t0_ps)

        for i in range(len(ectof_time)):
            t0 = np.random.normal(0, t0_sigma)
            t0_ps = t0 * 1e12
            t0_ns = t0 * 1e9
            ectof_t0_timing.append(ectof_time[i] - t0_ns)
            t0_timing.append(t0_ps)

        btof_time_include_resos = []
        ectof_time_include_resos = []

        time_resos_btof = []
        time_resos_ectof = []

        for i in range(len(btof_t0_timing)):
            time_reso_btof = np.random.normal(0, btof_time_reso)
            time_reso_ns_btof = time_reso_btof * 1e9  
            btof_time_value = btof_t0_timing[i].item()  
            btof_time_include_reso = btof_time_value + time_reso_ns_btof  
            time_resos_btof.append(time_reso_btof * 1e12) 
            btof_time_include_resos.append(btof_time_include_reso)

        for i in range(len(ectof_t0_timing)):
            time_reso_etof = np.random.normal(0, ectof_time_reso)
            time_reso_ns_etof = time_reso_etof * 1e9  
            ectof_time_value = ectof_t0_timing[i].item()  
            ectof_time_value = float(ectof_t0_timing[i])
            ectof_time_include_reso = ectof_time_value + time_reso_ns_etof
            time_resos_ectof.append(time_reso_etof * 1e12)
            ectof_time_include_resos.append(ectof_time_include_reso)

        myfunc.make_histogram_root(
            btof_time_include_resos,
            100,
            hist_range=[0, 10],
            title='BTOF T0 Timing_include_reso',
            xlabel='Time [ns]',
            ylabel='Entries',
            outputname=f'{name}/btof_t0_timing_pid_performance_include_reso',
            rootfile=rootfile
        )

        myfunc.make_histogram_root(
            time_resos_btof,
            100,
            hist_range=[-100, 100],
            title='BTOF Time Resolution',
            xlabel='Time [ps]',
            ylabel='Entries',
            outputname=f'{name}/btof_time_reso_pid_performance_include_reso',
            rootfile=rootfile
        )

        myfunc.make_stacked_histogram_root(
            [btof_time_include_resos, btof_t0_timing, btof_time],
            100,
            hist_range=[0, 10],
            title='BTOF T0 Timing vs TOF Timing_include_reso vs TOF Timing',
            xlabel='Time [ns]',
            ylabel='Entries',
            labels=['T0 Timing', 'T0 Timing_include_reso', 'Not T0 Timing'],
            outputname=f'{name}/btof_t0_timing_vs_btof_not_t0_timing_vs_btof_t0_timing_include_reso',
            rootfile=rootfile
        )

        track_momentums_on_btof = []
        track_momentums_on_ectof = []
        btof_beta_inversees = []
        etof_beta_inversees = []
        btof_calc_mass = []
        etof_calc_mass = []

        for i in range(len(btof_time_include_resos)):
            current_time = btof_time_include_resos[i]
            btof_beta = btof_pathlength[i] / current_time
            btof_beta_c = btof_beta / 299.792458

            btof_beta_inverse = 1 / btof_beta_c
            calc_mass = 1000 * track_momentum_on_btof[i] * np.sqrt(1 - btof_beta_c**2) / btof_beta_c
            btof_beta_inversees.append(btof_beta_inverse)
            btof_calc_mass.append(calc_mass)
            track_momentums_on_btof.append(track_momentum_on_btof[i])

        for i in range(len(ectof_time_include_resos)):
            current_time = ectof_time_include_resos[i]
            etof_beta = ectof_pathlength[i] / current_time
            etof_beta_c = etof_beta / 299.792458

            etof_beta_inverse = 1 / etof_beta_c
            calc_mass = 1000 * track_momentum_on_ectof[i] * np.sqrt(1 - etof_beta_c**2) / etof_beta_c
            etof_beta_inversees.append(etof_beta_inverse)
            etof_calc_mass.append(calc_mass)
            track_momentums_on_ectof.append(track_momentum_on_ectof[i])

        myfunc.make_histogram_root(
            track_momentums_on_btof,
                           100,
                           hist_range=[0, 5],
                        title='BTOF Momentum_include_reso',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_momentum_pid_performance_include_reso',
                        rootfile=rootfile
        )
        
        myfunc.make_histogram_root(
            btof_beta_inversees,
                        100,
                        hist_range=[0.8, 1.8],
                        title='BTOF Beta Inverse_include_reso',
                        xlabel='Beta Inverse',
                        ylabel='Entries',
                        outputname=f'{name}/btof_beta_inverse_pid_performance_include_reso',
                        rootfile=rootfile
        )
        
        myfunc.make_histogram_root(
            btof_calc_mass,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF Calculated Mass_include_reso',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_pid_performance_include_reso',
                        rootfile=rootfile
        )
        
        m_pi = 139 # MeV
        m_k = 493 # MeV
        m_p = 938 # MeV

        pi_calc_mass_on_btof = []
        k_calc_mass_on_btof = []
        p_calc_mass_on_btof = []
        pi_mass_count_btof = 0
        pi_mass_count_btof_large_mergin = 0
        pi_mass_count_btof_low_momentum = 0
        k_mass_count_btof = 0
        k_mass_count_btof_large_mergin = 0
        k_mass_count_btof_low_momentum = 0
        p_mass_count_btof = 0
        p_mass_count_btof_large_mergin = 0
        p_mass_count_btof_low_momentum = 0

        pi_low_momentum_btof = []
        k_low_momentum_btof = []
        p_low_momentum_btof = []

        for i in range(len(btof_calc_mass)):
            if track_momentums_on_btof[i] < MOMENTUM_RANGE:
                if btof_pdg[i] == 211 or btof_pdg[i] == -211:
                    pi_low_momentum_btof.append(btof_calc_mass[i])

                if btof_pdg[i] == 321 or btof_pdg[i] == -321:
                    k_low_momentum_btof.append(btof_calc_mass[i])

                if btof_pdg[i] == 2212 or btof_pdg[i] == -2212:
                    p_low_momentum_btof.append(btof_calc_mass[i])

        for i in range(len(btof_calc_mass)):
            if btof_pdg[i] == 211 or btof_pdg[i] == -211:
                pi_calc_mass_on_btof.append(btof_calc_mass[i])
                if -MERGIN_PI < btof_calc_mass[i] - m_pi < MERGIN_PI:
                    pi_mass_count_btof += 1
                if -m_pi < btof_calc_mass[i] - m_pi < LARGE_MERGIN_PI:
                    pi_mass_count_btof_large_mergin += 1
                if track_momentums_on_btof[i] < MOMENTUM_RANGE:
                    if -MERGIN_PI < btof_calc_mass[i] - m_pi < MERGIN_PI:
                        pi_mass_count_btof_low_momentum += 1
            if btof_pdg[i] == 321 or btof_pdg[i] == -321:
                k_calc_mass_on_btof.append(btof_calc_mass[i])
                if -MERGIN_K < btof_calc_mass[i] - m_k < MERGIN_K:
                    k_mass_count_btof += 1
                if -LARGE_MERGIN_K < btof_calc_mass[i] - m_k < LARGE_MERGIN_K:
                    k_mass_count_btof_large_mergin += 1
                if track_momentums_on_btof[i] < MOMENTUM_RANGE:
                    if -MERGIN_K < btof_calc_mass[i] - m_k < MERGIN_K:
                        k_mass_count_btof_low_momentum += 1
            if btof_pdg[i] == 2212 or btof_pdg[i] == -2212:
                p_calc_mass_on_btof.append(btof_calc_mass[i])
                if -MERGIN_P < btof_calc_mass[i] - m_p < MERGIN_P:
                    p_mass_count_btof += 1
                if -LARGE_MERGIN_P < btof_calc_mass[i] - m_p < LARGE_MERGIN_P:
                    p_mass_count_btof_large_mergin += 1
                if track_momentums_on_btof[i] < MOMENTUM_RANGE:
                    if -MERGIN_P < btof_calc_mass[i] - m_p < MERGIN_P:
                        p_mass_count_btof_low_momentum += 1

        pi_eff_btof = pi_mass_count_btof / len(pi_calc_mass_on_btof) if len(pi_calc_mass_on_btof) > 0 else 0
        pi_eff_btof_low_momentum = pi_mass_count_btof_low_momentum / len(pi_low_momentum_btof) if len(pi_low_momentum_btof) > 0 else 0
        pi_eff_btof_large_mergin = pi_mass_count_btof_large_mergin / len(pi_calc_mass_on_btof) if len(pi_calc_mass_on_btof) > 0 else 0
        k_eff_btof = k_mass_count_btof / len(k_calc_mass_on_btof) if len(k_calc_mass_on_btof) > 0 else 0
        k_eff_btof_low_momentum = k_mass_count_btof_low_momentum / len(k_low_momentum_btof) if len(k_low_momentum_btof) > 0 else 0
        k_eff_btof_large_mergin = k_mass_count_btof_large_mergin / len(k_calc_mass_on_btof) if len(k_calc_mass_on_btof) > 0 else 0
        p_eff_btof = p_mass_count_btof / len(p_calc_mass_on_btof) if len(p_calc_mass_on_btof) > 0 else 0
        p_eff_btof_low_momentum = p_mass_count_btof_low_momentum / len(p_low_momentum_btof) if len(p_low_momentum_btof) > 0 else 0
        p_eff_btof_large_mergin = p_mass_count_btof_large_mergin / len(p_calc_mass_on_btof) if len(p_calc_mass_on_btof) > 0 else 0

        with open(output_txt_name, 'a') as f:
            f.write(f'BTOF PID Performance_include_tof_time_reso_and_t0_effect\n')
            f.write(f'Pi Efficiency: {pi_eff_btof}\n')
            f.write(f'K Efficiency: {k_eff_btof}\n')
            f.write(f'P Efficiency: {p_eff_btof}\n')
            f.write(f'BTOF Low Momentum PID Performance_include_tof_time_reso_and_t0_effect\n')
            f.write(f'Pi Efficiency: {pi_eff_btof_low_momentum}\n')
            f.write(f'K Efficiency: {k_eff_btof_low_momentum}\n')
            f.write(f'P Efficiency: {p_eff_btof_low_momentum}\n')
            f.write(f'BTOF Large Mergin PID Performance_include_tof_time_reso_and_t0_effect\n')
            f.write(f'Pi Efficiency: {pi_eff_btof_large_mergin}\n')
            f.write(f'K Efficiency: {k_eff_btof_large_mergin}\n')
            f.write(f'P Efficiency: {p_eff_btof_large_mergin}\n')

        myfunc.make_histogram_root(
            pi_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF Calculated Mass for Pi_include_tof_time_reso_and_t0',
                        xlabel='Mass [GeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_pi_pid_performance_include_tof_time_reso_and_t0',
                        rootfile=rootfile
        )
        
        myfunc.make_histogram_root(
            k_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF Calculated Mass for K_include_tof_time_reso_and_t0',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_k_pid_performance_include_tof_time_reso_and_t0',
                        rootfile=rootfile
        )
        
        myfunc.make_histogram_root(
            p_calc_mass_on_btof,
                        100,
                        hist_range=[200, 1200],
                        title='BTOF Calculated Mass for P_include_tof_time_reso_and_t0',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{name}/btof_mass_p_pid_performance_include_tof_time_reso_and_t0',
                        rootfile=rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_on_btof,
            100,
            [0, 5],
            btof_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF Momentum vs Beta Inverse_include_tof_time_reso_and_t0',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{name}/btof_momentum_vs_beta_inverse_pid_performance_include_tof_time_reso_and_t0',
            cmap='plasma',
            logscale=True,
            rootfile=rootfile
        )

def analyze_t0_and_tof_reso_effect(
    t0_mean_values, t0_sigma_values, tof_reso_values, btof_and_track_matched, ectof_and_track_matched, rootfile
):
    results = {}
    
    for sigma in t0_sigma_values:
        for tof_reso in tof_reso_values:
            output_name = f't0_mean_{mean*1e12:.0f}ps_t0_sigma_{sigma*1e12:.0f}ps_tof_reso_{tof_reso*1e12:.0f}ps'
            
            btof_t0_timing, ectof_t0_timing, track_momentums_on_btof, btof_beta_inversees, btof_calc_mass, etof_calc_mass = ToFPIDPerformance.plot_pid_performance_include_tof_time_reso_and_t0_effect(
                name=output_name,
                btof_and_track_matched=btof_and_track_matched,
                ectof_and_track_matched=ectof_and_track_matched,
                btof_t0_timing=t0_mean_values,
                ectof_t0_timing=t0_mean_values,
                rootfile=rootfile,
                t0_sigma=sigma,
                btof_time_reso=tof_reso,
                ectof_time_reso=tof_reso,
                output_txt_name=f'{output_name}.txt'
            )
        
            results[(sigma, tof_reso)] = {
                "btof_t0_timing": btof_t0_timing,
                "ectof_t0_timing": ectof_t0_timing,
                "track_momentums_on_btof": track_momentums_on_btof,
                "btof_beta_inversees": btof_beta_inversees,
                "btof_calc_mass": btof_calc_mass,
                "etof_calc_mass": etof_calc_mass
            }
    
    canvas = r.TCanvas("c", "BTOF T0 Timing and TOF Resolution Overlaid", 800, 600)
    legend = r.TLegend(0.7, 0.7, 0.9, 0.9)
    colors = [r.kRed, r.kBlue, r.kGreen, r.kBlack, r.kMagenta]
    
    hist_list = []
    for idx, ((mean, sigma, tof_reso), data) in enumerate(results.items()):
        hist = r.TH1D(f"hist_{idx}", f"BTOF T0 Timing (t0_sigma={sigma*1e12:.0f} ps, tof_reso={tof_reso*1e12:.0f} ps)", 100, 0, 10)
        for val in data["btof_t0_timing"]:
            hist.Fill(val)
        hist.SetLineColor(colors[idx % len(colors)])
        hist.SetLineWidth(2)
        hist_list.append(hist)
        legend.AddEntry(hist, f"t0_sigma={sigma*1e12:.0f} ps, tof_reso={tof_reso*1e12:.0f} ps", "l")
    
    for i, hist in enumerate(hist_list):
        if i == 0:
            hist.Draw("HIST")
        else:
            hist.Draw("HIST SAME")
    legend.Draw()
    canvas.Update()
    canvas.SaveAs("btof_t0_timing_tof_reso_overlayed.png")

def analyze_separation_vs_vertex_z(
                                name: str,
                                rootfile: uproot.TTree,
                                vertex_z_ranges: List[Tuple[float, float]],
                                output_txt_name: str = 'pid_result.txt',
                                output_efficiency_result_name: str = 'matching_result.txt'
                                ):
    """
    """

    config = load_yaml_config('./config/execute_config.yaml')
    branch = load_yaml_config('./config/branch_name.yaml')
    file_path = load_yaml_config('./config/file_path.yaml')

    name = config['name']
    VERBOSE = config['VERBOSE']
    PLOT_VERBOSE = config['PLOT_VERBOSE']
    SELECTED_EVENTS = config['SELECTED_EVENTS']        
    output_name = config['output_name']

    filename = file_path['FILE_PATH_SINGLE_PARTICLE_PION_OLDVERSION']['path']

    tree = load_tree_file(filename)

    track = Track(
        tree, 
        config, 
        branch, 
        name, 
        dis_file=tree
    )

    # Retrieve track segments positions and momenta
    track_segments_x, track_segments_y, track_segments_z, track_segments_d, track_segments_r = track.get_track_segments_pos(
        name=name,
        rootfile=rootfile,
        verbose=VERBOSE, 
        plot_verbose=PLOT_VERBOSE
    )
    track_segments_px, track_segments_py, track_segments_pz, track_segments_p, tracksegments_pt,track_segments_p_theta, track_segments_p_phi, track_segment_pathlength = track.get_track_segments_momentum(
        name=name,
        rootfile=rootfile,
        verbose=VERBOSE,
        plot_verbose=PLOT_VERBOSE
    )

    # Split track segments into individual tracks
    all_tracks = track.split_track_segments(
        x_positions=track_segments_x,
        y_positions=track_segments_y,
        z_positions=track_segments_z,
        px_momenta=track_segments_px,
        py_momenta=track_segments_py,
        pz_momenta=track_segments_pz,
        track_segment_pathlength=track_segment_pathlength,
        margin_theta=0.6,
        margin_phi=0.6,
        rootfile=rootfile,
        verbose=VERBOSE,
        plot_verbose=PLOT_VERBOSE,
        SELECTED_EVENTS=SELECTED_EVENTS
    )

    mc = MC(tree, config, branch, name, dis_file=tree)

    mc_px, mc_py, mc_pz, mc_p, mc_p_theta, mc_p_phi, mc_PDG_ID, mc_charge, mc_generatorStatus, mc_vertex_x, mc_vertex_y, mc_vertex_z = mc.get_mc_info(
    name=name,
    rootfile=rootfile,
    verbose=VERBOSE,
    plot_verbose=PLOT_VERBOSE
    )

    matching = MatchingMCAndTrack(
        track=track,
        mc=mc,
    )

    r_min_tracks, r_min_track_index = matching.get_segments_nearest_impact_point(
    all_tracks, 
    rootfile=rootfile,
    verbose=VERBOSE,
    plot_verbose=PLOT_VERBOSE
    )

    separation_results = []
    vertex_labels = []

    for (min_z, max_z) in vertex_z_ranges:
        print(f"Analyzing separation for vertex z range: ({min_z}, {max_z})")
        matched_pairs, min_delta_angles, matched_pairs_on_btof, matched_pairs_on_etof = matching.match_track_to_mc(
            name=name,
            track_momentum=track_segments_p,
            track_momentum_x=track_segments_px,
            track_momentum_y=track_segments_py,
            track_momentum_z=track_segments_pz,
            track_momentum_transverse=tracksegments_pt,
            track_momentum_theta=track_segments_p_theta,
            track_momentum_phi=track_segments_p_phi,
            track_pos_x=track_segments_x,
            track_pos_y=track_segments_y,
            track_pos_z=track_segments_z,
            track_pathlength=track_segment_pathlength,
            mc_momentum_theta=mc_p_theta,
            mc_momentum_phi=mc_p_phi,
            mc_momentum=mc_p,
            mc_pdg_ID=mc_PDG_ID,
            mc_generator_status=mc_generatorStatus,
            mc_charge=mc_charge,
            mc_vertex_x=mc_vertex_x,
            mc_vertex_y=mc_vertex_y,
            mc_vertex_z=mc_vertex_z,
            r_min_track_index=r_min_track_index,
            threshold=0.5,
            rootfile=rootfile,
            output_txt=output_efficiency_result_name,
            vertex_z_max=max_z,
            vertex_z_min=min_z,
            verbose=VERBOSE,
            plot_verbose=PLOT_VERBOSE
        )

        match_track_and_tof = MatchingTrackandToFHits( 
            dis_file=tree,
            branch=branch,
        )

        btof_phi_theta, ectof_phi_theta = match_track_and_tof.get_tof_info(
            name, 
            SELECTED_EVENTS, 
            rootfile=rootfile,
            verbose=VERBOSE,
            plot_verbose=PLOT_VERBOSE
        )

        # Match tracks to TOF hits
        matched_tracks_and_btof_phi_theta, matched_tracks_and_etof_phi_theta = match_track_and_tof.match_track_to_tof(
            name, 
            matched_pairs_on_btof, 
            matched_pairs_on_etof, 
            btof_phi_theta, 
            ectof_phi_theta, 
            rootfile=rootfile,
            output_txt=output_efficiency_result_name,
            verbose=VERBOSE,
            plot_verbose=PLOT_VERBOSE
        )

        pid = ToFPIDPerformance( 
            dis_file=tree,
            branch=branch
        )

        btof_calc_mass, btof_pdg, track_momentums_on_btof, track_momentums_transverse_on_btof = pid.plot_pid_performance(
            name, 
            matched_tracks_and_btof_phi_theta,
            matched_tracks_and_etof_phi_theta,
            rootfile=rootfile,
            output_txt_name=output_txt_name
        )

        bin_center, separation_power = pid.plot_separation_power_vs_momentum(
            btof_calc_mass,
            btof_pdg,
            track_momentums_on_btof,
            track_momentums_transverse_on_btof,
            name,
            rootfile=rootfile
        )

        pid.plot_pid_performance_vs_momentum(
            btof_calc_mass=btof_calc_mass,
            btof_pdg=btof_pdg,
            track_momentums_on_btof=track_momentums_on_btof,
            track_momentums_transverse_on_btof=track_momentums_transverse_on_btof,
            name=name,
            rootfile=rootfile
        )

        pid.plot_pid_performance_vs_momentum_with_TEfficiency(
            btof_calc_mass=btof_calc_mass,
            btof_pdg=btof_pdg,
            track_momentums_on_btof=track_momentums_on_btof,
            track_momentums_transverse_on_btof=track_momentums_transverse_on_btof,
            name=name,
            rootfile=rootfile
        )

        separation_results.append(separation_power)
        vertex_labels.append(f"({min_z}, {max_z})")

    for i, (separation_power, label) in enumerate(zip(separation_results, vertex_labels)):
        print(f"Separation power for vertex z range {label}: {separation_power}")

    canvas = r.TCanvas("c", "Separation Power vs Vertex Z Range", 800, 600)
    hist = r.TH1D("hist", "Separation Power vs Vertex Z Range", len(separation_results), 0, len(separation_results))

    c = r.TCanvas("Separation Power vs Momentum vertex comparison", "Separation Power vs Momentum", 800, 600)
    mg = r.TMultiGraph()

    colors = [r.kRed, r.kBlue, r.kGreen+2, r.kMagenta, r.kOrange+1]
    markers = [20, 21, 22]  # ● ◆ ■ ▲ ▼  

    for i, (sep_list, label) in enumerate(zip(separation_results, vertex_labels)):

        mom_arr = np.array(bin_center)
        sep_arr = np.array(sep_list)
        
        g = r.TGraph(len(mom_arr), mom_arr, sep_arr)

        #errobar is not working, so I will fix it later

        color = colors[i % len(colors)]     
        marker = markers[i % len(markers)] 
        g.SetLineColor(color)
        g.SetMarkerColor(color)
        g.SetMarkerStyle(marker)
        
        g.SetTitle(label)

        mg.Add(g, "P")  

    mg.SetTitle("Separation Power vs Momentum;Momentum (GeV/c);Separation Power")
    mg.Draw("A") 
    mg.GetYaxis().SetRangeUser(0, 30)
    mg.GetXaxis().SetLimits(0, 3.5)
    c.BuildLegend()
    c.SetLogy()

    c.Update()

    if rootfile:
        c.Write()
    

# Main execution
def main():
    # Load configurations
    config = load_yaml_config('./config/execute_config.yaml')
    branch = load_yaml_config('./config/branch_name.yaml')
    file_path = load_yaml_config('./config/file_path.yaml')

    name = config['name']
    VERBOSE = config['VERBOSE']
    PLOT_VERBOSE = config['PLOT_VERBOSE']
    SELECTED_EVENTS = config['SELECTED_EVENTS']        
    output_name = config['output_name']

    filename = file_path['FILE_PATH_SINGLE_PARTICLE_PION_OLDVERSION']['path']

    # Create directory for output ROOT file
    directory_name = f'./out/{name}'
    make_directory(directory_name)

    out_put_file_path = os.path.join(directory_name, output_name)
    output_txt_name = os.path.join(directory_name, 'pid_result.txt')
    output_efficiency_name = os.path.join(directory_name, 'matching_result.txt')
    
    rootfile = r.TFile(out_put_file_path, 'recreate')

    # Load ROOT tree
    tree = load_tree_file(filename)


    # Not yet capable of analyzing t0 or treso and vertex at the same time, will do so in the future 


    # # Initialize classes
    # track = Track(
    #     tree, 
    #     config, 
    #     branch, 
    #     name, 
    #     dis_file=tree
    # )

    # # Retrieve track segments positions and momenta
    # track_segments_x, track_segments_y, track_segments_z, track_segments_d, track_segments_r = track.get_track_segments_pos(
    #     name=name,
    #     rootfile=rootfile,
    #     verbose=VERBOSE, 
    #     plot_verbose=PLOT_VERBOSE
    # )
    # track_segments_px, track_segments_py, track_segments_pz, track_segments_p, tracksegments_pt,track_segments_p_theta, track_segments_p_phi, track_segment_pathlength = track.get_track_segments_momentum(
    #     name=name,
    #     rootfile=rootfile,
    #     verbose=VERBOSE,
    #     plot_verbose=PLOT_VERBOSE
    # )

    # # Split track segments into individual tracks
    # all_tracks = track.split_track_segments(
    #     x_positions=track_segments_x,
    #     y_positions=track_segments_y,
    #     z_positions=track_segments_z,
    #     px_momenta=track_segments_px,
    #     py_momenta=track_segments_py,
    #     pz_momenta=track_segments_pz,
    #     track_segment_pathlength=track_segment_pathlength,
    #     margin_theta=0.6,
    #     margin_phi=0.6,
    #     rootfile=rootfile,
    #     verbose=VERBOSE,
    #     plot_verbose=PLOT_VERBOSE,
    #     SELECTED_EVENTS=SELECTED_EVENTS
    # )

    # # Initialize MC class and retrieve MC information
    # mc = MC(
    #     tree, 
    #     config, 
    #     branch, 
    #     name, 
    #     dis_file=tree
    # )
    # mc_px, mc_py, mc_pz, mc_p, mc_p_theta, mc_p_phi, mc_PDG_ID, mc_charge, mc_generatorStatus, mc_vertex_x, mc_vertex_y, mc_vertex_z = mc.get_mc_info(
    #     name=name,
    #     rootfile=rootfile,
    #     verbose=VERBOSE,
    #     plot_verbose=PLOT_VERBOSE
    # )

    # # Initialize MatchingMCAndTrack class and perform matching
    # matching = MatchingMCAndTrack(
    #     track, 
    #     mc
    # )
    # r_min_tracks, r_min_track_index = matching.get_segments_nearest_impact_point(
    #     all_tracks, 
    #     rootfile=rootfile,
    #     verbose=VERBOSE,
    #     plot_verbose=PLOT_VERBOSE
    # )

    # matched_tracks, min_delta_angles, matched_pairs_on_btof, matched_pairs_on_etof = matching.match_track_to_mc(
    #     name=name,
    #     track_momentum=track_segments_p,
    #     track_momentum_x=track_segments_px,
    #     track_momentum_y=track_segments_py,
    #     track_momentum_z=track_segments_pz,
    #     track_momentum_transverse=tracksegments_pt,
    #     track_momentum_theta=track_segments_p_theta,
    #     track_momentum_phi=track_segments_p_phi,
    #     track_pos_x=track_segments_x,
    #     track_pos_y=track_segments_y,
    #     track_pos_z=track_segments_z,
    #     track_pathlength=track_segment_pathlength,
    #     mc_momentum_theta=mc_p_theta,
    #     mc_momentum_phi=mc_p_phi,
    #     mc_momentum=mc_p,
    #     mc_pdg_ID=mc_PDG_ID,
    #     mc_generator_status=mc_generatorStatus,
    #     mc_charge=mc_charge,
    #     mc_vertex_x=mc_vertex_x,
    #     mc_vertex_y=mc_vertex_y,
    #     mc_vertex_z=mc_vertex_z,
    #     r_min_track_index=r_min_track_index,
    #     threshold=0.5,
    #     rootfile=rootfile,
    #     verbose=VERBOSE,
    #     plot_verbose=PLOT_VERBOSE
    # )

    # # Plot efficiency
    # matching.plot_efficiency(
    #     name,
    #     matched_tracks, 
    #     min_delta_angles,
    #     threshold=0.5,
    #     rootfile=rootfile
    # )

    # # Initialize MatchingTrackandToFHits class and retrieve TOF information
    # tof = MatchingTrackandToFHits(
    #     dis_file=tree, 
    #     branch=branch
    # )
    # btof_phi_theta, ectof_phi_theta = tof.get_tof_info(
    #     name, 
    #     SELECTED_EVENTS, 
    #     rootfile=rootfile,
    #     verbose=VERBOSE,
    #     plot_verbose=PLOT_VERBOSE
    # )

    # # Match tracks to TOF hits
    # matched_tracks_and_btof_phi_theta, matched_tracks_and_etof_phi_theta = tof.match_track_to_tof(
    #     name, 
    #     matched_pairs_on_btof, 
    #     matched_pairs_on_etof, 
    #     btof_phi_theta, 
    #     ectof_phi_theta, 
    #     rootfile=rootfile,
    #     verbose=VERBOSE,
    #     plot_verbose=PLOT_VERBOSE
    # )

    # # Initialize ToFPIDPerformance class and plot PID performance
    # tof_pid = ToFPIDPerformance(
    #     dis_file=tree, 
    #     branch=branch
    # )

    # btof_calc_mass, btof_pdg, track_momentums_on_btof, track_momentums_transverse_on_btof = tof_pid.plot_pid_performance(
    #     name, 
    #     matched_tracks_and_btof_phi_theta,
    #     matched_tracks_and_etof_phi_theta,
    #     rootfile=rootfile,
    #     output_txt_name=output_txt_name
    # )

    # bincenter, sep = tof_pid.plot_separation_power_vs_momentum(
    #     btof_calc_mass,
    #     btof_pdg,
    #     track_momentums_on_btof,
    #     track_momentums_transverse_on_btof,
    #     name,
    #     rootfile=rootfile
    # )

    # btof_t0_timing, ectof_t0_timing, _, _ = tof_pid.pid_performance_include_t0_effect(
    #     name, 
    #     matched_tracks_and_btof_phi_theta, 
    #     matched_tracks_and_etof_phi_theta,
    #     SELECTED_EVENTS=SELECTED_EVENTS,
    #     rootfile=rootfile,
    #     output_txt_name=output_txt_name
    # )

    # tof_pid.plot_pid_performance_include_tof_time_reso_and_t0_effect(
    #     name, 
    #     matched_tracks_and_btof_phi_theta, 
    #     matched_tracks_and_etof_phi_theta,
    #     btof_t0_timing, 
    #     ectof_t0_timing,
    #     SELECTED_EVENTS=SELECTED_EVENTS,
    #     rootfile=rootfile,
    #     output_txt_name=output_txt_name
    # )

    # T0_MEAN_VALUES = [0]  # 0ps
    # T0_SIGMA_VALUES = [15e-12, 30e-12, 50e-12]  # 15ps, 30ps, 50ps
    # TOF_RESO_VALUES = [25e-12, 35e-12, 50e-12] 

    # analyze_t0_and_tof_reso_effect(
    #     T0_MEAN_VALUES, 
    #     T0_SIGMA_VALUES, 
    #     TOF_RESO_VALUES, 
    #     matched_tracks_and_btof_phi_theta, 
    #     matched_tracks_and_etof_phi_theta, 
    #     rootfile
    #     )
    
    analyze_separation_vs_vertex_z(
        name=name,
        rootfile=rootfile,
        vertex_z_ranges=[(-5, 5), (-35, 35), (-55, 55)],
        output_txt_name=output_txt_name,
        output_efficiency_result_name=output_efficiency_name
    )

if __name__ == "__main__":
    main()