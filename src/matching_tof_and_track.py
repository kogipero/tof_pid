import numpy as np
import awkward as ak
import uproot

from utility_function import angular_distance

from typing import Tuple
from matching_tof_and_track_plotter import MatchingTOFAndTrackPlotter

class MatchingTOFAndTrackReader:
  def __init__(self, dis_file: uproot.TTree, branch: dict, name: str, rootfile):
      self.name = name
      self.rootfile = rootfile
      self.branch = branch
      self.dis_file = dis_file
      self.plotter = MatchingTOFAndTrackPlotter(rootfile, name)

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
        self.plotter.plot_matching_tof_and_track(
          btof_and_track_matched,
          delta_angles_all,
          min_delta_angles_events,
          angle_threshold,
        )

    return btof_and_track_matched, ectof_and_track_matched