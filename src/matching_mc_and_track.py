import numpy as np
import awkward as ak
from utility_function import angular_distance, calc_delta_phi, calc_delta_theta

from typing import List, Tuple, Dict
from matching_mc_and_track_plotter import MatchingMCAndTrackPlotter


class MatchingMCAndTrack:
  def __init__(self, track, mc, rootfile, name: str):
      self.track = track
      self.mc = mc
      self.rootfile = rootfile
      self.name = name
      self.plotter = MatchingMCAndTrackPlotter(rootfile, name)  

  def get_segments_nearest_impact_point(
        self, 
        all_tracks: List[List[List[Tuple]]], 
        verbose: bool = False, 
        plot_verbose: bool = False
    ) -> Tuple[List[List[float]], List[List[int]], List[List[int]]]:
    """
    Identifies the segments closest to the impact point for each track.
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

            min_track = min([segment[9] if len(segment) > 9 and segment[9] is not None else float('inf') for segment in track])
            min_index = [segment[9] if len(segment) > 9 and segment[9] is not None else float('inf') for segment in track].index(min_track)

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
        self.plotter.plot_minimum_track_distances(r_min_tracks)

    print('End getting nearest impact point')
        
    return r_min_tracks, r_min_track_index

  def build_all_segments_indices(self, all_tracks: List[List[List[Tuple]]]) -> List[List[List[int]]]:
      """
      From a structure where all_tracks[event_idx][track_idx] has a list (tuple) of segments,
      Create and return a list of the original segment index i for each event, track.
      """
      print('Start building all segments indices')

      all_segments_indices = []
      
      for event_idx, event_tracks in enumerate(all_tracks):
          event_indices_list = []
          for track_idx, track_segments in enumerate(event_tracks):
              seg_indices = [seg[1] for seg in track_segments] 
              event_indices_list.append(seg_indices)
          all_segments_indices.append(event_indices_list)

      return all_segments_indices


  def match_track_to_mc(
    self,
    name: str,
    track_momentum: ak.Array,
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
    all_segments_indices: List[List[List[int]]],
    threshold: float,
    rootfile,
    output_txt: str = 'matching_result.txt',
    vertex_z_min: float = -5,
    vertex_z_max: float = 5,
    verbose: bool = False,
    plot_verbose: bool = False
  ) -> Tuple[Dict[str, List], List[float], Dict[str, List], Dict[str, List]]:
    """
    Matches tracks to MC particles based on angular distance, then checks BTOF/ETOF hits.
    This version ensures that we only loop up to the minimum number of events across arrays,
    preventing out-of-range errors.
    """

    print('Start matching track to MC')

    #=================================================================================================
    # 1. Find the minimum value from all array lengths to avoid inconsistency in the number of events
    #=================================================================================================
    n_events_min = min(
        len(r_min_track_index),
        len(track_momentum_theta),
        len(track_pos_x),
        len(track_pos_y),
        len(track_pos_z),
        len(track_momentum),
        len(track_pathlength),
        len(mc_momentum_theta),
        len(mc_momentum_phi),
        len(mc_momentum),
        len(mc_pdg_ID),
        len(mc_generator_status),
        len(mc_charge),
        len(mc_vertex_x),
        len(mc_vertex_y),
        len(mc_vertex_z),
    )
    if verbose:
        print(f"[DEBUG] n_events_min = {n_events_min}")

    #============================================================
    # Initialize dictionaries, etc. to be used as return values
    #============================================================
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
        "mc_momentum_phi": [],
        "mc_momentum_theta": [],
        "mc_vertex_x": [],
        "mc_vertex_y": [],
        "mc_vertex_z": [],
        "track_pathlength": [],
        "match_momentum_resolutions_phi": [],
        "match_momentum_resolutions_theta": [],
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
        "track_momentum_theta_on_btof": [],
        "track_momentum_phi_on_btof": [],
        "track_pos_theta_on_btof": [],
        "track_pos_phi_on_btof": [],
        "track_pos_x_on_btof": [],
        "track_pos_y_on_btof": [],
        "track_pos_z_on_btof": [],
        "mc_pdg": [],
        "mc_momentum": [],
        "mc_momentum_phi": [],
        "mc_momentum_theta": [],
        "mc_vertex_x": [],
        "mc_vertex_y": [],
        "mc_vertex_z": [],
        "mc_vertex_d": [],
        "track_pathlength": [],
        "match_momentum_resolutions_on_btof": [],
        "match_momentum_resolutions_phi_on_btof": [],
        "match_momentum_resolutions_theta_on_btof": [],

    }

    matched_pairs_on_etof = {
        "event_idx": [],
        "track_idx": [],
        "track_momentum_on_etof": [],
        "track_momentum_transverse_on_etof": [],
        "track_momentum_theta_on_etof": [],
        "track_momentum_phi_on_etof": [],
        "track_pos_theta_on_etof": [],
        "track_pos_phi_on_etof": [],
        "track_pos_x_on_etof": [],
        "track_pos_y_on_etof": [],
        "track_pos_z_on_etof": [],
        "mc_pdg": [],
        "mc_momentum": [],
        "mc_momentum_phi": [],
        "mc_momentum_theta": [],
        "mc_vertex_x": [],
        "mc_vertex_y": [],
        "mc_vertex_z": [],
        "track_pathlength": [],
        "match_momentum_resolutions_on_etof": [],
        "match_momentum_resolutions_phi_on_etof": [],
        "match_momentum_resolutions_theta_on_etof": [],
    }

    #======================================================================
    # 2. Event loop: 0 ~ n_events_min-1
    #======================================================================
    for event_idx in range(n_events_min):

        if len(track_momentum_theta[event_idx]) == 0 or len(mc_momentum_theta[event_idx]) == 0:
            if verbose:
                print(f"Skipping empty event at index {event_idx}")
            continue

        mc_momentum_event          = np.array(mc_momentum[event_idx])
        mc_momentum_theta_event    = np.array(mc_momentum_theta[event_idx])
        mc_momentum_phi_event      = np.array(mc_momentum_phi[event_idx])
        mc_pdg_event               = np.array(mc_pdg_ID[event_idx])
        mc_genstat_event           = np.array(mc_generator_status[event_idx])
        mc_charge_event            = np.array(mc_charge[event_idx])
        mc_vx_event                = np.array(mc_vertex_x[event_idx])
        mc_vy_event                = np.array(mc_vertex_y[event_idx])
        mc_vz_event                = np.array(mc_vertex_z[event_idx])

        mc_vertex_d_event = np.sqrt(mc_vx_event**2 + mc_vy_event**2 + mc_vz_event**2)

        stable_indices = (
            (mc_genstat_event == 1) &
            (mc_charge_event != 0) &
            (mc_vx_event > -100) & (mc_vx_event < 100) &
            (mc_vy_event > -100) & (mc_vy_event < 100)
        )
        vertex_z_indices = ((mc_vz_event > vertex_z_min) & (mc_vz_event < vertex_z_max))
        final_indices    = stable_indices & vertex_z_indices

        mc_momentum_event = mc_momentum_event[final_indices]
        mc_momentum_theta_event    = mc_momentum_theta_event[final_indices]
        mc_momentum_phi_event     = mc_momentum_phi_event[final_indices]
        mc_pdg_event      = mc_pdg_event[final_indices]
        mc_vx_event       = mc_vx_event[final_indices]
        mc_vy_event       = mc_vy_event[final_indices]
        mc_vz_event       = mc_vz_event[final_indices]
        mc_vertex_d_event = mc_vertex_d_event[final_indices]

        track_pos_x_event  = np.array(track_pos_x[event_idx])
        track_pos_y_event  = np.array(track_pos_y[event_idx])
        track_pos_z_event  = np.array(track_pos_z[event_idx])
        track_p_theta_event= np.array(track_momentum_theta[event_idx])
        track_p_phi_event  = np.array(track_momentum_phi[event_idx])
        track_p_event      = np.array(track_momentum[event_idx])
        track_pt_event     = np.array(track_momentum_transverse[event_idx])
        track_path_event   = np.array(track_pathlength[event_idx])

        track_pos_phi_event   = np.arctan2(track_pos_y_event, track_pos_x_event)
        track_pos_theta_event = np.arctan2(np.sqrt(track_pos_x_event**2 + track_pos_y_event**2),
                                        track_pos_z_event)

        
        #=======================================================================================
        # r_min_track_index[event_idx] → minimum radius segment per track for the relevant event
        #=======================================================================================
        if event_idx >= len(r_min_track_index):
            if verbose:
                print(f"Warning: event_idx={event_idx} but r_min_track_index has only {len(r_min_track_index)} elements.")
            continue

        min_index_list = r_min_track_index[event_idx]

        for track_idx, min_index in enumerate(min_index_list):

            if (min_index < 0) or (min_index >= len(track_p_theta_event)):
                if verbose:
                    print(f"Skipping invalid index: min_index={min_index} (track_p_theta_event size={len(track_p_theta_event)})")
                continue

            track_p_theta_val = track_p_theta_event[min_index]
            track_p_phi_val   = track_p_phi_event[min_index]
            track_p_val       = track_p_event[min_index]
            track_pt_val      = track_pt_event[min_index]

            delta_angles = angular_distance(
                phi1   = track_p_phi_val, # scalar
                theta1 = track_p_theta_val, # scalar
                phi2   = mc_momentum_phi_event, # np.array
                theta2 = mc_momentum_theta_event # np.array
            )

            if len(delta_angles) == 0:
                # kokoniyokuhairunohananddeda????????????????????????? 2/28
                if verbose:
                    print(f"Warning: delta_angles is empty (event {event_idx}, track {track_idx}).")
                continue

            imin_mc = np.argmin(delta_angles)
            min_delta_angle = delta_angles[imin_mc]
            delta_angles_all.append(delta_angles)

            all_matched_pairs["event_idx"].append(event_idx)
            all_matched_pairs["track_idx"].append(track_idx)
            all_matched_pairs["track_pos_theta"].append(track_pos_theta_event[min_index])
            all_matched_pairs["track_pos_phi"].append(track_pos_phi_event[min_index])

            if min_delta_angle > threshold:
                if verbose:
                    print(f"Skipping track {track_idx} in event {event_idx} due to large delta angle: {min_delta_angle}")
                continue

            matched_pairs["event_idx"].append(event_idx)
            matched_pairs["track_idx"].append(track_idx)
            matched_pairs["track_p"].append(track_p_val)
            matched_pairs["track_pt"].append(track_pt_val)
            matched_pairs["track_p_theta"].append(track_p_theta_val)
            matched_pairs["track_p_phi"].append(track_p_phi_val)
            matched_pairs["track_pos_theta"].append(track_pos_theta_event[min_index])
            matched_pairs["track_pos_phi"].append(track_pos_phi_event[min_index])
            matched_pairs["track_pos_x"].append(track_pos_x_event[min_index])
            matched_pairs["track_pos_y"].append(track_pos_y_event[min_index])
            matched_pairs["track_pos_z"].append(track_pos_z_event[min_index])
            matched_pairs["mc_theta"].append(mc_momentum_theta_event[imin_mc])
            matched_pairs["mc_phi"].append(mc_momentum_phi_event[imin_mc])
            matched_pairs["min_delta_angle"].append(min_delta_angle)
            matched_pairs["mc_pdg"].append(mc_pdg_event[imin_mc])
            matched_pairs["mc_momentum"].append(mc_momentum_event[imin_mc])
            matched_pairs["mc_momentum_phi"].append(mc_momentum_phi_event[imin_mc])
            matched_pairs["mc_momentum_theta"].append(mc_momentum_theta_event[imin_mc])
            matched_pairs["mc_vertex_x"].append(mc_vx_event[imin_mc])
            matched_pairs["mc_vertex_y"].append(mc_vy_event[imin_mc])
            matched_pairs["mc_vertex_z"].append(mc_vz_event[imin_mc])
            matched_pairs["track_pathlength"].append(track_path_event[min_index])

            delta_phi = calc_delta_phi(mc_momentum_phi_event[imin_mc], track_p_phi_val)
            delta_theta = calc_delta_theta(mc_momentum_theta_event[imin_mc], track_p_theta_val)

            matched_pairs["match_momentum_resolutions_phi"].append(delta_phi)
            matched_pairs["match_momentum_resolutions_theta"].append(delta_theta)


            min_delta_angles_all_tracks.append(min_delta_angle)

            if event_idx >= len(all_segments_indices):
                if verbose:
                    print(f"Warning: event_idx={event_idx} out of range for all_segments_indices.")
                continue

            if track_idx >= len(all_segments_indices[event_idx]):
                #comment
                #Current analysis enters here, improvement needed.
                if verbose:
                    print(f"Warning: track_idx={track_idx} out of range for all_segments_indices in event {event_idx}.")
                continue

            same_track_segment_list = all_segments_indices[event_idx][track_idx]

            for seg_idx in same_track_segment_list:
                if (seg_idx < 0) or (seg_idx >= len(track_pos_z_event)):
                    continue

                seg_x       = track_pos_x_event[seg_idx]
                seg_y       = track_pos_y_event[seg_idx]
                seg_z       = track_pos_z_event[seg_idx]
                seg_r       = np.sqrt(seg_x**2 + seg_y**2)
                seg_p       = track_p_event[seg_idx]
                seg_pt      = track_pt_event[seg_idx]
                seg_p_phi   = track_p_phi_event[seg_idx]
                seg_p_theta = track_p_theta_event[seg_idx]
                seg_theta   = track_pos_theta_event[seg_idx]
                seg_phi     = track_pos_phi_event[seg_idx]
                seg_path    = track_path_event[seg_idx]

                # --- judge if the track is on ETOF or BTOF ---
                # --- ETOF ---
                if (1840 <= seg_z <= 1880) and (0 <= seg_r <= 600):
                    matched_pairs_on_etof["event_idx"].append(event_idx)
                    matched_pairs_on_etof["track_idx"].append(track_idx)
                    matched_pairs_on_etof["track_pos_theta_on_etof"].append(seg_theta)
                    matched_pairs_on_etof["track_pos_phi_on_etof"].append(seg_phi)
                    matched_pairs_on_etof["track_pos_x_on_etof"].append(seg_x)
                    matched_pairs_on_etof["track_pos_y_on_etof"].append(seg_y)
                    matched_pairs_on_etof["track_pos_z_on_etof"].append(seg_z)
                    matched_pairs_on_etof["track_momentum_on_etof"].append(seg_p)
                    matched_pairs_on_etof["track_momentum_transverse_on_etof"].append(seg_pt)
                    matched_pairs_on_etof["track_momentum_theta_on_etof"].append(seg_p_theta)
                    matched_pairs_on_etof["track_momentum_phi_on_etof"].append(seg_p_phi)
                    matched_pairs_on_etof["mc_pdg"].append(mc_pdg_event[imin_mc])
                    matched_pairs_on_etof["mc_momentum"].append(mc_momentum_event[imin_mc])
                    matched_pairs_on_etof["mc_momentum_phi"].append(mc_momentum_phi_event[imin_mc])
                    matched_pairs_on_etof["mc_momentum_theta"].append(mc_momentum_theta_event[imin_mc])
                    matched_pairs_on_etof["mc_vertex_x"].append(mc_vx_event[imin_mc])
                    matched_pairs_on_etof["mc_vertex_y"].append(mc_vy_event[imin_mc])
                    matched_pairs_on_etof["mc_vertex_z"].append(mc_vz_event[imin_mc])
                    matched_pairs_on_etof["track_pathlength"].append(seg_path)
                    matched_pairs_on_etof["match_momentum_resolutions_on_etof"].append(mc_momentum_event[imin_mc] - seg_p)

                    delta_phi_on_etof = calc_delta_phi(mc_momentum_phi_event[imin_mc], seg_phi)
                    delta_theta_on_etof = calc_delta_theta(mc_momentum_theta_event[imin_mc], seg_theta)

                    matched_pairs_on_etof["match_momentum_resolutions_phi_on_etof"].append(delta_phi_on_etof)
                    matched_pairs_on_etof["match_momentum_resolutions_theta_on_etof"].append(delta_theta_on_etof)

                # --- BTOF ---
                if (-1500 <= seg_z <= 1840) and (625 <= seg_r <= 642):
                    matched_pairs_on_btof["event_idx"].append(event_idx)
                    matched_pairs_on_btof["track_idx"].append(track_idx)
                    matched_pairs_on_btof["track_pos_theta_on_btof"].append(seg_theta)
                    matched_pairs_on_btof["track_pos_phi_on_btof"].append(seg_phi)
                    matched_pairs_on_btof["track_pos_x_on_btof"].append(seg_x)
                    matched_pairs_on_btof["track_pos_y_on_btof"].append(seg_y)
                    matched_pairs_on_btof["track_pos_z_on_btof"].append(seg_z)
                    matched_pairs_on_btof["track_momentum_on_btof"].append(seg_p)
                    matched_pairs_on_btof["track_momentum_transverse_on_btof"].append(seg_pt)
                    matched_pairs_on_btof["track_momentum_theta_on_btof"].append(seg_p_theta)
                    matched_pairs_on_btof["track_momentum_phi_on_btof"].append(seg_p_phi)
                    matched_pairs_on_btof["mc_pdg"].append(mc_pdg_event[imin_mc])
                    matched_pairs_on_btof["mc_momentum"].append(mc_momentum_event[imin_mc])
                    matched_pairs_on_btof["mc_momentum_phi"].append(mc_momentum_phi_event[imin_mc])
                    matched_pairs_on_btof["mc_momentum_theta"].append(mc_momentum_theta_event[imin_mc])
                    matched_pairs_on_btof["mc_vertex_x"].append(mc_vx_event[imin_mc])
                    matched_pairs_on_btof["mc_vertex_y"].append(mc_vy_event[imin_mc])
                    matched_pairs_on_btof["mc_vertex_z"].append(mc_vz_event[imin_mc])
                    matched_pairs_on_btof["mc_vertex_d"].append(mc_vertex_d_event[imin_mc])
                    matched_pairs_on_btof["track_pathlength"].append(seg_path)
                    matched_pairs_on_btof["match_momentum_resolutions_on_btof"].append(mc_momentum_event[imin_mc] - seg_p)

                    delta_phi_on_btof = calc_delta_phi(mc_momentum_phi_event[imin_mc], seg_phi)
                    delta_theta_on_btof = calc_delta_theta(mc_momentum_theta_event[imin_mc], seg_theta)

                    matched_pairs_on_btof["match_momentum_resolutions_phi_on_btof"].append(delta_phi_on_btof)
                    matched_pairs_on_btof["match_momentum_resolutions_theta_on_btof"].append(delta_theta_on_btof)

    with open(output_txt, 'a') as f:
        f.write(f'{name}, all tracks: {len(ak.flatten(r_min_track_index))}\n')
        f.write(f'{name}, MC and Track matched tracks: {len(matched_pairs["track_idx"])}\n')
        f.write(f'{name}, MC and Track matched tracks on BTOF: {len(matched_pairs_on_btof["track_idx"])}\n')
        f.write(f'{name}, MC and Track matched tracks on ETOF: {len(matched_pairs_on_etof["track_idx"])}\n')
        f.write(f'{name}, MC and Track matching efficiency: {len(matched_pairs["track_idx"]) / len(ak.flatten(r_min_track_index))}\n')
        f.write(f'{name}, MC and Track matching Number of cut tracks: {len(ak.flatten(r_min_track_index)) - len(matched_pairs["track_idx"])}\n')
        f.write(f'{name}, MC and Track threshold cut efficiency: {len(matched_pairs["track_idx"]) / len(all_matched_pairs["track_idx"])}\n')

    if plot_verbose:
        self.plotter.plot_matching_results(min_delta_angles_all_tracks,
                                            delta_angles_all, 
                                            matched_pairs, matched_pairs_on_btof
                                            )
        
    print('End matching track to MC')


    self.plotter.plot_matching_efficiency(name, matched_pairs, all_matched_pairs, threshold, rootfile)

    if plot_verbose:
        self.plotter.plot_matching_results(min_delta_angles_all_tracks,
                                            delta_angles_all, 
                                            matched_pairs, matched_pairs_on_btof
                                            )

    return matched_pairs, min_delta_angles_all_tracks, matched_pairs_on_btof, matched_pairs_on_etof