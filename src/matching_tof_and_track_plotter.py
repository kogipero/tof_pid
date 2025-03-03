import awkward as ak
import helper_functions as myfunc
import ROOT as r
import numpy as np

class MatchingTOFAndTrackPlotter:
  def __init__(self, rootfile, name: str):
      self.rootfile = rootfile
      self.name = name

  def plot_matching_tof_and_track(self, btof_and_track_matched, delta_angles_all, min_delta_angles_events, angle_threshold):
      """
      Plots the matching between BTOF and track information.

      Args:
          btof_and_track_matched: Dictionary containing BTOF and track information.
          delta_angles_all: Delta angles between BTOF and track.
          min_delta_angles_events: Minimum delta angles between BTOF and track.
          angle_threshold: Threshold for the delta angle.
      """
      print('Start plotting matching TOF and track')

      myfunc.make_histogram_root(
          ak.flatten(delta_angles_all),
          100,
          hist_range=[0, angle_threshold],
          title='Delta_angles_for_all_tracks_matched_to_TOF',
          xlabel='Delta angle [rad]',
          ylabel='Entries',
          outputname=f'{self.name}/delta_angles_match_track_to_tof',
          rootfile=self.rootfile
      )

      myfunc.make_histogram_root(
          min_delta_angles_events,
          100,
          hist_range=[0, angle_threshold],
          title='Minimum_delta_angles_for_all_tracks_matched_to_TOF',
          xlabel='Delta angle [rad]',
          ylabel='Entries',
          outputname=f'{self.name}/min_delta_angles_match_track_to_tof',
          rootfile=self.rootfile
      )

      if len(btof_and_track_matched['tof_time']) > 0:
          myfunc.make_histogram_root(
              btof_and_track_matched['tof_time'],
              100,
              hist_range=[0, 10],
              title='BTOF_Time_matched_track_to_TOF',
              xlabel='Time [ns]',
              ylabel='Entries',
              outputname=f'{self.name}/btof_time_match_track_to_tof',
              rootfile=self.rootfile
          )

          myfunc.make_histogram_root(
              btof_and_track_matched['track_p'],
              100,
              hist_range=[0, 5],
              title='BTOF_Track_Momentum_matched_track_to_TOF',
              xlabel='Momentum [GeV]',
              ylabel='Entries',
              outputname=f'{self.name}/btof_track_momentum_match_track_to_tof',
              rootfile=self.rootfile
          )

          myfunc.make_histogram_root(
              btof_and_track_matched['mc_pdg'],
              100,
              hist_range=[-250, 250],
              title='BTOF_MC_PDG_ID_matched_track_to_TOF',
              xlabel='PDG ID',
              ylabel='Entries',
              outputname=f'{self.name}/btof_mc_pdg_match_track_to_tof',
              rootfile=self.rootfile
          )

          myfunc.make_histogram_root(
              btof_and_track_matched['track_pathlength'],
              100,
              hist_range=[0, 3000],
              title='BTOF_Track_Pathlength_matched_track_to_TOF',
              xlabel='Pathlength [mm]',
              ylabel='Entries',
              outputname=f'{self.name}/btof_track_pathlength_match_track_to_tof',
              rootfile=self.rootfile
          )
