import numpy as np
import helper_functions as myfunc
import awkward as ak

class TrackPlotter:
    def __init__(self, rootfile, name: str):
        """
        Constructor for the TrackPlotter class.
        
        Args:
            rootfile: ROOT file to save the histograms
            name (str): Name of the track
        """
        self.rootfile = rootfile
        self.name = name

    def plot_track_segments_pos(self, track_segments_pos_x, track_segments_pos_y, track_segments_pos_z, track_segments_pos_d, track_segments_pos_r):
        print('Start plotting track segment positions')

        myfunc.make_histogram_root(ak.flatten(track_segments_pos_x),
                        100, [-1000, 1000], 'Track_segments_pos_x', 'x [mm]', 'Entries',
                        f'{self.name}/track_segments_pos_x', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_pos_y),
                        100, [-1000, 1000], 'Track_segments_pos_y', 'y [mm]', 'Entries',
                        f'{self.name}/track_segments_pos_y', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_pos_z),
                        100, [-1000, 1000], 'Track_segments_pos_z', 'z [mm]', 'Entries',
                        f'{self.name}/track_segments_pos_z', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_pos_d),
                        300, [0, 3000], 'Track_segments_pos_d', 'd [mm]', 'Entries',
                        f'{self.name}/track_segments_pos_d', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_pos_r),
                        300, [0, 3000], 'Track_segments_r', 'r [mm]', 'Entries',
                        f'{self.name}/track_segments_pos_r', self.rootfile)

        print('End plotting track segment positions')

    def plot_track_segments_momentum(self, track_segments_px, track_segments_py, track_segments_pz, track_segments_p, track_segments_pt, track_segments_p_theta, track_segments_p_phi, track_segment_pathlength):
        print('Start plotting track segment momentum')

        myfunc.make_histogram_root(ak.flatten(track_segments_px),
                        30, [0, 30], 'Track_segments_momentum_x', 'px [GeV]', 'Entries',
                        f'{self.name}/track_segments_momentum_x', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_py),
                        30, [0, 30], 'Track_segments_momentum_y', 'py [GeV]', 'Entries',
                        f'{self.name}/track_segments_momentum_y', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_pz),
                        30, [0, 30], 'Track_segments_momentum_z', 'pz [GeV]', 'Entries',
                        f'{self.name}/track_segments_momentum_z', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_p),
                        100, [0, 20], 'Track_segments_p', 'p [GeV]', 'Entries',
                        f'{self.name}/track_segments_momentum', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_pt),
                        100, [0, 20], 'Track_segments_transverse_momentum', 'pt [GeV]', 'Entries',
                        f'{self.name}/track_segments_transverse_momentum', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_p_theta),
                        100, [0, 3.2], 'Track_segments_momentum_theta', 'theta [rad]', 'Entries',
                        f'{self.name}/track_segments_momentum_theta', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segments_p_phi),
                        100, [-3.2, 3.2], 'Track_segments_momentum_phi', 'phi [rad]', 'Entries',
                        f'{self.name}/track_segments_momentum_phi', self.rootfile)

        myfunc.make_histogram_root(ak.flatten(track_segment_pathlength),
                        300, [0, 3000], 'Track_segments_pathlength', 'Pathlength [mm]', 'Entries',
                        f'{self.name}/track_segments_pathlength', self.rootfile)

        print('End plotting track segment momentum')

    def plot_split_tracks(self, all_tracks):
      """
      Plots the split tracks using ROOT.

      Args:
          all_tracks (List[List[List[Tuple]]]): Nested list of split tracks for each event.
      """
      print('Start plotting split tracks')

      for event_idx, event_tracks in enumerate(all_tracks[:5]):
          for track_idx, track in enumerate(event_tracks):
              x_pos_per_track = np.array([segment[3] for segment in track])
              y_pos_per_track = np.array([segment[4] for segment in track])

              myfunc.make_TGraph(
                  x_pos_per_track, 
                  y_pos_per_track,
                  title=f'Track_segments_{track_idx}',
                  xlabel='x [mm]',
                  ylabel='y [mm]',
                  outputname=f'{self.name}/track_{track_idx}',
                  rangex=1000,
                  rangey=1000,
                  rootfile=self.rootfile
              )

      print('End plotting split tracks')