import awkward as ak
import numpy as np
import ROOT as r
import helper_functions as myfunc

class MatchingMCAndTrackPlotter:
    def __init__(self, rootfile, name: str):
        self.rootfile = rootfile
        self.name = name

    def plot_minimum_track_distances(self, r_min_tracks):
        """
        Plots minimum track distances to the impact point.
        """
        print('Start plotting minimum track distances')

        myfunc.make_histogram_root(ak.flatten(r_min_tracks),
                        100,
                        hist_range=[0, 500],
                        title='Minimum_track_distances_to_impact_point',
                        xlabel='Distance [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/min_track_distances',
                        rootfile=self.rootfile
                        )
        print('End plotting minimum track distances')

    def plot_matching_results(self, 
                              min_delta_angles_all_tracks, 
                              delta_angles_all, 
                              matched_pairs, 
                              matched_pairs_on_btof
                              ):
        """
        Plots histograms related to track and MC matching.
        """
        print('Start plotting matching results')

        myfunc.make_histogram_root(min_delta_angles_all_tracks,
                          100,
                          hist_range=[0, 3.2],
                          title='Minimum_delta_angles_for_all_tracks_matched_to_MC',
                          xlabel='Delta angle [rad]',
                          ylabel='Entries',
                          outputname=f'{self.name}/min_delta_angles',
                          rootfile=self.rootfile
                          )
          
        myfunc.make_histogram_root(ak.flatten(delta_angles_all),
                        100,
                        hist_range=[0, 3.2],
                        title='Delta_angles_for_all_tracks_matched_to_MC',
                        xlabel='Delta angle [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/delta_angles_all',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs["track_pathlength"],
                        100,
                        hist_range=[0, 7000],
                        title='Track_pathlength_matched_to_MC',
                        xlabel='Pathlength [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/track_pathlength',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs["track_pt"],
                        100,
                        hist_range=[0, 5],
                        title='Track_pt_matched_to_MC',
                        xlabel='pt [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/track_pt',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs["track_p"],
                        100,
                        hist_range=[0, 5],
                        title='Track_momentum_matched_to_track',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs["mc_pdg"],
                        100,
                        hist_range=[-250, 250],
                        title='MC_PDG_ID_matched_to_track',
                        xlabel='PDG ID',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_pdg',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs_on_btof["mc_pdg"],
                        100,
                        hist_range=[-250, 250],
                        title='MC_PDG_ID_on_BTOF_matched_to_track',
                        xlabel='PDG ID',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_pdg_on_btof',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs["mc_vertex_x"],
                        100,
                        hist_range=[-1000, 1000],
                        title='MC_vertex_pos_x_matched_to_track',
                        xlabel='x [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_vertex_x',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs["mc_vertex_y"],
                        100,
                        hist_range=[-1000, 1000],
                        title='MC_vertex_pos_y_matched_to_track',
                        xlabel='y [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_vertex_y',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs["mc_vertex_z"],
                        100,
                        hist_range=[-1000, 1000],
                        title='MC_vertex_pos_z_matched_to_track',
                        xlabel='z [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_vertex_z',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(matched_pairs_on_btof["mc_momentum"],
                        100,
                        hist_range=[0, 5],
                        title='MC_momentum_on_BTOF_matched_to_track',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum_on_btof',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(matched_pairs_on_btof["track_momentum_on_btof"],
                        100,
                        hist_range=[0, 5],
                        title='Track_momentum_on_BTOF_matched_to_track',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/track_momentum_on_btof',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs_on_btof["track_pathlength"],
                        100,
                        hist_range=[0, 3000],
                        title='Track_pathlength_on_BTOF_matched_to_MC',
                        xlabel='Pathlength [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/track_pathlength_on_btof',
                        rootfile=self.rootfile
                        )

        myfunc.make_2Dhistogram_root(matched_pairs_on_btof["track_pos_x_on_btof"],
                        100,
                        [-1000, 1000],
                        matched_pairs_on_btof["track_pos_y_on_btof"],
                        100,
                        [-1000, 1000],
                        title='Track_pos_xy_on_BTOF_matched_to_MC',
                        xlabel='x [mm]',
                        ylabel='y [mm]',
                        outputname=f'{self.name}/track_pos_on_btof',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(matched_pairs_on_btof["track_momentum_theta_on_btof"],
                        100,
                        hist_range=[0, 3.2],
                        title='Track_momentum_theta_on_BTOF_matched_to_MC',
                        xlabel='Theta [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/momentum_theta_on_btof',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs_on_btof["track_momentum_phi_on_btof"],
                        100,
                        hist_range=[-3.2, 3.2],
                        title='Track_momentum_phi_on_BTOF_matched_to_MC',
                        xlabel='Phi [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/momentum_phi_on_btof',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs_on_btof["mc_momentum_theta"],
                        100,
                        hist_range=[0, 3.2],
                        title='MC_momentum_theta_matched_to_track',
                        xlabel='Theta [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum_theta',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs_on_btof["mc_momentum_phi"],
                        100,
                        hist_range=[-3.2, 3.2],
                        title='MC_momentum_phi_matched_to_track',
                        xlabel='Phi [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum_phi',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(matched_pairs_on_btof["match_momentum_resolutions_on_btof"],
                        100,
                        hist_range=[-0.5, 0.5],
                        title='Momentum_resolutions_on_BTOF_matched_to_MC',
                        xlabel='Momentum resolution [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/momentum_resolutions_on_btof',
                        rootfile=self.rootfile
                        )

        myfunc.make_2Dhistogram_root(matched_pairs_on_btof["match_momentum_resolutions_phi_on_btof"],
                        100,
                        [-0.5, 0.5],
                        matched_pairs_on_btof["match_momentum_resolutions_theta_on_btof"],
                        100,
                        [-0.5, 0.5],
                        title='Momentum_resolutions_on_BTOF_matched_to_MC',
                        xlabel='Phi resolution [rad]',
                        ylabel='Theta resolution [rad]',
                        outputname=f'{self.name}/momentum_resolutions_on_btof',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs_on_btof["match_momentum_resolutions_phi_on_btof"],
                        100,
                        hist_range=[-0.5, 0.5],
                        title='Phi_resolution_on_BTOF_matched_to_MC',
                        xlabel='Phi resolution [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/phi_resolutions_on_btof',
                        rootfile=self.rootfile
                        )
        
        myfunc.make_histogram_root(matched_pairs_on_btof["match_momentum_resolutions_theta_on_btof"],
                        100,
                        hist_range=[-0.5, 0.5],
                        title='Theta_resolution_on_BTOF_matched_to_MC',
                        xlabel='Theta resolution [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/theta_resolutions_on_btof',
                        rootfile=self.rootfile
                        )

        print('End plotting matching results')

    def plot_matching_efficiency(self, name: str, matched_pairs, all_matched_pairs, threshold: float, rootfile):
      """
      Performs the actual matching between tracks and MC data.
      """
      """
      Plots efficiency of track-matching based on a threshold for minimum delta angles, 
      but the final ratio is now plotted in a Root TH2D and stored in the rootfile.
      """

      track_phi_all = ak.to_numpy(all_matched_pairs['track_pos_phi'])
      track_theta_all = ak.to_numpy(all_matched_pairs['track_pos_theta'])

      track_phi_include_threshold = ak.to_numpy(matched_pairs['track_pos_phi'])
      track_theta_include_threshold = ak.to_numpy(matched_pairs['track_pos_theta'])

      # threshold = 0.5
      for threshold in [threshold]:

        myfunc.make_stacked_histogram_root([track_phi_all, track_phi_include_threshold],
                                    nbins=100,
                                    hist_range=[-3.5, 3.5],
                                    labels=['All', f'Include threshold {threshold}'],
                                    title='Track_phi_comparison_with_threshold',
                                    xlabel='phi [rad]',
                                    ylabel='Entries',
                                    outputname=f'{name}/track_phi_threshold_{threshold}',
                                    rootfile=rootfile
                                    )
        
        myfunc.make_stacked_histogram_root([track_theta_all, track_theta_include_threshold],
                                    nbins=100,
                                    hist_range=[0, 3.5],
                                    labels=['All', f'Include threshold {threshold}'],
                                    title='Track_theta_comparison_with_threshold',
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
                                        title='Track_phi_vs_theta',
                                        xlabel='phi [rad]',
                                        ylabel='theta [rad]',
                                        cmap='viridis')

        fill2, ax2, fill2, x_edge2, y_edge2 = myfunc.make_2Dhistogram(track_phi_include_threshold,
                                        32,
                                        [-3.2, 3.2],
                                        track_theta_include_threshold,
                                        32,
                                        [0, 3.2],
                                        title=f'Track_phi_vs_theta_include_threshold_{threshold}',
                                        xlabel='phi [rad]',
                                        ylabel='theta [rad]',
                                        cmap='viridis')

        ratio = np.divide(fill2, fill1, out=np.zeros_like(fill1), where=fill1 != 0)

        x_nbins = len(x_edge1)-1
        y_nbins = len(y_edge1)-1
        hist_eff_name = f"hist_mc_track_efficiency_map_th_{threshold}"
        hist_eff = r.TH2D(hist_eff_name,
                        f"Efficiency_map_(threshold={threshold});phi_[rad];theta_[rad]",
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

      print('End plotting efficiency')
