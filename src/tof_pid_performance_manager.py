import numpy as np
import awkward as ak
import uproot
import ROOT as r

from typing import List, Tuple, Dict
from tof_pid_performance_plotter import TOFPIDPerformancePlotter

class ToFPIDPerformanceManager:
  def __init__(self, dis_file: uproot.TTree, branch: dict, name: str, rootfile):
      self.name = name
      self.rootfile = rootfile
      self.branch = branch
      self.dis_file = dis_file
      self.tof_pid_performance_plotter = TOFPIDPerformancePlotter(rootfile, name)

  def process_pid_performance_plot(self, 
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
                            plot_verbose: bool = False
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
      track_momentums_pi_on_btof = []
      track_momentums_k_on_btof = []
      track_momentums_p_on_btof = []
      track_momentums_e_on_btof = []
      btof_beta_inversees = []
      btof_pi_beta_inversees = []
      btof_k_beta_inversees = []
      btof_p_beta_inversees = []
      btof_e_beta_inversees = []
      etof_beta_inversees = []
      btof_calc_mass = []
      etof_calc_mass = []

      incorrect_masses_btof_pi = []
      incorrect_time_btof_pi = []
      incorrect_momentums_btof_pi = []
      incorrect_track_pathlength_btof_pi = []
      correct_time_btof_pi = []
      correct_momentums_btof_pi = []
      correct_track_pathlength_btof_pi = []

      incorrect_masses_btof_k = []
      incorrect_time_btof_k = []
      incorrect_momentums_btof_k = []
      incorrect_track_pathlength_btof_k = []
      correct_time_btof_k = []
      correct_momentums_btof_k = []
      correct_track_pathlength_btof_k = []

      incorrect_masses_btof_p = []
      incorrect_time_btof_p = []
      incorrect_momentums_btof_p = []
      incorrect_track_pathlength_btof_p = []
      correct_time_btof_p = []
      correct_momentums_btof_p = []
      correct_track_pathlength_btof_p = []

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

          if btof_pdg[i] == 211 or btof_pdg[i] == -211:
              if abs(calc_mass - 139) < MERGIN_PI:
                  btof_pi_beta_inversees.append(btof_beta_inverse)
                  track_momentums_pi_on_btof.append(track_momentum_on_btof[i])
              if abs(calc_mass - 139) < LARGE_MERGIN_PI:
                  incorrect_masses_btof_pi.append(calc_mass)
                  incorrect_time_btof_pi.append(current_time)
                  incorrect_momentums_btof_pi.append(track_momentum_on_btof[i])
                  incorrect_track_pathlength_btof_pi.append(btof_pathlength[i])
              else:
                  correct_time_btof_pi.append(current_time)
                  correct_momentums_btof_pi.append(track_momentum_on_btof[i])
                  correct_track_pathlength_btof_pi.append(btof_pathlength[i])

          elif btof_pdg[i] == 321 or btof_pdg[i] == -321:
              if abs(calc_mass - 493) < MERGIN_K:
                  btof_k_beta_inversees.append(btof_beta_inverse)
                  track_momentums_k_on_btof.append(track_momentum_on_btof[i])
              if abs(calc_mass - 493) < LARGE_MERGIN_K:
                  incorrect_masses_btof_k.append(calc_mass)
                  incorrect_time_btof_k.append(current_time)
                  incorrect_momentums_btof_k.append(track_momentum_on_btof[i])
                  incorrect_track_pathlength_btof_k.append(btof_pathlength[i])
              else:
                  correct_time_btof_k.append(current_time)
                  correct_momentums_btof_k.append(track_momentum_on_btof[i])
                  correct_track_pathlength_btof_k.append(btof_pathlength[i])

          elif btof_pdg[i] == 2212 or btof_pdg[i] == -2212:
              if abs(calc_mass - 938) < MERGIN_P:
                  btof_p_beta_inversees.append(btof_beta_inverse)
                  track_momentums_p_on_btof.append(track_momentum_on_btof[i])
              if abs(calc_mass - 938) < LARGE_MERGIN_P:
                  incorrect_masses_btof_p.append(calc_mass)
                  incorrect_time_btof_p.append(current_time)
                  incorrect_momentums_btof_p.append(track_momentum_on_btof[i])
                  incorrect_track_pathlength_btof_p.append(btof_pathlength[i])
              else:
                  correct_time_btof_p.append(current_time)
                  correct_momentums_btof_p.append(track_momentum_on_btof[i])
                  correct_track_pathlength_btof_p.append(btof_pathlength[i])

          elif btof_pdg[i] == 11 or btof_pdg[i] == -11:
              btof_e_beta_inversees.append(btof_beta_inverse)
              track_momentums_e_on_btof.append(track_momentum_on_btof[i])

      for i in range(len(ectof_time)):
          current_time = ectof_time[i]
          etof_beta = ectof_pathlength[i] / current_time
          etof_beta_c = etof_beta / 299.792458  # Speed of light in mm/ns
          etof_beta_inverse = 1 / etof_beta_c
          calc_mass = 1000 * track_momentum_on_ectof[i] * np.sqrt(1 - etof_beta_c**2) / etof_beta_c
          etof_beta_inversees.append(etof_beta_inverse)
          etof_calc_mass.append(calc_mass)
          track_momentums_on_ectof.append(track_momentum_on_ectof[i])

      if plot_verbose:
          self.tof_pid_performance_plotter.plot_tof_pid_performance(
              track_momentums_on_btof,
              track_momentums_on_ectof,
              btof_beta_inversees,
              btof_calc_mass
          )

      m_pi = 139 # MeV
      m_k = 493 # MeV
      m_p = 938 # MeV
      m_e = 0.511 # MeV

      pi_calc_mass_on_btof = []
      k_calc_mass_on_btof = []
      p_calc_mass_on_btof = []
      e_calc_mass_on_btof = []

      pi_mass_count_btof = 0
      pi_mass_count_btof_large_mergin = 0
      pi_mass_count_btof_low_momentum = 0
      k_mass_count_btof = 0
      k_mass_count_btof_large_mergin = 0
      k_mass_count_btof_low_momentum = 0
      p_mass_count_btof = 0
      p_mass_count_btof_large_mergin = 0
      p_mass_count_btof_low_momentum = 0

      pi_momentum_in_low_momentum_btof = []
      k_momentum_in_low_momentum_btof = []
      p_momentum_in_low_momentum_btof = []

      for i in range(len(btof_calc_mass)):
          if track_momentums_on_btof[i] < MOMENTUM_RANGE:
              if btof_pdg[i] == 211 or btof_pdg[i] == -211:
                  pi_momentum_in_low_momentum_btof.append(track_momentums_on_btof[i])

              if btof_pdg[i] == 321 or btof_pdg[i] == -321:
                  k_momentum_in_low_momentum_btof.append(track_momentums_on_btof[i])

              if btof_pdg[i] == 2212 or btof_pdg[i] == -2212:
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

          if btof_pdg[i] == 11 or btof_pdg[i] == -11:
              e_calc_mass_on_btof.append(btof_calc_mass[i])

      pi_eff_btof = pi_mass_count_btof / len(pi_calc_mass_on_btof) if len(pi_calc_mass_on_btof) > 0 else 0
      pi_eff_btof_large_mergin = pi_mass_count_btof_large_mergin / len(pi_calc_mass_on_btof) if len(pi_calc_mass_on_btof) > 0 else 0
      k_eff_btof = k_mass_count_btof / len(k_calc_mass_on_btof) if len(k_calc_mass_on_btof) > 0 else 0
      k_eff_btof_large_mergin = k_mass_count_btof_large_mergin / len(k_calc_mass_on_btof) if len(k_calc_mass_on_btof) > 0 else 0
      p_eff_btof = p_mass_count_btof / len(p_calc_mass_on_btof) if len(p_calc_mass_on_btof) > 0 else 0
      p_eff_btof_large_mergin = p_mass_count_btof_large_mergin / len(p_calc_mass_on_btof) if len(p_calc_mass_on_btof) > 0 else 0

      if plot_verbose:
          self.tof_pid_performance_plotter.plot_tof_pid_reconstruction_mass(
              pi_calc_mass_on_btof,
              k_calc_mass_on_btof,
              p_calc_mass_on_btof,
              e_calc_mass_on_btof,
              track_momentums_on_btof,
              btof_beta_inversees,
              track_momentums_pi_on_btof,
              track_momentums_k_on_btof,
              track_momentums_p_on_btof,
              track_momentums_e_on_btof,
              btof_pi_beta_inversees,
              btof_k_beta_inversees,
              btof_p_beta_inversees,
              btof_e_beta_inversees
          )

      btof_calc_mass = np.array(btof_calc_mass)
      btof_pdg       = np.array(btof_pdg)      
      track_momentums_on_btof = np.array(track_momentums_on_btof)
      track_momentums_transverse_on_btof = np.array(track_momentums_transverse_on_btof)

      return btof_calc_mass, btof_pdg, track_momentums_on_btof, track_momentums_transverse_on_btof
  
  def process_separation_power_vs_momentum(
        self,
        btof_calc_mass: np.ndarray,
        btof_pdg: np.ndarray,
        track_momentums_on_btof: np.ndarray,
        track_momentums_transverse_on_btof: np.ndarray,
        nbins: int = 35,
        momentum_range: tuple = (0, 3.5),
        plot_verbose: bool = False
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

            if self.rootfile:
                hist_pi.Write()
                hist_k.Write()
                f_pi.Write()
                f_k.Write()

        separation_list = np.array(separation_list, dtype=object)
        valid_mask = (separation_list != None)
        valid_sep = separation_list[valid_mask].astype(float)
        valid_bin_center = bin_centers[valid_mask]

        if plot_verbose:
            self.tof_pid_performance_plotter.plot_separation_power_vs_momentum(
                btof_calc_mass,
                btof_pdg,
                track_momentums_on_btof,
                track_momentums_transverse_on_btof
            )

        return valid_bin_center, valid_sep
  
  def process_purity_vs_momentum(
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
        rootfile=None,
        plot_verbose: bool = False
    ):
        """
        With the mass btof_calc_mass calculated by BTOF,
        PDG (btof_pdg), we plot the Efficiency (recognition rate) for each momentum.

        - For each of π, K, and p
          (A) Conventional efficiency (normal)
               eff_pi = (#(true π enters π window)) / (#(true π))
          (B) unique efficiency
               - Denominator is limited to “the number of events that do not overlap with other windows
               - The numerator is further limited to “the number of events in that window that are in your window”.
            This eliminates duplicate events from the denominator as well, so unique may be larger.
        """

        #--------------------------------
        # 1) mask each particle
        #--------------------------------
        pi_mask = (btof_pdg ==  211) | (btof_pdg == -211)
        k_mask  = (btof_pdg ==  321) | (btof_pdg == -321)
        p_mask  = (btof_pdg == 2212) | (btof_pdg == -2212)

        pi_mass_all = btof_calc_mass[pi_mask]
        pi_mom_all  = track_momentums_on_btof[pi_mask]

        k_mass_all  = btof_calc_mass[k_mask]
        k_mom_all   = track_momentums_on_btof[k_mask]

        p_mass_all  = btof_calc_mass[p_mask]
        p_mom_all   = track_momentums_on_btof[p_mask]

        # define momentum bins
        p_bins      = np.linspace(momentum_range[0], momentum_range[1], nbins+1)
        bin_centers = 0.5 * (p_bins[:-1] + p_bins[1:])

        #--------------------------------
        # 2) array for each particle
        #--------------------------------
        pi_mass_count_list_normal    = []
        pi_mass_correct_list_normal  = []
        pi_mass_count_list_unique    = []
        pi_mass_correct_list_unique  = []

        k_mass_count_list_normal     = []
        k_mass_correct_list_normal   = []
        k_mass_count_list_unique     = []
        k_mass_correct_list_unique   = []

        p_mass_count_list_normal     = []
        p_mass_correct_list_normal   = []
        p_mass_count_list_unique     = []
        p_mass_correct_list_unique   = []

        PI_MASS     = 139.57039
        KAON_MASS   = 493.677
        PROTON_MASS = 938.272

        #--------------------------------
        # 3) loop over bins
        #--------------------------------
        for i in range(nbins):
            p_low  = p_bins[i]
            p_high = p_bins[i+1]

            pi_in_bin = pi_mass_all[(pi_mom_all >= p_low) & (pi_mom_all < p_high)]
            k_in_bin  = k_mass_all[(k_mom_all  >= p_low) & (k_mom_all  < p_high)]
            p_in_bin  = p_mass_all[(p_mom_all  >= p_low) & (p_mom_all  < p_high)]

            # initialize
            pi_count_normal     = 0
            pi_correct_normal   = 0
            pi_count_unique     = 0
            pi_correct_unique   = 0

            k_count_normal      = 0
            k_correct_normal    = 0
            k_count_unique      = 0
            k_correct_unique    = 0

            p_count_normal      = 0
            p_correct_normal    = 0
            p_count_unique      = 0
            p_correct_unique    = 0

            #=== π ===
            for val in pi_in_bin:
                pi_count_normal += 1  
                diff_pi = abs(val - PI_MASS)
                diff_k  = abs(val - KAON_MASS)
                diff_p  = abs(val - PROTON_MASS)

                if diff_pi < MERGIN_PI:
                    pi_correct_normal += 1

                # -- unique --
                # Denominator: “not in any other window” → not(diff_k < MERGIN_K) & not(diff_p < MERGIN_P)
                # Numerator: “in your window” → diff_pi < MERGIN_PI

                is_k_window = (diff_k < MERGIN_K)
                is_p_window = (diff_p < MERGIN_P)

                if (not is_k_window) and (not is_p_window):
                    pi_count_unique += 1
                    if diff_pi < MERGIN_PI:
                        pi_correct_unique += 1

            #=== K ===
            for val in k_in_bin:
                k_count_normal += 1
                diff_pi = abs(val - PI_MASS)
                diff_k  = abs(val - KAON_MASS)
                diff_p  = abs(val - PROTON_MASS)

                if diff_k < MERGIN_K:
                    k_correct_normal += 1

                is_pi_window = (diff_pi < MERGIN_PI)
                is_p_window  = (diff_p < MERGIN_P)

                if (not is_pi_window) and (not is_p_window):
                    k_count_unique += 1
                    if diff_k < MERGIN_K:
                        k_correct_unique += 1

            #=== p ===
            for val in p_in_bin:
                p_count_normal += 1
                diff_pi = abs(val - PI_MASS)
                diff_k  = abs(val - KAON_MASS)
                diff_p_ = abs(val - PROTON_MASS)

                if diff_p_ < MERGIN_P:
                    p_correct_normal += 1

                is_pi_window = (diff_pi < MERGIN_PI)
                is_k_window  = (diff_k < MERGIN_K)
                if (not is_pi_window) and (not is_k_window):
                    p_count_unique += 1
                    if diff_p_ < MERGIN_P:
                        p_correct_unique += 1

            pi_mass_count_list_normal.append(pi_count_normal)
            pi_mass_correct_list_normal.append(pi_correct_normal)
            pi_mass_count_list_unique.append(pi_count_unique)
            pi_mass_correct_list_unique.append(pi_correct_unique)

            k_mass_count_list_normal.append(k_count_normal)
            k_mass_correct_list_normal.append(k_correct_normal)
            k_mass_count_list_unique.append(k_count_unique)
            k_mass_correct_list_unique.append(k_correct_unique)

            p_mass_count_list_normal.append(p_count_normal)
            p_mass_correct_list_normal.append(p_correct_normal)
            p_mass_count_list_unique.append(p_count_unique)
            p_mass_correct_list_unique.append(p_correct_unique)

        #-------------------------------------------------
        # 4) Convert array to numpy & calculate efficiency
        #-------------------------------------------------
        pi_mass_count_list_normal    = np.array(pi_mass_count_list_normal,    dtype=float)
        pi_mass_correct_list_normal  = np.array(pi_mass_correct_list_normal,  dtype=float)
        k_mass_count_list_normal     = np.array(k_mass_count_list_normal,     dtype=float)
        k_mass_correct_list_normal   = np.array(k_mass_correct_list_normal,   dtype=float)
        p_mass_count_list_normal     = np.array(p_mass_count_list_normal,     dtype=float)
        p_mass_correct_list_normal   = np.array(p_mass_correct_list_normal,   dtype=float)

        pi_mass_count_list_unique    = np.array(pi_mass_count_list_unique,    dtype=float)
        pi_mass_correct_list_unique  = np.array(pi_mass_correct_list_unique,  dtype=float)
        k_mass_count_list_unique     = np.array(k_mass_count_list_unique,     dtype=float)
        k_mass_correct_list_unique   = np.array(k_mass_correct_list_unique,   dtype=float)
        p_mass_count_list_unique     = np.array(p_mass_count_list_unique,     dtype=float)
        p_mass_correct_list_unique   = np.array(p_mass_correct_list_unique,   dtype=float)

        pi_eff_normal = np.divide(
            pi_mass_correct_list_normal, pi_mass_count_list_normal,
            out=np.zeros_like(pi_mass_correct_list_normal),
            where=(pi_mass_count_list_normal>0)
        )
        k_eff_normal  = np.divide(
            k_mass_correct_list_normal,  k_mass_count_list_normal,
            out=np.zeros_like(k_mass_correct_list_normal),
            where=(k_mass_count_list_normal>0)
        )
        p_eff_normal  = np.divide(
            p_mass_correct_list_normal,  p_mass_count_list_normal,
            out=np.zeros_like(p_mass_correct_list_normal),
            where=(p_mass_count_list_normal>0)
        )

        pi_eff_unique = np.divide(
            pi_mass_correct_list_unique, pi_mass_count_list_unique,
            out=np.zeros_like(pi_mass_correct_list_unique),
            where=(pi_mass_count_list_unique>0)
        )
        k_eff_unique  = np.divide(
            k_mass_correct_list_unique,  k_mass_count_list_unique,
            out=np.zeros_like(k_mass_correct_list_unique),
            where=(k_mass_count_list_unique>0)
        )
        p_eff_unique  = np.divide(
            p_mass_correct_list_unique,  p_mass_count_list_unique,
            out=np.zeros_like(p_mass_correct_list_unique),
            where=(p_mass_count_list_unique>0)
        )

        # errorbar calculation (binomial error)
        pi_eff_err_normal = np.sqrt(pi_eff_normal*(1-pi_eff_normal)/pi_mass_count_list_normal, 
                                    where=(pi_mass_count_list_normal>0),
                                    out=np.zeros_like(pi_eff_normal))
        pi_eff_err_unique = np.sqrt(pi_eff_unique*(1-pi_eff_unique)/pi_mass_count_list_unique, 
                                    where=(pi_mass_count_list_unique>0),
                                    out=np.zeros_like(pi_eff_unique))

        k_eff_err_normal  = np.sqrt(k_eff_normal*(1-k_eff_normal)/k_mass_count_list_normal,
                                    where=(k_mass_count_list_normal>0),
                                    out=np.zeros_like(k_eff_normal))
        k_eff_err_unique  = np.sqrt(k_eff_unique*(1-k_eff_unique)/k_mass_count_list_unique,
                                    where=(k_mass_count_list_unique>0),
                                    out=np.zeros_like(k_eff_unique))

        p_eff_err_normal  = np.sqrt(p_eff_normal*(1-p_eff_normal)/p_mass_count_list_normal,
                                    where=(p_mass_count_list_normal>0),
                                    out=np.zeros_like(p_eff_normal))
        p_eff_err_unique  = np.sqrt(p_eff_unique*(1-p_eff_unique)/p_mass_count_list_unique,
                                    where=(p_mass_count_list_unique>0),
                                    out=np.zeros_like(p_eff_unique))

        print("[PID] π Normal  Eff:", pi_eff_normal)
        print("[PID] π Unique  Eff:", pi_eff_unique)
        print("[PID] K Normal  Eff:", k_eff_normal)
        print("[PID] K Unique  Eff:", k_eff_unique)
        print("[PID] p Normal  Eff:", p_eff_normal)
        print("[PID] p Unique  Eff:", p_eff_unique)

        if plot_verbose:
            self.tof_pid_performance_plotter.plot_purity_vs_momentum(
                bin_centers,
                pi_eff_normal, pi_eff_err_normal,
                k_eff_normal,  k_eff_err_normal,
                p_eff_normal,  p_eff_err_normal,
                pi_eff_unique, pi_eff_err_unique,
                k_eff_unique,  k_eff_err_unique,
                p_eff_unique,  p_eff_err_unique,
                name=name
            )