import numpy as np
import ROOT as r
import helper_functions as myfunc


class TOFPIDPerformancePlotter:
    def __init__(self, rootfile, name: str):
        self.rootfile = rootfile
        self.name = name

    def plot_tof_pid_performance(self, track_momentums_on_btof, track_momentum_on_ectof, btof_beta_inversees, btof_calc_mass):
        """
        Plots TOF PID performance.
        """
        print('Start plotting TOF PID performance')

        myfunc.make_histogram_root(
            track_momentums_on_btof,
                           100,
                           hist_range=[0, 5],
                        title='BTOF_Momentum_PID_Performance',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_momentum_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            track_momentum_on_ectof,
                           100,
                           hist_range=[0, 5],
                        title='ETOF_Momentum_PID_Performance',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/etof_momentum_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_beta_inversees,
                        100,
                        hist_range=[0.8, 1.8],
                        title='BTOF_Beta_Inverse_PID_Performance',
                        xlabel='Beta Inverse',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_beta_inverse_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_calc_mass,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF_Calculated_Mass',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_pid_performance',
                        rootfile=self.rootfile
        )

        print('End plotting TOF PID performance')

    def plot_tof_pid_reconstruction_mass(self, 
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
                                         ):
        """
        Plots TOF PID mass reconstruction.
        """
        print('Start plotting TOF PID mass reconstruction')

        myfunc.make_histogram_root(
            pi_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF_Calculated_Mass_for_Pi',
                        xlabel='Mass [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_pi_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            k_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF_Calculated_Mass_for_K',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_k_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            p_calc_mass_on_btof,
                        100,
                        hist_range=[200, 1200],
                        title='BTOF_Calculated_Mass_for_P',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_p_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            e_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF_Calculated_Mass_for_e',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_e_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_on_btof,
            100,
            [0, 3.5],
            btof_beta_inversees,
            100,
            [0.8, 3.5],
            title='BTOF_Momentum_vs_Beta_Inverse',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_on_btof,
            100,
            [0, 5],
            btof_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_pid_performance_diff_range',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_pi_on_btof,
            100,
            [0, 3.5],
            btof_pi_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse_for_Pi',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_pi_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_k_on_btof,
            100,
            [0, 3.5],
            btof_k_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse_for_K',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_k_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_p_on_btof,
            100,
            [0, 3.5],
            btof_p_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse_for_P',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_p_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_e_on_btof,
            100,
            [0, 3.5],
            btof_e_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse_for_e',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_e_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        print('End plotting TOF PID mass reconstruction')

    def plot_separation_power_vs_momentum(
        self,
        btof_calc_mass: np.ndarray,
        btof_pdg: np.ndarray,
        track_momentums_on_btof: np.ndarray,
        track_momentums_transverse_on_btof: np.ndarray,
        nbins: int = 35,
        momentum_range: tuple = (0, 3.5),
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

        gr = r.TGraph()
        gr.SetName("sep_power_vs_mom")
        gr.SetTitle("Separation Power vs Momentum;pt [GeV];Separation Power")
        idx = 0
        for bc, sep in zip(valid_bin_center, valid_sep):
            gr.SetPoint(idx, bc, sep)
            idx += 1

        if self.rootfile:
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

        if self.rootfile:
            c1.Write("canvas_sep_power_logy")  
            
        if self.rootfile:
            gr = r.TGraph()
            gr.SetName("sep_power_vs_mom")
            idx = 0
            for bc, sep in zip(valid_bin_center, valid_sep):
                gr.SetPoint(idx, bc, sep)
                idx+=1
            gr.Write()

        return valid_bin_center, valid_sep
    

    def plot_purity_vs_momentum(
        self,
        bin_centers: np.ndarray,
        pi_eff_normal: np.ndarray,
        pi_eff_err_normal: np.ndarray,
        pi_eff_unique: np.ndarray,
        pi_eff_err_unique: np.ndarray,
        k_eff_normal: np.ndarray,
        k_eff_err_normal: np.ndarray,
        k_eff_unique: np.ndarray,
        k_eff_err_unique: np.ndarray,
        p_eff_normal: np.ndarray,
        p_eff_err_normal: np.ndarray,
        p_eff_unique: np.ndarray,
        p_eff_err_unique: np.ndarray,
        momentum_range: tuple = (0, 3.5),
    ):
        """
        Plot the purity of each particle as a function of momentum.
        """

        gr_pi_normal  = r.TGraphErrors()
        gr_pi_unique  = r.TGraphErrors()
        gr_pi_normal.SetName("pi_eff_normal")
        gr_pi_normal.SetTitle("Pi Efficiency (Normal);p [GeV];Efficiency")
        gr_pi_unique.SetName("pi_eff_unique")
        gr_pi_unique.SetTitle("Pi Efficiency (Unique);p [GeV];Efficiency")

        for ibin, (bc, eff_n, err_n, eff_u, err_u) in enumerate(zip(
            bin_centers, pi_eff_normal, pi_eff_err_normal, pi_eff_unique, pi_eff_err_unique
        )):
            gr_pi_normal.SetPoint(ibin, bc, eff_n)
            gr_pi_normal.SetPointError(ibin, 0, err_n)
            gr_pi_unique.SetPoint(ibin, bc, eff_u)
            gr_pi_unique.SetPointError(ibin, 0, err_u)

        gr_pi_normal.SetMarkerStyle(20)
        gr_pi_normal.SetMarkerColor(r.kRed)
        gr_pi_normal.SetLineColor(r.kRed)

        gr_pi_unique.SetMarkerStyle(21)
        gr_pi_unique.SetMarkerColor(r.kBlue)
        gr_pi_unique.SetLineColor(r.kBlue)

        c_pi = r.TCanvas("c_pi","Pi Efficiency",800,600)
        c_pi.Draw()
        frame_pi = c_pi.DrawFrame(0, 0, momentum_range[1], 1.05)
        frame_pi.GetXaxis().SetTitle("p [GeV]")
        frame_pi.GetYaxis().SetTitle("Efficiency")

        gr_pi_normal.Draw("P SAME")
        gr_pi_unique.Draw("P SAME")
        c_pi.BuildLegend()
        c_pi.Update()

        if self.rootfile:
            gr_pi_normal.Write()
            gr_pi_unique.Write()
            c_pi.Write("canvas_pi_eff")

        gr_k_normal  = r.TGraphErrors()
        gr_k_unique  = r.TGraphErrors()
        gr_k_normal.SetName("k_eff_normal")
        gr_k_normal.SetTitle("K Efficiency (Normal);p [GeV];Efficiency")
        gr_k_unique.SetName("k_eff_unique")
        gr_k_unique.SetTitle("K Efficiency (Unique);p [GeV];Efficiency")

        for ibin, (bc, eff_n, err_n, eff_u, err_u) in enumerate(zip(
            bin_centers, k_eff_normal, k_eff_err_normal, k_eff_unique, k_eff_err_unique
        )):
            gr_k_normal.SetPoint(ibin, bc, eff_n)
            gr_k_normal.SetPointError(ibin, 0, err_n)
            gr_k_unique.SetPoint(ibin, bc, eff_u)
            gr_k_unique.SetPointError(ibin, 0, err_u)

        gr_k_normal.SetMarkerStyle(20)
        gr_k_normal.SetMarkerColor(r.kGreen+2)
        gr_k_normal.SetLineColor(r.kGreen+2)

        gr_k_unique.SetMarkerStyle(21)
        gr_k_unique.SetMarkerColor(r.kOrange+1)
        gr_k_unique.SetLineColor(r.kOrange+1)

        c_k = r.TCanvas("c_k","K Efficiency",800,600)
        frame_k = c_k.DrawFrame(0,0,momentum_range[1],1.05)
        frame_k.GetXaxis().SetTitle("p [GeV]")
        frame_k.GetYaxis().SetTitle("Efficiency")
        gr_k_normal.Draw("P SAME")
        gr_k_unique.Draw("P SAME")
        c_k.BuildLegend()
        c_k.Update()

        if self.rootfile:
            gr_k_normal.Write()
            gr_k_unique.Write()
            c_k.Write("canvas_k_eff")

        gr_p_normal  = r.TGraphErrors()
        gr_p_unique  = r.TGraphErrors()
        gr_p_normal.SetName("p_eff_normal")
        gr_p_normal.SetTitle("Proton Efficiency (Normal);p [GeV];Efficiency")
        gr_p_unique.SetName("p_eff_unique")
        gr_p_unique.SetTitle("Proton Efficiency (Unique);p [GeV];Efficiency")

        for ibin, (bc, eff_n, err_n, eff_u, err_u) in enumerate(zip(
            bin_centers, p_eff_normal, p_eff_err_normal, p_eff_unique, p_eff_err_unique
        )):
            gr_p_normal.SetPoint(ibin, bc, eff_n)
            gr_p_normal.SetPointError(ibin, 0, err_n)
            gr_p_unique.SetPoint(ibin, bc, eff_u)
            gr_p_unique.SetPointError(ibin, 0, err_u)

        gr_p_normal.SetMarkerStyle(20)
        gr_p_normal.SetMarkerColor(r.kViolet)
        gr_p_normal.SetLineColor(r.kViolet)

        gr_p_unique.SetMarkerStyle(21)
        gr_p_unique.SetMarkerColor(r.kAzure+1)
        gr_p_unique.SetLineColor(r.kAzure+1)

        c_p = r.TCanvas("c_p","P Efficiency",800,600)
        frame_p = c_p.DrawFrame(0,0,momentum_range[1],1.05)
        frame_p.GetXaxis().SetTitle("p [GeV]")
        frame_p.GetYaxis().SetTitle("Efficiency")
        gr_p_normal.Draw("P SAME")
        gr_p_unique.Draw("P SAME")
        c_p.BuildLegend()
        c_p.Update()

        if self.rootfile:
            gr_p_normal.Write()
            gr_p_unique.Write()
            c_p.Write("canvas_p_eff")