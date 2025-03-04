import uproot
import numpy as np
import ROOT as r
import argparse
import sys
import os

from typing import List, Tuple

current_dir = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from utility_function import load_yaml_config, load_tree_file, make_directory
from track_reader import TrackReader
from mc_reader import MCReader
from tof_reader import TOFReader
from matching_mc_and_track import MatchingMCAndTrack
from matching_tof_and_track import MatchingTOFAndTrackReader
from tof_pid_performance_manager import ToFPIDPerformanceManager

def analyze_separation_vs_vertex_z(
    name: str,
    rootfile: uproot.TTree,
    vertex_z_ranges: List[Tuple[float, float]],
    output_txt_name: str = 'pid_result.txt',
    output_efficiency_result_name: str = 'matching_result.txt'
):
    """
    Executes the full track and MC matching analysis with PID performance evaluation.
    """
    config = load_yaml_config('./config/execute_config.yaml')
    branch = load_yaml_config('./config/branch_name.yaml')
    file_path = load_yaml_config('./config/file_path.yaml')

    name = config['directory_name']
    VERBOSE = config['VERBOSE']
    PLOT_VERBOSE = config['PLOT_VERBOSE']
    SELECTED_EVENTS = config['SELECTED_EVENTS']        
    analysis_event_type = config['analysis_event_type']  
    matching_threshold_mc_track = config['matching_threshold_mc_track']     

    filename = file_path[analysis_event_type]['path']
    tree = load_tree_file(filename)

    track = TrackReader(dis_file=tree, branch=branch, name=name, rootfile=rootfile)

    # Retrieve track segments positions and momenta
    track_segments_x, track_segments_y, track_segments_z, _, _ = track.get_track_segments_pos(
        name=name, rootfile=rootfile, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
    )
    track_segments_px, track_segments_py, track_segments_pz, track_segments_p, tracksegments_pt, track_segments_p_theta, track_segments_p_phi, track_segment_pathlength = track.get_track_segments_momentum(
        name=name, rootfile=rootfile, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
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

    mc = MCReader(dis_file=tree, branch=branch, name=name, rootfile=rootfile)
    mc_px, mc_py, mc_pz, mc_p, mc_p_theta, mc_p_phi, mc_PDG_ID, mc_charge, mc_generatorStatus, mc_vertex_x, mc_vertex_y, mc_vertex_z = mc.get_mc_info(
        verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
    )

    matching = MatchingMCAndTrack(track=track, mc=mc, rootfile=rootfile, name=name)

    r_min_tracks, r_min_track_index = matching.get_segments_nearest_impact_point(
        all_tracks, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
    )

    all_segments_indices = matching.build_all_segments_indices(all_tracks)

    separation_results = []
    vertex_labels = []

    for (min_z, max_z) in vertex_z_ranges:
        print(f"Analyzing separation for vertex z range: ({min_z}, {max_z})")
        _, _, matched_pairs_on_btof, matched_pairs_on_etof = matching.match_track_to_mc(
            name=name,
            track_momentum=track_segments_p,
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
            all_segments_indices=all_segments_indices,
            threshold=matching_threshold_mc_track,
            rootfile=rootfile,
            vertex_z_min=min_z,
            vertex_z_max=max_z,
            verbose=VERBOSE,
            plot_verbose=PLOT_VERBOSE
        )

        tof = TOFReader(dis_file=tree, branch=branch, name=name, rootfile=rootfile)
        btof_phi_theta, ectof_phi_theta = tof.get_tof_info(
            name, SELECTED_EVENTS, rootfile=rootfile, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
        )

        match_track_and_tof = MatchingTOFAndTrackReader(dis_file=tree, branch=branch, name=name, rootfile=rootfile)
        matched_tracks_and_btof_phi_theta, matched_tracks_and_etof_phi_theta = match_track_and_tof.match_track_to_tof(
            name, matched_pairs_on_btof, matched_pairs_on_etof, btof_phi_theta, ectof_phi_theta,
            rootfile=rootfile, output_txt=output_efficiency_result_name, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
        )

        pid = ToFPIDPerformanceManager(dis_file=tree, branch=branch, name=name, rootfile=rootfile)
        btof_calc_mass, btof_pdg, track_momentums_on_btof, track_momentums_transverse_on_btof = pid.process_pid_performance_plot(
            name, matched_tracks_and_btof_phi_theta, matched_tracks_and_etof_phi_theta, rootfile=rootfile,
            output_txt_name=output_txt_name, plot_verbose=PLOT_VERBOSE
        )

        bin_center, separation_power = pid.process_separation_power_vs_momentum(
            btof_calc_mass=btof_calc_mass,
            btof_pdg=btof_pdg,
            track_momentums_on_btof=track_momentums_on_btof,
            track_momentums_transverse_on_btof=track_momentums_transverse_on_btof,
            plot_verbose=PLOT_VERBOSE
        )

        pid.process_purity_vs_momentum(
            btof_calc_mass=btof_calc_mass,
            btof_pdg=btof_pdg,
            track_momentums_on_btof=track_momentums_on_btof,
            track_momentums_transverse_on_btof=track_momentums_transverse_on_btof,
            plot_verbose=PLOT_VERBOSE
        )

        separation_results.append(separation_power)
        vertex_labels.append(f"({min_z}, {max_z})")

    for label, separation_power in zip(vertex_labels, separation_results):
        print(f"Separation power for vertex z range {label}: {separation_power}")

    # Plot Separation Power vs Momentum
    c = r.TCanvas("Separation Power vs Momentum vertex comparison", "Separation Power vs Momentum", 800, 600)
    mg = r.TMultiGraph()

    colors = [r.kRed, r.kBlue, r.kGreen+2, r.kMagenta, r.kOrange+1]
    markers = [20, 21, 22]

    for i, (sep_list, label) in enumerate(zip(separation_results, vertex_labels)):
        mom_arr = np.array(bin_center)
        sep_arr = np.array(sep_list)
        g = r.TGraph(len(mom_arr), mom_arr, sep_arr)

        g.SetLineColor(colors[i % len(colors)])
        g.SetMarkerColor(colors[i % len(colors)])
        g.SetMarkerStyle(markers[i % len(markers)])
        g.SetTitle(label)

        mg.Add(g, "P")

    mg.SetTitle("Separation_Power_vs_Momentum;Momentum_(GeV/c);Separation_Power")
    mg.GetYaxis().SetRangeUser(1e-3, 30)
    mg.GetXaxis().SetLimits(0, 3.5)
    mg.Draw("A")

    c.SetLogy()
    c.Update()
    c.BuildLegend()

    if rootfile:
        c.Write()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run track-MC matching and ToF PID analysis.")
    parser.add_argument("--rootfile", type=str, required=True, help="Output ROOT file name")
    args = parser.parse_args()

    config = load_yaml_config('./config/execute_config.yaml')
    name = config['directory_name']  
    directory_name = f'./out/{name}'
    make_directory(directory_name)

    rootfile_path = os.path.join(directory_name, args.rootfile)
    vertex_z_ranges = [(-5, 5), (-35, 35), (-55, 55)]

    rootfile = r.TFile(rootfile_path, "RECREATE")
    analyze_separation_vs_vertex_z("Analysis", rootfile, vertex_z_ranges)

    rootfile.Close()
    print("Analysis completed.")
