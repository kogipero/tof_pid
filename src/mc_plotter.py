import awkward as ak
import helper_functions as myfunc
import ROOT as r
import numpy as np

class MCPlotter:
    def __init__(self, rootfile, name: str):
        self.rootfile = rootfile
        self.name = name

    def plot_mc_info(self, mc_px, mc_py, mc_pz, mc_p, mc_p_theta, mc_p_phi, mc_PDG_ID, mc_vertex_x, mc_vertex_y, mc_vertex_z):
        """
        Plots MC information.

        Args:
            mc_px, mc_py, mc_pz: MC momenta.
            mc_p, mc_p_theta, mc_p_phi: MC derived quantities.
            mc_PDG_ID: MC particle types.
            mc_vertex_x, mc_vertex_y, mc_vertex_z: MC vertex positions.
        """
        print('Start plotting MC info')

        myfunc.make_histogram_root(ak.flatten(mc_px),
                        100,
                        hist_range=[-20, 20],
                        title='MC_momentum_x',
                        xlabel='px [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum_x',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(mc_py),
                        100,
                        hist_range=[-20, 20],
                        title='MC_momentum_y',
                        xlabel='py [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum_y',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(mc_pz),
                        100,
                        hist_range=[-200, 400],
                        title='MC_momentum_z',
                        xlabel='pz [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum_z',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(mc_p),
                        50,
                        hist_range=[0, 5],
                        title='MC_momentum',
                        xlabel='p [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(mc_p_theta),
                        50,
                        hist_range=[0, 3.2],
                        title='MC_momentum_theta',
                        xlabel='theta [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum_theta',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(mc_p_phi),
                        100,
                        hist_range=[-3.2, 3.2],
                        title='MC_momentum_phi',
                        xlabel='phi [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_momentum_phi',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(mc_PDG_ID),
                        500,
                        hist_range=[-250, 250],
                        title='MC_PDG_ID',
                        xlabel='PDG ID',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_PDG_ID',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(mc_vertex_x),
                        100,
                        hist_range=[-200, 200],
                        title='MC_vertex_pos_x',
                        xlabel='x [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_vertex_pos_x',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(mc_vertex_y),
                        100,
                        hist_range=[-200, 200],
                        title='MC_vertex_pos_y',
                        xlabel='y [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_vertex_pos_y',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(mc_vertex_z),
                        100,
                        hist_range=[-200, 200],
                        title='MC_vertex_pos_z',
                        xlabel='z [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/mc_vertex_pos_z',
                        rootfile=self.rootfile
                        )

        print('End plotting MC info')
