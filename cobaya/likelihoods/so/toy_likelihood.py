"""
.. module:: toy_likelihood

:Synopsis: Definition of simplistic likelihood for Simons Observatory
:Authors: Thibaut Louis and Xavier Garrido

"""
# Global
import os
import numpy as np

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError

class toy_likelihood(Likelihood):

    def initialize(self):
        if not self.data_folder:
            raise LoggedError(
                self.log, "No data folder has been set. Set the "
                "likelihood property 'data_folder'.")

        # State requisites to the theory code
        self.requested_cls = ["tt", "ee", "te"]
        self._prepare_data()
        pass

    def add_theory(self):
        # Same lmax for different cls
        self.l_maxs_cls = [self.lmax for i in self.requested_cls]
        self.theory.needs(Cl=dict(zip(self.requested_cls, self.l_maxs_cls)))

    def logp(self, **params_values):
        return 0.

    def _prepare_data(self):
        loc = self.data_folder
        self.Bbl = {}
        self.data_vec = {s:[] for s in self.requested_cls}
        self.spec_list = []

        # Internal function to check for file existence
        def _check_filename(fname):
            if not os.path.exists(fname):
                raise LoggedError(
                    self.log, "The {} file was not found within "
                    "{} directory.".format(os.path.basename(fname), self.data_folder))
            return fname

        # Load cross power spectra
        for exp in self.experiments:
            for exp1, freqs1 in exp.items():
                for id_f1, f1 in enumerate(freqs1):
                    for exp2, freqs2 in exp.items():
                        for id_f2, f2 in enumerate(freqs2):
                            if exp1 == exp2 and id_f1 > id_f2: continue

                            spec_name = "{}_{}x{}_{}".format(exp1, f1, exp2, f2)
                            file_name = "{}/Dl_{}".format(loc, spec_name)
                            file_name += "_{:05d}.dat".format(self.sim_id) \
                                if isinstance(self.sim_id, int) else ".dat"

                            l, ps = self._read_spectra(_check_filename(file_name))
                            for s in self.requested_cls:
                                self.Bbl[s, spec_name] = np.loadtxt(
                                    _check_filename("{}/Bbl_{}_{}.dat".format(loc, spec_name, s.upper())))
                                if s == "te":
                                    self.data_vec[s] = np.append(
                                        self.data_vec[s], (ps["te"]+ps["et"])/2)
                                else:
                                    self.data_vec[s] = np.append(self.data_vec[s], ps[s])
                            self.spec_list += [spec_name]

        # Read covariance matrix file
        cov_mat = np.loadtxt(_check_filename("{}/covariance.dat".format(loc)))

        # Set data given selection
        if self.select == "tt-te-ee":
            self.data_vec = np.concatenate([self.data_vec[s] for s in self.requested_cls])
        else:
            for count, s in enumerate(self.requested_cls):
                if select == s:
                    n_bins = int(cov_mat.shape[0])
                    cov_mat = cov_mat[count*n_bins//3:(count+1)*n_bins//3,
                                      count*n_bins//3:(count+1)*n_bins//3]
        # Store inverted covariance matrix
        self.inv_cov = np.linalg.inv(cov_mat)

    def _read_spectra(self, fname):
        data = np.loadtxt(fname)
        l = data[:, 0]
        spectra = ["tt","te","tb","et","bt","ee","eb","be","bb"]
        ps = {f: data[:, c+1] for c,f in enumerate(spectra)}
        return l, ps
