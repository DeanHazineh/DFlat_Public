import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import tools.graphFunc as graphFunc
from util_library_lookup import *

listLibraryNames = ["Nanofins_U350nm_H600nm", "Nanocylinders_U180nm_H600nm", "Nanoellipse_U350nm_H600nm"]


class Nanofins_U350nm_H600nm:
    def __init__(self):
        __rawPath = "datasets_metasurface_cells/raw_meta_libraries/data_Nanofins_Unit350nm_Height600nm_EngineFDTD.mat"
        data = scipy.io.loadmat(__rawPath)

        # Phase and transmission has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        self.phase = data["phase"]
        self.transmission = data["transmission"]

        # the input parameters (lenx, leny, wavelength) are loaded in here
        # all parameters must be converted to meshgrid format for compatibility with model class
        # param1, param2, and param3 all have units of m
        param1 = data["lenx"]
        param2 = data["leny"]
        param3 = data["wavelength_m"].flatten()
        param1, param2, param3 = np.meshgrid(param1, param2, param3)
        self.params = [param1, param2, param3]

        # These are the min-max for the FDTD gridsweep
        self.__param1Limits = [60e-9, 300e-9]
        self.__param2Limits = [60e-9, 300e-9]
        self.__param3Limits = [310e-9, 750e-9]

    def plotLibrary(self, idx):
        # This function is added so the user may get familiar with the data themselves
        # and poke around the visualization as a test

        lx = self.params[0][0, :, 0] * 1e9
        ly = self.params[1][:, 0, 0] * 1e9
        wavelength = self.params[2][0, 0, :] * 1e9

        fig = plt.figure(figsize=(22, 18))
        axisList = graphFunc.addAxis(fig, 2, 2)
        phix = axisList[0].imshow(self.phase[0, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)))
        phiy = axisList[1].imshow(self.phase[1, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)))
        tx = axisList[2].imshow(
            self.transmission[0, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)), vmin=0, vmax=1
        )
        ty = axisList[3].imshow(
            self.transmission[1, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)), vmin=0, vmax=1
        )

        graphFunc.formatPlots(fig, axisList[0], phix, "len x (nm)", "len y (nm)", "phase x", addcolorbar=True)
        graphFunc.formatPlots(fig, axisList[1], phiy, "len x (nm)", "len y (nm)", "phase y", addcolorbar=True)
        graphFunc.formatPlots(fig, axisList[2], tx, "len x (nm)", "len y (nm)", "Trans x", addcolorbar=True)
        graphFunc.formatPlots(fig, axisList[3], ty, "len x (nm)", "len y (nm)", "Trans y", addcolorbar=True)

        plt.show()

    def optical_response_to_param(self, trans_asList, phase_asList, wavelength_asList, reshape):
        ### Run input assertions
        length = len(wavelength_asList)

        if not all([trans.shape[0] == 2 for trans in trans_asList]):
            raise ValueError(
                "optical_response_to_param: All transmission profiles in the list must be a stack of two profiles, (2, Ny, Nx)"
            )

        if not all([phase.shape[0] == 2 for phase in phase_asList]):
            raise ValueError(
                "optical_response_to_param: All phase profiles in the list must be a stack of two profiles, (2, Ny, Nx)"
            )

        if not all(type(input) is list for input in [trans_asList, phase_asList, wavelength_asList]):
            raise TypeError("optical_response_to_param: trans, phase, and wavelength must all be passed in as lists")

        if not all(len(lst) == length for lst in [trans_asList, phase_asList]):
            raise ValueError("optical_response_to_param: All lists must be the same length")

        ### Get the library data
        phaseTable = self.phase
        transTable = np.clip(self.transmission, 0.0, 1.0)
        lx = self.params[0][:, :, 0].flatten()
        ly = self.params[1][:, :, 0].flatten()
        wavelength = self.params[2][0, 0, :]

        ### Assemble metasurfaces
        shape_Vector = []
        shape_Vector_norm = []
        for i in range(length):
            use_wavelength = wavelength_asList[i]
            ms_trans = trans_asList[i]
            ms_phase = phase_asList[i]
            initial_shape = ms_trans[0:1].shape

            design_lx, design_ly = lookup_D2_pol2(
                phaseTable, transTable, lx, ly, wavelength, use_wavelength, ms_trans, ms_phase
            )

            # Define a normalized shape vector for convenience
            norm_design_lx = (design_lx - self.__param1Limits[0]) / (self.__param1Limits[1] - self.__param1Limits[0])
            norm_design_ly = (design_ly - self.__param2Limits[0]) / (self.__param2Limits[1] - self.__param2Limits[0])

            if reshape:
                shape_Vector.append(
                    np.vstack((np.reshape(design_lx, initial_shape), np.reshape(design_ly, initial_shape),))
                )

                shape_Vector_norm.append(
                    np.vstack((np.reshape(norm_design_lx, initial_shape), np.reshape(norm_design_ly, initial_shape),))
                )
            else:
                shape_Vector.append(np.hstack((np.expand_dims(design_lx, -1), np.expand_dims(design_ly, -1))))
                shape_Vector_norm.append(
                    np.hstack((np.expand_dims(norm_design_lx, -1), np.expand_dims(norm_design_ly, -1)))
                )

        return shape_Vector, shape_Vector_norm


class Nanocylinders_U180nm_H600nm:
    def __init__(self):

        __rawPath = (
            "datasets_metasurface_cells/raw_meta_libraries/data_Nanocylinders_Unit180nm_Height600nm_EngineFDTD.mat"
        )
        data = scipy.io.loadmat(__rawPath)

        # Phase and transmission has shape [wavelength=441, lenr=191]
        self.phase = data["phase"]
        self.transmission = data["transmission"]

        # the input parameters (lenr, wavelength) are loaded in here
        # all parameters must be converted to meshgrid format for compatibility with model class
        # param1, param2, all have units of m
        param1, param2 = np.meshgrid(data["radius_m"], data["wavelength_m"])
        self.params = [param1, param2]

        # These are the min-max for the FDTD gridsweep
        self.__param1Limits = [30e-9, 150e-9]
        self.__param2Limits = [310e-9, 750e-9]

    def plotLibrary(self):
        # This function is added so the user may get familiar with the data themselves
        # and poke around the visualization as a test
        lr = self.params[0][0, :] * 1e9
        wl = self.params[1][:, 0] * 1e9

        fig = plt.figure(figsize=(20, 15))
        axisList = graphFunc.addAxis(fig, 1, 2)
        phi = axisList[0].imshow(self.phase[:, :], extent=(min(lr), max(lr), max(wl), min(wl)))
        tt = axisList[1].imshow(self.transmission[:, :], extent=(min(lr), max(lr), max(wl), min(wl)), vmin=0, vmax=1)
        graphFunc.formatPlots(fig, axisList[0], phi, "len r (nm)", "wavelength (nm)", "phase", addcolorbar=True)
        graphFunc.formatPlots(fig, axisList[1], tt, "len r (nm)", "wavelength (nm)", "transmission", addcolorbar=True)

        plt.show()

    def optical_response_to_param(self, trans_asList, phase_asList, wavelength_asList, reshape):
        ### Run input assertions
        length = len(wavelength_asList)

        if not all([trans.shape[0] == 1 for trans in trans_asList]):
            raise ValueError(
                "optical_response_to_param: All transmission profiles in the list must be a single transmission profile, (1, Ny, Nx)"
            )
        if not all([phase.shape[0] == 1 for phase in phase_asList]):
            raise ValueError(
                "optical_response_to_param: All phase profiles in the list must be a single phase profile, (1, Ny, Nx)"
            )
        if (
            (type(trans_asList) is not list)
            or (type(phase_asList) is not list)
            or (type(wavelength_asList) is not list)
        ):
            raise TypeError("optical_response_to_param: trans, phase, and wavelength must all be passed in as lists")
        if not all(len(lst) == length for lst in [trans_asList, phase_asList]):
            raise ValueError("optical_response_to_param: All lists must be the same length")

        ### Get the library data
        phaseTable = self.phase
        transTable = np.clip(self.transmission, 0.0, 1.0)
        radius = self.params[0][0, :].flatten()
        wavelength = self.params[1][:, 0]

        ### Assemble metasurfaces
        shape_Vector = []
        shape_Vector_norm = []
        for i in range(length):
            use_wavelength = wavelength_asList[i]
            ms_trans = trans_asList[i]
            ms_phase = phase_asList[i]
            initial_shape = ms_trans[0:1].shape

            design_radius = lookup_D1_pol1(
                phaseTable, transTable, radius, wavelength, use_wavelength, ms_trans, ms_phase
            )
            norm_design_radius = (design_radius - self.__param1Limits[0]) / (
                self.__param1Limits[1] - self.__param1Limits[0]
            )

            if reshape:
                shape_Vector.append(np.reshape(design_radius, initial_shape))
                shape_Vector_norm.append(np.reshape(norm_design_radius, initial_shape))
            else:
                shape_Vector.append(np.expand_dims(design_radius, -1))
                shape_Vector_norm.append(np.expand_dims(norm_design_radius, -1))

        return shape_Vector, shape_Vector_norm


class Nanoellipse_U350nm_H600nm:
    def __init__(self):
        __rawPath = (
            "datasets_metasurface_cells/raw_meta_libraries/data_NanoEllipse_Unit350nm_Height600nm_EngineFDTD.mat"
        )
        data = scipy.io.loadmat(__rawPath)

        # Phase and transmission has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        self.phase = data["phase"]
        self.transmission = data["transmission"]

        # the input parameters (lenx, leny, wavelength) are loaded here
        # all parameters must be converted to meshgrid format for compatibility with model class
        param1 = data["lenx"]
        param2 = data["leny"]
        param3 = data["wavelength_m"].flatten()
        param1, param2, param3 = np.meshgrid(param1, param2, param3)
        self.params = [param1, param2, param3]

        # These are the min-max for the FDTD gridsweep
        self.__param1Limits = [60e-9, 300e-9]
        self.__param2Limits = [60e-9, 300e-9]
        self.__param3Limits = [310e-9, 750e-9]

    def plotLibrary(self, idx):
        # This function is added so the user may get familiar with the data themselves
        # and poke around the visualization as a test

        lx = self.params[0][0, :, 0] * 1e9
        ly = self.params[1][:, 0, 0] * 1e9
        wavelength = self.params[2][0, 0, :] * 1e9

        fig = plt.figure(figsize=(22, 18))
        axisList = graphFunc.addAxis(fig, 2, 2)
        phix = axisList[0].imshow(self.phase[0, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)))
        phiy = axisList[1].imshow(self.phase[1, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)))
        tx = axisList[2].imshow(
            self.transmission[0, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)), vmin=0, vmax=1
        )
        ty = axisList[3].imshow(
            self.transmission[1, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)), vmin=0, vmax=1
        )

        graphFunc.formatPlots(fig, axisList[0], phix, "len x (nm)", "len y (nm)", "phase x", addcolorbar=True)
        graphFunc.formatPlots(fig, axisList[1], phiy, "len x (nm)", "len y (nm)", "phase y", addcolorbar=True)
        graphFunc.formatPlots(fig, axisList[2], tx, "len x (nm)", "len y (nm)", "Trans x", addcolorbar=True)
        graphFunc.formatPlots(fig, axisList[3], ty, "len x (nm)", "len y (nm)", "Trans y", addcolorbar=True)

        plt.show()

    def optical_response_to_param(self, trans_asList, phase_asList, wavelength_asList, reshape):
        ### Run input assertions
        length = len(wavelength_asList)

        if not all([trans.shape[0] == 2 for trans in trans_asList]):
            raise ValueError(
                "optical_response_to_param: All transmission profiles in the list must be a stack of two profiles, (2, Ny, Nx)"
            )

        if not all([phase.shape[0] == 2 for phase in phase_asList]):
            raise ValueError(
                "optical_response_to_param: All phase profiles in the list must be a stack of two profiles, (2, Ny, Nx)"
            )

        if not all(type(input) is list for input in [trans_asList, phase_asList, wavelength_asList]):
            raise TypeError("optical_response_to_param: trans, phase, and wavelength must all be passed in as lists")

        if not all(len(lst) == length for lst in [trans_asList, phase_asList]):
            raise ValueError("optical_response_to_param: All lists must be the same length")

        ### Get the library data
        phaseTable = self.phase
        transTable = np.clip(self.transmission, 0.0, 1.0)
        lx = self.params[0][:, :, 0].flatten()
        ly = self.params[1][:, :, 0].flatten()
        wavelength = self.params[2][0, 0, :]

        ### Assemble metasurfaces
        shape_Vector = []
        shape_Vector_norm = []
        for i in range(length):
            use_wavelength = wavelength_asList[i]
            ms_trans = trans_asList[i]
            ms_phase = phase_asList[i]
            initial_shape = ms_trans[0:1].shape

            design_lx, design_ly = lookup_D2_pol2(
                phaseTable, transTable, lx, ly, wavelength, use_wavelength, ms_trans, ms_phase
            )

            # Define a normalized shape vector for convenience
            norm_design_lx = (design_lx - self.__param1Limits[0]) / (self.__param1Limits[1] - self.__param1Limits[0])
            norm_design_ly = (design_ly - self.__param2Limits[0]) / (self.__param2Limits[1] - self.__param2Limits[0])

            if reshape:
                shape_Vector.append(
                    np.vstack((np.reshape(design_lx, initial_shape), np.reshape(design_ly, initial_shape),))
                )

                shape_Vector_norm.append(
                    np.vstack((np.reshape(norm_design_lx, initial_shape), np.reshape(norm_design_ly, initial_shape),))
                )
            else:
                shape_Vector.append(np.hstack((np.expand_dims(design_lx, -1), np.expand_dims(design_ly, -1))))
                shape_Vector_norm.append(
                    np.hstack((np.expand_dims(norm_design_lx, -1), np.expand_dims(norm_design_ly, -1)))
                )

        return shape_Vector, shape_Vector_norm

