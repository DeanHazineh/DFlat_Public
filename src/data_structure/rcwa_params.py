from copy import deepcopy
import numpy as np
import tensorflow as tf

from physical_optical_layer.core.ms_parameterization import ALLOWED_PARAMETERIZATION_TYPE, CELL_SHAPE_DEGREE
from physical_optical_layer.core.material_utils import MATERIAL_DICT, get_material_index

ALL_MANDATORY_KEYS = [
    "wavelength_set_m",
    "thetas",
    "phis",
    "pte",
    "ptm",
    "pixelsX",
    "pixelsY",
    "PQ",
    "Lx",
    "Ly",
    "L",
    "Lay_mat",
    "material_dielectric",
    "er1",
    "er2",
    "Nx",
    "Ny",
    "parameterization_type",
    "batch_wavelength_dim",
]

ALL_OPTIONAL_KEYS = {
    "force_span_limits": None,
    "ur1": 1.0,
    "ur2": 1.0,
    "urd": 1.0,
    "urs": 1.0,
    "eps": 1e-6,
    "dtype": tf.float32,
    "cdtype": tf.complex64,
}

ADDED_KEYS_PASS = ["shape_vect_size", "span_limits"]

DEFAULT_SPAN_LIMITS = {
    "rectangular_resonators": {"min": 0.10, "max": 0.80},
    "elliptical_resonators": {"min": 0.10, "max": 0.80},
    "cylindrical_nanoposts": {"min": 0.10, "max": 0.80},
    "coupled_elliptical_resonators": {"min": 0.05, "max": 0.20},
    "coupled_rectangular_resonators": {"min": 0.05, "max": 0.20},
}


class rcwa_params(dict):
    """Parameters object (dictionary) used for the rcwa_layer selections of the physical optical model. 
    Defines the simulation settings
    """

    def __init__(self, input_dict: dict, bare=False):
        """Parameters object (dictionary) used for the rcwa_layer selections of the physical optical model. 
        Defines the simulation settings

        Args:
            dict (dict): Input dictionary with the required keys:\\
            `"wavelength_set_m"`: List of wavelengths of length batch_size and type float, to simulate the optical 
                response of cells\\
            `"thetas"`: List of polar angles (degrees) of length batch_size and type float for incident light\\
            `"phis"`: List of azimuthal angles (degrees) of length batch_size and type float for incident light,\\
            `"pte"`: List of TE polarization component magnitudes, of length batch_size and type float,\\
            `"ptm"`: Liist of TM polarization component magnitudes, of length batch_size and type float,\\
            `"pixelsX"`: Integer number of cells in the x-direction,\\
            `"pixelsY"`: Integer number of cells in the y-direction,\\
            `"PQ"`: List of type int and length 2 specifying the number of Fourier harmonics in the x and y-direction (Must be odd values),\\
            `"Lx"`: Float specifying all cell widths in the x-directions (in meters),\\
            `"Ly"`: Float specifing al cell widths in the y-direction (in meters),\\
            `"L"`: List of float specifying the layer thickness (in meters),\\
            `"Lay_mat"`: List, same length as L, containing the embedding medium in each layer. Each layer specifier may be a string containing the
            material name or the relative elevtric permittivity as a complex float. 
            `"material_dielectric"': String specifying dielectric material (see technical 
                documents for list of options). A single complex float may be passed instead.\\
            `"Nx"`: Integer number of sample points for a discretized cell, in the x-direction,\\
            `"Ny"`: Integer number of sample points for a discretized clel, in the y-direction,\\
            `"parameterization_type"`: String defining the shape type of all cells. See technical documentation
                 for valid implementations\\
            (optional) `force_span_limits`: Dict with keys "min" and "max", denoting minimum and maximum span for 
                structures placed in the cell; Defaults to parameterization_type specific bounds.\\
            (optional) `ur1`: Magnetic permeability in the reflected region of space; Defaults to 1.0\\
            (optional) `er1`: Electric permittivity in the reflected region of space; Defaults to 1.0\\
            (optional) `ur2`: Magnetic permeability in the transmitted region of space; Defaults to 1.0\\
            (optional) `er2`: Electric permittivity in the transmitted region of space; Defaults to 1.0\\
            (optional) `urd`: Magnetic permeability of the dielectric structures; Defaults to 1.0\\
            (optional) `urs`: Magnetic permeability of the substrate; Defaults to 1.0\\
        """
        # Check input conditions
        self.__dict__ = deepcopy(input_dict)
        self.__check_mandatory_keys()
        self.__check_optional_keys()
        self.__check_unknown_keys()
        self.__check_material_entry()

        if not bare:
            self.__check_parameterization_type()
            self.__get_param_shape()  # add shape vect size to dictionary keys
            self.__regularize_span_limits()

        # Add required simulation keys
        if not input_dict["batch_wavelength_dim"]:
            self.__add_sim_keys(input_dict)

        return

    def __check_mandatory_keys(self):
        for exclusive_key in ALL_MANDATORY_KEYS:
            if isinstance(exclusive_key, list):
                if not any(check_key in self.__dict__.keys() for check_key in exclusive_key):
                    print(exclusive_key)
                    raise KeyError("\n rcwa_params: one of the above must be included")
            else:
                if not (exclusive_key in self.__dict__.keys()):
                    raise KeyError(
                        "rcwa_params: Missing mandatory parameter option for simulation settings: " + exclusive_key
                    )
        return

    def __check_optional_keys(self):
        for optional_key in ALL_OPTIONAL_KEYS.keys():
            if not (optional_key in self.__dict__.keys()):
                self.__dict__[optional_key] = ALL_OPTIONAL_KEYS[optional_key]

        return

    def __check_unknown_keys(self):
        # Flatten the mandatory key list given nested optionals
        flattened_all_mandatory_keys = []
        for item in ALL_MANDATORY_KEYS:
            if isinstance(item, list):
                [flattened_all_mandatory_keys.append(subitem) for subitem in item]
            else:
                flattened_all_mandatory_keys.append(item)

        # Check unknown keys against all possible keys
        for providedKey in self.__dict__.keys():
            if not providedKey in (flattened_all_mandatory_keys + list(ALL_OPTIONAL_KEYS.keys()) + ADDED_KEYS_PASS):
                raise KeyError("params: unknown parameter key/setting inputed: " + providedKey)

        return

    def __check_material_entry(self):
        # User must have passed in both Lay_mat and material_dielectric

        ### process Lay_mat
        Lay_mat = self.__dict__["Lay_mat"]

        if not isinstance(Lay_mat, list):
            raise TypeError("Error in rcwa_params: Lay_mat must be a list of strings")
        if len(Lay_mat) != len(self.__dict__["L"]):
            raise ValueError("Error in rcwa_param: list of layer materials must be same length as L")

        # Check validity of each layers entry
        for lay_entry in Lay_mat:

            if isinstance(lay_entry, str):
                if not (lay_entry in MATERIAL_DICT.keys()):
                    print(MATERIAL_DICT.keys())
                    raise ValueError("Error in rcwa_params: 'material_dielectric must be one from the above")

            elif isinstance(lay_entry, complex):
                continue

            else:
                raise TypeError(
                    "Layer Material entries must be either string containing the material name or a complex value"
                )

        ### Check dielectric specifier
        material_dielectric = self.__dict__["material_dielectric"]
        if isinstance(material_dielectric, str):
            if not (material_dielectric in MATERIAL_DICT.keys()):
                print(MATERIAL_DICT.keys())
                raise ValueError("Error in rcwa_params: 'material_dielectric must be one from the above")

        elif not isinstance(material_dielectric, complex):
            raise TypeError("Error in rcwa_params: material_dielectric must be string name or complex float")

        ### Check entry for transmission and reflection layers
        er1 = self.__dict__["er1"]
        er2 = self.__dict__["er2"]
        if isinstance(er1, str):
            if not (er1 in MATERIAL_DICT.keys()):
                print(MATERIAL_DICT.keys())
                raise ValueError("Error in rcwa_params: 'er1 must be string from the above")
        elif not isinstance(er1, complex):
            raise TypeError("er1 must be either string containing the material name or a complex value")

        if isinstance(er2, str):
            if not (er2 in MATERIAL_DICT.keys()):
                print(MATERIAL_DICT.keys())
                raise ValueError("Error in rcwa_params: 'er2 must be string from the above")
        elif not isinstance(er2, complex):
            raise TypeError("er2 must be either string containing the material name or a complex value")

        return

    def __check_parameterization_type(self):
        if not (self.__dict__["parameterization_type"] in ALLOWED_PARAMETERIZATION_TYPE.keys()):
            raise ValueError("Error in rcwa_params: parameterization_type not one of the allowed options")

        return

    def __get_param_shape(self):
        parameterization_type = self.__dict__["parameterization_type"]
        cell_shape_degree = CELL_SHAPE_DEGREE[parameterization_type]
        pixelsX = self.__dict__["pixelsX"]
        pixelsY = self.__dict__["pixelsY"]

        shape_vect_size = [cell_shape_degree[0], pixelsX, pixelsY, cell_shape_degree[1]]
        self.__dict__["shape_vect_size"] = shape_vect_size

        return

    def __regularize_span_limits(self):
        if self.__dict__["force_span_limits"] == None:
            # set latent limits to the default values
            self.__dict__["span_limits"] = DEFAULT_SPAN_LIMITS[self.__dict__["parameterization_type"]]
        else:
            # set latent limits to the users
            force_span = self.__dict__["force_span_limits"]
            self.__check_force_span_simple(force_span)
            self.__dict__["span_limits"] = force_span

        return

    def __add_sim_keys(self, input_dict):
        ### unpack parameters
        batchSize = len(input_dict["wavelength_set_m"])
        pixelsX = input_dict["pixelsX"]
        pixelsY = input_dict["pixelsY"]
        wavelength_set_m = input_dict["wavelength_set_m"]
        thetas = input_dict["thetas"]
        phis = input_dict["phis"]
        pte = input_dict["pte"]
        ptm = input_dict["ptm"]
        L = input_dict["L"]
        dtype = input_dict["dtype"]
        cdtype = input_dict["cdtype"]

        ### Add required simulation Keys to the dictionary
        # Add the appropriate ["lay_eps_list"] and ["erd"] vectors
        self.__get_broadband_permittivity()
        self.__set_refl_trans_perm()

        self.__dict__["batchSize"] = batchSize
        self.__dict__["Nlay"] = len(L)

        lam0 = tf.convert_to_tensor(wavelength_set_m, dtype=dtype)
        lam0 = lam0[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
        lam0 = tf.tile(lam0, multiples=(1, pixelsX, pixelsY, 1, 1, 1))
        self.__dict__["lam0"] = lam0

        theta = tf.convert_to_tensor(thetas, dtype=dtype)
        theta = theta[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
        theta = tf.tile(theta, multiples=(1, pixelsX, pixelsY, 1, 1, 1))
        self.__dict__["theta"] = theta

        phi = tf.convert_to_tensor(phis, dtype=dtype)
        phi = phi[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
        phi = tf.tile(phi, multiples=(1, pixelsX, pixelsY, 1, 1, 1))
        self.__dict__["phi"] = phi

        pte = tf.convert_to_tensor(pte, dtype=cdtype)
        pte = pte[:, tf.newaxis, tf.newaxis, tf.newaxis]
        pte = tf.tile(pte, multiples=(1, pixelsX, pixelsY, 1))
        self.__dict__["pte"] = pte

        ptm = tf.convert_to_tensor(ptm, dtype=cdtype)
        ptm = ptm[:, tf.newaxis, tf.newaxis, tf.newaxis]
        ptm = tf.tile(ptm, multiples=(1, pixelsX, pixelsY, 1))
        self.__dict__["ptm"] = ptm

        L = tf.convert_to_tensor(L, dtype=cdtype)
        L = L[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
        self.__dict__["L"] = L

        # RCWA parameters.
        PQ = input_dict["PQ"]
        Nx = input_dict["Nx"]
        Ly = input_dict["Ly"]
        Lx = input_dict["Lx"]
        if PQ[1] == 1:
            self.__dict__["Ny"] = 1
        else:
            self.__dict__["Ny"] = int(np.round(Nx * Ly / Lx))  # number of point along y in real-space grid

        # Coefficient for the argument of tf.math.sigmoid() when generating
        # permittivity distributions with geometric parameters.
        self.__dict__["sigmoid_coeff"] = 1000.0

        # Polynomial order for rectangular resonators definition.
        # self.__dict__["rectangle_power"] = 200

        return

    def __get_broadband_permittivity(self):
        wavelength_set_m = self.__dict__["wavelength_set_m"]
        cdtype = self.__dict__["cdtype"]

        # Add tensor form layer electric permittivity embeddings
        lay_eps_list = []
        Lay_mat = self.__dict__["Lay_mat"]
        for lay_entry in Lay_mat:
            if isinstance(lay_entry, str):
                eps_rel = get_material_index(lay_entry, wavelength_set_m) ** 2
            else:
                eps_rel = np.ones_like(wavelength_set_m) * lay_entry

            eps_rel = tf.convert_to_tensor(
                eps_rel[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], dtype=cdtype
            )
            lay_eps_list.append(eps_rel)
        self.__dict__["lay_eps_list"] = lay_eps_list

        # Get the dieletric permittivity
        material_dielectric = self.__dict__["material_dielectric"]
        if isinstance(material_dielectric, str):
            eps_d = get_material_index(material_dielectric, wavelength_set_m) ** 2
        else:
            eps_d = np.ones_like(wavelength_set_m) * material_dielectric

        self.__dict__["erd"] = tf.convert_to_tensor(
            eps_d[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], dtype=cdtype
        )

        return

    def __set_refl_trans_perm(self):
        wavelength_set_m = self.__dict__["wavelength_set_m"]
        cdtype = self.__dict__["cdtype"]

        er1 = self.__dict__["er1"]
        if isinstance(er1, str):
            eps_rel = get_material_index(er1, wavelength_set_m) ** 2
        else:
            eps_rel = np.ones_like(wavelength_set_m) * er1

        eps_rel = tf.convert_to_tensor(
            eps_rel[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], dtype=cdtype
        )
        self.__dict__["er1"] = eps_rel

        er2 = self.__dict__["er2"]
        if isinstance(er2, str):
            eps_rel = get_material_index(er2, wavelength_set_m) ** 2
        else:
            eps_rel = np.ones_like(wavelength_set_m) * er2
        eps_rel = tf.convert_to_tensor(
            eps_rel[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], dtype=cdtype
        )
        self.__dict__["er2"] = eps_rel

        ur1 = np.ones_like(wavelength_set_m) * self.__dict__["ur1"]
        ur1 = tf.convert_to_tensor(ur1[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], dtype=cdtype)
        self.__dict__["ur1"] = ur1

        ur2 = np.ones_like(wavelength_set_m) * self.__dict__["ur2"]
        ur2 = tf.convert_to_tensor(ur2[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], dtype=cdtype)
        self.__dict__["ur2"] = ur2
        return

    def __check_force_span_simple(self, span_limits):
        if not ("min" in span_limits.keys() and "max" in span_limits.keys()):
            raise ValueError("rcwa_params: 'force_span_limits' must have keys 'min' and 'max'")
        return

    def __setitem__(self, key, item):
        if key in self.__dict__.keys():
            # no change on the items after initialization shall be allowed
            raise "The params cannot be changed after initialization"
        else:
            # allow adding new keys
            self.__dict__[key] = item
        return item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))

    def get_dict(self):
        return deepcopy(self.__dict__)
