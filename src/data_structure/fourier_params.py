from copy import deepcopy
import numpy as np
import tensorflow as tf
import math

ALL_MANDATORY_KEYS = [
    ["wavelength_m", "wavelength_set_m"],
    "ms_length_m",
    "ms_dx_m",
    "sensor_distance_m",
    "initial_sensor_dx_m",
    "sensor_pixel_size_m",
    "sensor_pixel_number",
    "radial_symmetry",
    "diffractionEngine",
]

diffractionEngines = ["fresnel_fourier", "ASM_fourier"]

ALL_OPTIONAL_KEYS = {
    "nyquist_modifier": 1,
    "antialias_ms": False,
    "radius_m": None,
    "dtype": tf.float64,
    "accurate_measurement": True,
}

HIDDEN_KEYS = ["_prop_params__verbose"]

ADDED_KEYS = [
    "ms_samplesM",
    "calc_ms_dx_m",
    "padms_half",
    "calc_samplesN",
    "calc_sensor_dx_m",
    "ratio_pixel_to_grid",
    "broadband_flag",
    "grid_shape",
]


def estimateBandwidth(parameters):
    # Compute fresnel number to determine estimate for minimum Whittaker-Shannon lens sampling
    # Use fresnel number to define an approximate fourier bandwidth

    bandwidthxy = []
    for dimIdx in ["x", "y"]:
        ms_length_m = parameters["ms_length_m"][dimIdx]
        wavelength_m = parameters["wavelength_m"]
        sensor_distance_m = parameters["sensor_distance_m"]

        # Compute the Fresnel Number
        Nf = (ms_length_m / 2.0) ** 2 / wavelength_m / sensor_distance_m

        # Define Fourier bandwidth based on Fresnel Number Regime
        if Nf < 0.25:
            bandwidth = 1 / ms_length_m
        else:
            bandwidth = ms_length_m / wavelength_m / sensor_distance_m

        bandwidthxy.append(bandwidth)

    return np.array(bandwidthxy)


class prop_params(dict):
    """Parameters object (dictionary) used for the propagation in the Fourier layers. Defines the simulation settings.    
    """

    def __init__(self, input_dict, verbose=False):
        """Parameters object (dictionary) used for the propagation in the Fourier layers. Defines the simulation settings.

        Args:
            `input_dict` (dict): Input dictionary with keys listed below.
            `verbose` (bool, optional): Boolean flag to print simulation settings info after initialization. Defaults
                to False.

            A description of keys required/used for the input_dict:\\
                `"wavelength_m"` or `"wavelength_set_m"`: Simulation wavelength(s) in units of m, as a float or a list 
                    of floats in the case of the latter. \\
                `"ms_length_m"`: Dict containing the metasurface length along x and y, via {"x": float, "y": float}.\\
                `"ms_dx_m"`: Dict containing the metasurface grid discretization along x and y, 
                    via {"x": float, "y": float}. \\
                `"sensor_distance_m"`: Float constant providing the distance between the metasurface and the sensor 
                    plane (or input plane to output plane for propagation more generally). \\
                `"initial_sensor_dx_m"`: Dict containing the sensor plane grid discretization along x and y,
                     via {"x": float, "y":float}. \\
                `"sensor_pixel_size_m"`: Dict containing the pixel size of a detector placed at the sensor plane,
                    via {"x": float, "y":float}. \\
                `"sensor_pixel_number"`: Dict containing the number of detector pixels at the sensor plane, 
                    via {"x": int, "y": int}. \\
                `"radial_symmetry": Boolean flag indicating if input data is of radial vector form or 2D cartesian. \\
                `"diffractionEngine": String selection for the diffraction engine to use in propagation. 
                    See documentation for valid options. \\
                (optional) `"radius_m"`: float indicating the radius of a circular field aperture to be placed at the
                     metasurface/input plane. Defaults to None.\\ 
                (optional) `"dtype"`: tf.dtype to be used during all calculations. Defaults to tf.float64 and should
                     not be changed in the current version!
        """
        self.__dict__ = deepcopy(input_dict)
        self.__check_mandatory_keys()
        self.__check_wavelength_key()
        self.__check_optional_keys()
        self.__check_unknown_keys()
        self.__check_diffractionEngine_selection()
        self.__regularize_radial_symmetry()
        self.__check_detector_pixel_size()
        self.__add_implied_keys()
        self.__verbose = verbose
        self.__enforce_odd_sensor_pixel_number()

        # If wavelength_m is given, this is a usable prop_instance keys
        # so generate the additional keys required for function calls.
        # If wavelength_set_m is given instead, this is a parent object and
        # additional keys should not be added to avoid confusion.
        if "wavelength_m" in self.__dict__.keys():
            self.__regularizeInputOutputSpace()

        return

    def __check_mandatory_keys(self):
        for exclusive_key in ALL_MANDATORY_KEYS:

            if isinstance(exclusive_key, list):
                if not any(check_key in self.__dict__.keys() for check_key in exclusive_key):
                    print(exclusive_key)
                    raise KeyError("\n params: one of the above keys must be included")
            else:
                if not (exclusive_key in self.__dict__.keys()):
                    raise KeyError(
                        "params: Missing mandatory parameter option for simulation settings: " + exclusive_key
                    )

        return

    def __check_wavelength_key(self):
        # only one of wavelength_m or wavelength_set_m should be provided.
        if "wavelength_m" in self.__dict__.keys():
            self.__dict__["broadband_flag"] = False
            if "wavelength_set_m" in self.__dict__.keys():
                raise KeyError(
                    "params: wavelength_m and wavelength_set_m cannot both be given. Choose one or the other."
                )

        if "wavelength_set_m" in self.__dict__.keys():
            self.__dict__["broadband_flag"] = True
            if "wavelength_m" in self.__dict__.keys():
                raise KeyError(
                    "params: wavelength_m and wavelength_set_m cannot both be given. Choose one or the other."
                )
            if not isinstance(self.__dict__["wavelength_set_m"], (list, np.ndarray)):
                raise ValueError("params: wavelength_set_m must be either a list or a numpy array")

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
            if not providedKey in (
                flattened_all_mandatory_keys + list(ALL_OPTIONAL_KEYS.keys()) + HIDDEN_KEYS + ADDED_KEYS
            ):
                raise KeyError("params: unknown parameter key/setting inputed: " + providedKey)

        return

    def __check_diffractionEngine_selection(self):
        if not (self.__dict__["diffractionEngine"] in diffractionEngines):
            raise ValueError("diffractionEngine selection invalid: must be either 'fresnel_fourier' or 'ASM_fourier'.")

        return

    def __regularize_radial_symmetry(self):
        # If radial_symmetry flag is activated, ensure initial lens space is square and uniformly sampled
        if self.__dict__["radial_symmetry"]:
            ms_length_m = self.__dict__["ms_length_m"]
            ms_dx_m = self.__dict__["ms_dx_m"]

            if not (ms_length_m["x"] == ms_length_m["y"]):
                raise ValueError("params: initial_lens_length_m must be square when radial_symmetry flag is used")

            if not (ms_dx_m["x"] == ms_dx_m["y"]):
                raise ValueError(
                    "params: iniital_lens_dx_m must be same along x and y when radial_symmetry flag is used"
                )

        return

    def __check_detector_pixel_size(self):
        initial_sensor_dx_m = self.__dict__["initial_sensor_dx_m"]
        sensor_pixel_size_m = self.__dict__["sensor_pixel_size_m"]
        if (initial_sensor_dx_m["x"] > sensor_pixel_size_m["x"]) or (
            initial_sensor_dx_m["y"] > sensor_pixel_size_m["y"]
        ):
            raise ValueError(
                "params: initial sensor plane field grid cannot be discretized larger than the detector pixels!"
            )
        return

    def __add_implied_keys(self):
        # add key for number samples in user's metasurface
        # For convenience lets enforce odd number of samples
        ms_length_m = self.__dict__["ms_length_m"]
        ms_dx_m = self.__dict__["ms_dx_m"]
        ms_samplesM_x = int(math.ceil(ms_length_m["x"] / ms_dx_m["x"]))
        ms_samplesM_y = int(math.ceil(ms_length_m["y"] / ms_dx_m["y"]))

        if np.mod(ms_samplesM_x, 2) == 0:
            ms_samplesM_x += 1
        if np.mod(ms_samplesM_y, 2) == 0:
            ms_samplesM_y += 1

        ms_samplesM_r = int((ms_samplesM_x - 1) / 2 + 1)

        self.__dict__["ms_samplesM"] = {
            "x": ms_samplesM_x,
            "y": ms_samplesM_y,
            "r": ms_samplesM_r,
        }

        # It is handy downstream to add a grid_shape key:
        if self.__dict__["radial_symmetry"]:
            self.__dict__["grid_shape"] = [1, 1, ms_samplesM_r]
        else:
            self.__dict__["grid_shape"] = [1, ms_samplesM_y, ms_samplesM_x]

        return

    # def __check_sensorgrid_to_sensorpixel_ratio(self):
    #     # For the current version, we want to enforce that the ratio of the sensor pixel size to the
    #     # sensor grid pitch is an integer value >= 1
    #     initial_sensor_dx_m = self.__dict__["initial_sensor_dx_m"]
    #     sensor_pixel_size_m = self.__dict__["sensor_pixel_size_m"]

    #     if (np.mod(sensor_pixel_size_m["x"], initial_sensor_dx_m["x"]) >= 1e-12) or (
    #         np.mod(sensor_pixel_size_m["y"], initial_sensor_dx_m["y"]) >= 1e-12
    #     ):
    #         raise ValueError(
    #             "Invalid sensor space; current version requires sensor pixel size to be an integer multiple of initial_sensor_dx!"
    #         )

    #     self.__dict__["sensratio_pixel_to_grid"] = {
    #         "x": int(sensor_pixel_size_m["x"] / initial_sensor_dx_m["x"]),
    #         "y": int(sensor_pixel_size_m["y"] / initial_sensor_dx_m["y"]),
    #     }

    #     return

    def __enforce_odd_sensor_pixel_number(self):
        sensor_pixel_number = self.__dict__["sensor_pixel_number"]
        sensor_num_x = sensor_pixel_number["x"]
        sensor_num_y = sensor_pixel_number["y"]

        if np.mod(sensor_num_x, 2) == 0:
            sensor_num_x += 1
        if np.mod(sensor_num_y, 2) == 0:
            sensor_num_y += 1

        # Add a radial key for convenience in the propagator functions
        sensor_num_r = int((sensor_num_x - 1) / 2 + 1)
        self.__dict__["sensor_pixel_number"] = {"x": sensor_num_x, "y": sensor_num_y, "r": sensor_num_r}

        return

    def __regularizeInputOutputSpace(self):
        # Call to update lens sampling rate and number of samples representing the upadded lens
        self.__regularizeLensPlaneSampling()

        # Call to compute the required padding to achieve an approximate target sampling for field before sensor plane
        self.__regularizeSensorPlaneSampling()

        # Print statement to show if samples have changed relative to what the user requested
        if self.__verbose:
            ms_dx_m = self.__dict__["ms_dx_m"]
            calc_ms_dx_m = self.__dict__["calc_ms_dx_m"]
            ms_samplesM = self.__dict__["ms_samplesM"]
            calc_samplesM = self.__dict__["calc_samplesM"]
            calc_samplesN = self.__dict__["calc_samplesN"]
            initial_sensor_dx_m = self.__dict__["initial_sensor_dx_m"]
            calc_sensor_dx_m = self.__dict__["calc_sensor_dx_m"]
            sensor_pixel_size_m = self.__dict__["sensor_pixel_size_m"]
            sensor_pixel_number = self.__dict__["sensor_pixel_number"]

            print("\n OVERVIEW OF PARAMETERS \n")
            print("\n", "ms_dx_m: ", ms_dx_m)
            print("\n", "calc_ms_dx_m: ", calc_ms_dx_m)
            print("\n", "ms_samplesM: ", ms_samplesM)
            print("\n", "calc_samplesM: ", calc_samplesM)
            print("\n", "calc_samplesN: ", calc_samplesN)
            print("\n", "initial_sensor_dx_m: ", initial_sensor_dx_m)
            print("\n", "calc_sensor_dx_m: ", calc_sensor_dx_m)
            print("\n", "detector pixel size: ", sensor_pixel_size_m)
            print("\n", "detector pixel number: ", sensor_pixel_number)
            print("\n")
        return

    def __regularizeLensPlaneSampling(self):

        nyquist_modifier = self.__dict__["nyquist_modifier"]
        diffractionEngine = self.__dict__["diffractionEngine"]
        # antialias_ms = self.__dict__["antialias_ms"]

        # Obtain the max estimated bandwidth required to compute metasurface sampling rate cutoff
        # For the fresnel case, just get estBandwidth from quadratic phase profile
        # For the exact transfer function, we should consider the estBandwidth vs the H bandwidth
        if diffractionEngine == "fresnel_fourier":
            estBandwidth = estimateBandwidth(self.__dict__)
        elif diffractionEngine == "ASM_fourier":
            wavelength_m = self.__dict__["wavelength_m"]
            tf_bandwidth = np.array([1 / wavelength_m, 1 / wavelength_m])
            quad_bandwidth = estimateBandwidth(self.__dict__)
            estBandwidth = np.maximum(tf_bandwidth, quad_bandwidth)
        samplingCutoff = 1 / 2 / estBandwidth * nyquist_modifier
        ms_dx_m = self.__dict__["ms_dx_m"]
        # Print information to the user
        if self.__verbose:
            print(
                "PARAMS: nyquist_modifier: ", nyquist_modifier, " SamplingCutoff: ", samplingCutoff / nyquist_modifier,
            )

        # define calculation sampling of lens space in x (consistent to nyquist rule)
        calc_ms_dx = ms_dx_m["x"]
        calc_ms_dy = ms_dx_m["y"]
        if ms_dx_m["x"] > samplingCutoff[0]:
            calc_ms_dx = samplingCutoff[0]
        if ms_dx_m["y"] > samplingCutoff[1]:
            calc_ms_dy = samplingCutoff[1]

        # If we are using the transferFunction engine, we should also verify that calc_ms_dx is
        # smaller than the initial_sensor_dx_m requested by the user. This is because the lens sampling will
        # directly equal the sensor plane sampling. To give the user the right value, we will need to interpolate
        if diffractionEngine == "ASM_fourier":
            initial_sensor_dx_m = self.__dict__["initial_sensor_dx_m"]
            if calc_ms_dx > initial_sensor_dx_m["x"]:
                calc_ms_dx = initial_sensor_dx_m["x"]
            if calc_ms_dy > initial_sensor_dx_m["y"]:
                calc_ms_dy = initial_sensor_dx_m["y"]
        self.__dict__["calc_ms_dx_m"] = {"x": calc_ms_dx, "y": calc_ms_dy}

        # add a parameter to reperesent the corresponding number of samples to be used for unpadded, upsampled lens ("M")
        ms_length_m_x = self.__dict__["ms_length_m"]
        calc_samplesM_x = int(math.ceil(ms_length_m_x["x"] / calc_ms_dx))
        calc_samplesM_y = int(math.ceil(ms_length_m_x["y"] / calc_ms_dy))
        # Ensure number of samples for unpadded lens is odd
        if np.mod(calc_samplesM_x, 2) == 0:
            calc_samplesM_x += 1
        if np.mod(calc_samplesM_y, 2) == 0:
            calc_samplesM_y += 1

        # samples along r is added in for convenience
        calc_samplesM_r = int((calc_samplesM_x - 1) / 2 + 1)
        self.__dict__["calc_samplesM"] = {
            "x": calc_samplesM_x,
            "y": calc_samplesM_y,
            "r": calc_samplesM_r,
        }

        # # Modify the sampling rate if anti-aliasing is requested
        # ms_samplesM = self.__dict__["ms_samplesM"]
        # calc_samplesM = self.__dict__["calc_samplesM"]
        # if antialias_ms:
        #     Rx = np.ceil(calc_samplesM["x"] / ms_samplesM["x"])
        #     Ry = np.ceil(calc_samplesM["y"] / ms_samplesM["y"])
        #     self.__dict__["calc_samplesM"] = {"x": int(ms_samplesM["x"] * Rx), "y": int(ms_samplesM["y"] * Ry)}
        #     self.__dict__["calc_ms_dx_m"] = {
        #         "x": ms_length_m_x["x"] / ms_samplesM["x"] / Rx,
        #         "y": ms_length_m_x["y"] / ms_samplesM["y"] / Ry,
        #     }

        return

    def __regularizeSensorPlaneSampling(self):

        # Based on new ms sampling rate, define required padding for the fresnel fourier engine
        #  to achieve target sensor sampling
        # samplesN corresponds to the length of the lens space after padding
        initial_sensor_dx_m = self.__dict__["initial_sensor_dx_m"]
        calc_samplesM = self.__dict__["calc_samplesM"]
        diffractionEngine = self.__dict__["diffractionEngine"]
        calc_ms_dx_m = self.__dict__["calc_ms_dx_m"]
        padms_halfx = 0
        padms_halfy = 0

        if diffractionEngine == "fresnel_fourier":
            wavelength_m = self.__dict__["wavelength_m"]
            sensor_distance_m = self.__dict__["sensor_distance_m"]

            estN_x = int(math.ceil(wavelength_m * sensor_distance_m / initial_sensor_dx_m["x"] / calc_ms_dx_m["x"]))
            estN_y = int(math.ceil(wavelength_m * sensor_distance_m / initial_sensor_dx_m["y"] / calc_ms_dx_m["y"]))

            if estN_x > calc_samplesM["x"]:
                padms_halfx = int(math.ceil((estN_x - calc_samplesM["x"]) / 2))
            if estN_y > calc_samplesM["y"]:
                padms_halfy = int(math.ceil((estN_y - calc_samplesM["y"]) / 2))

        # For the ASM case, we pad to ensure sensor space span
        if diffractionEngine == "ASM_fourier":
            sensor_pixel_size_m = self.__dict__["sensor_pixel_size_m"]
            sensor_pixel_number = self.__dict__["sensor_pixel_number"]
            current_span_x = calc_ms_dx_m["x"] * calc_samplesM["x"]
            current_span_y = calc_ms_dx_m["y"] * calc_samplesM["y"]
            desired_span_x = sensor_pixel_size_m["x"] * sensor_pixel_number["x"]
            desired_span_y = sensor_pixel_size_m["y"] * sensor_pixel_number["y"]
            if current_span_x < desired_span_x:
                padms_halfx = int(math.ceil((desired_span_x - current_span_x) / 2))
            if current_span_y < desired_span_y:
                padms_halfy = int(math.ceil((desired_span_y - current_span_y) / 2))

        # Update the parameter settings based on new padding settings
        self.__dict__["padms_half"] = {"x": padms_halfx, "y": padms_halfy}
        calc_samplesN_x = padms_halfx * 2 + calc_samplesM["x"]
        calc_samplesN_y = padms_halfy * 2 + calc_samplesM["y"]
        calc_samplesN_r = int((calc_samplesN_x - 1) / 2 + 1)
        self.__dict__["calc_samplesN"] = {
            "x": calc_samplesN_x,
            "y": calc_samplesN_y,
            "r": calc_samplesN_r,
        }

        # Redefine the sensor sampling for the fresnel case since the array length is now N insead of M
        if diffractionEngine == "fresnel_fourier":
            calc_sensor_dx_m = wavelength_m * sensor_distance_m / calc_ms_dx_m["x"] / calc_samplesN_x
            calc_sensor_dy_m = wavelength_m * sensor_distance_m / calc_ms_dx_m["y"] / calc_samplesN_y
        elif diffractionEngine == "ASM_fourier":
            calc_ms_dx_m = self.__dict__["calc_ms_dx_m"]
            calc_sensor_dx_m = calc_ms_dx_m["x"]
            calc_sensor_dy_m = calc_ms_dx_m["y"]

        # Update the parameter setting with the new sensor plane calculation
        self.__dict__["calc_sensor_dx_m"] = {
            "x": calc_sensor_dx_m,
            "y": calc_sensor_dy_m,
        }

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

    def keys(self):
        return self.__dict__.keys()

    def get_dict(self):
        return deepcopy(self.__dict__)

    def has_key(self, key_name):
        if key_name in self.__dict__.keys():
            return True
        else:
            return False

