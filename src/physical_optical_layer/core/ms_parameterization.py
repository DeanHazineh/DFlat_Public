import tensorflow as tf
import numpy as np

# NOTE: THESE CONSTANTS NEED TO BE MOVED INSIDE EACH FUNCTION CALL SO THE GRAPH IS NOT EAGER MODE!
# D.H.
# TF_ZERO = tf.constant(0.0, dtype=tf.float32)
# TF_ONE = tf.constant(1.0, dtype=tf.float32)
# TF_ONE_COMPLEX = tf.complex(TF_ONE, TF_ZERO)
# POWER_EXP = 10


def get_cartesian_grid(Lx, Nx, Ly, Ny):
    dx = Lx / Nx  # grid resolution along x
    dy = Ly / Ny  # grid resolution along y
    xa = np.linspace(0, Nx - 1, Nx) * dx  # x axis array
    xa = xa - np.mean(xa)  # center x axis at zero
    ya = np.linspace(0, Ny - 1, Ny) * dy  # y axis vector
    ya = ya - np.mean(ya)  # center y axis at zero
    [y_mesh, x_mesh] = np.meshgrid(ya, xa)

    y_mesh = tf.convert_to_tensor(y_mesh, dtype=tf.float32)
    x_mesh = tf.convert_to_tensor(x_mesh, dtype=tf.float32)

    return x_mesh, y_mesh


def build_rectangle_resonator(norm_param, span_limits, Lx, Ly, x_mesh, y_mesh, sigmoid_coeff, Nlay):
    # norm_param: A 'tf.Tensor' of shape (2, pixelsX, pixelsY, 1)
    TF_ZERO = tf.constant(0.0, dtype=tf.float32)
    TF_ONE = tf.constant(1.0, dtype=tf.float32)
    TF_ONE_COMPLEX = tf.complex(TF_ONE, TF_ZERO)

    # unpack inputs
    norm_px = norm_param[0:1, :, :, :]
    norm_py = norm_param[1:2, :, :, :]
    span_max = tf.cast(span_limits["max"], dtype=tf.float32)
    span_min = tf.cast(span_limits["min"], dtype=tf.float32)

    #
    r_x = (norm_px * (span_max - span_min) + span_min) * Lx
    r_y = (norm_py * (span_max - span_min) + span_min) * Ly
    r_x = tf.tile(r_x, multiples=(1, 1, 1, 1))
    r_y = tf.tile(r_y, multiples=(1, 1, 1, 1))
    r_x = r_x[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
    r_y = r_y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
    r1 = (
        1
        - tf.abs((x_mesh / r_x[:, :, :, :, :, :, 0]) - (y_mesh / r_y[:, :, :, :, :, :, 0]))
        - tf.abs((x_mesh / r_x[:, :, :, :, :, :, 0]) + (y_mesh / r_y[:, :, :, :, :, :, 0]))
    )
    r1 = tf.complex(tf.math.sigmoid(sigmoid_coeff * r1), TF_ZERO)

    struct_binaries = []
    for i in range(Nlay):
        struct_binaries.append(None)
    struct_binaries[1] = r1

    return struct_binaries


def build_elliptical_resonator(norm_param, span_limits, Lx, Ly, erd, x_mesh, y_mesh, sigmoid_coeff):
    # norm_param: A 'tf.Tensor' of shape (2, pixelsX, pixelsY, 1)

    # unpack inputs
    norm_px = norm_param[0:1, :, :, :]
    norm_py = norm_param[1:2, :, :, :]
    batchSize = erd.shape[0]
    span_max = tf.cast(span_limits["max"], dtype=tf.float32)
    span_min = tf.cast(span_limits["min"], dtype=tf.float32)

    #
    r_x = (norm_px * (span_max - span_min) + span_min) * Lx
    r_y = (norm_py * (span_max - span_min) + span_min) * Ly
    r_x = tf.tile(r_x, multiples=(batchSize, 1, 1, 1))
    r_y = tf.tile(r_y, multiples=(batchSize, 1, 1, 1))
    r_x = r_x[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
    r_y = r_y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
    r1 = 1 - (x_mesh * 2 / r_x[:, :, :, :, :, :, 0]) ** 2 - (y_mesh * 2 / r_y[:, :, :, :, :, :, 0]) ** 2

    r1 = tf.complex(tf.math.sigmoid(sigmoid_coeff * r1), TF_ZERO)
    r1 = 1 + (erd - TF_ONE_COMPLEX) * r1

    return r1


def build_cylindrical_nanoposts(norm_param, span_limits, Lx, Ly, erd, x_mesh, y_mesh, sigmoid_coeff):
    # norm_param: A 'tf.Tensor' of shape (1, pixelsX, pixelsY, 1)

    # unpack inputs
    norm_pr = norm_param[0:1, :, :, :]
    batchSize = erd.shape[0]
    span_max = tf.cast(span_limits["max"], dtype=tf.float32)
    span_min = tf.cast(span_limits["min"], dtype=tf.float32)

    #
    radius = ((norm_pr * (span_max - span_min) + span_min) * 0.5) * tf.math.minimum(Lx, Ly)
    radius = tf.tile(radius, multiples=(batchSize, 1, 1, 1))
    radius = radius[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]

    r1 = 1 - (x_mesh / radius[:, :, :, :, :, :, 0]) ** 2 - (y_mesh / radius[:, :, :, :, :, :, 0]) ** 2
    r1 = tf.complex(tf.math.sigmoid(sigmoid_coeff * r1), TF_ZERO)
    r1 = 1 + (erd - TF_ONE_COMPLEX) * r1

    return r1


def build_coupled_elliptical_resonators(norm_param, span_limits, Lx, Ly, erd, x_mesh, y_mesh, sigmoid_coeff):
    # norm_param: A 'tf.Tensor' of shape (2, pixelsX, pixelsY, 4)

    # unpack inputs
    norm_px = norm_param[0:1, :, :, :]
    norm_py = norm_param[1:2, :, :, :]
    batchSize = erd.shape[0]
    span_max = tf.cast(span_limits["max"], dtype=tf.float32)
    span_min = tf.cast(span_limits["min"], dtype=tf.float32)

    # Nanopost centers.
    c1_x = -Lx / 4
    c1_y = -Ly / 4
    c2_x = -Lx / 4
    c2_y = Ly / 4
    c3_x = Lx / 4
    c3_y = -Ly / 4
    c4_x = Lx / 4
    c4_y = Ly / 4

    # Clip the optimization ranges.
    r_x = (norm_px * (span_max - span_min) + span_min) * Lx
    r_y = (norm_py * (span_max - span_min) + span_min) * Ly
    r_x = tf.tile(r_x, multiples=(batchSize, 1, 1, 1))
    r_y = tf.tile(r_y, multiples=(batchSize, 1, 1, 1))
    r_x = r_x[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
    r_y = r_y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]

    # Calculate the nanopost boundaries.
    c1 = 1 - ((x_mesh - c1_x) / r_x[:, :, :, :, :, :, 0]) ** 2 - ((y_mesh - c1_y) / r_y[:, :, :, :, :, :, 0]) ** 2
    c2 = 1 - ((x_mesh - c2_x) / r_x[:, :, :, :, :, :, 1]) ** 2 - ((y_mesh - c2_y) / r_y[:, :, :, :, :, :, 1]) ** 2
    c3 = 1 - ((x_mesh - c3_x) / r_x[:, :, :, :, :, :, 2]) ** 2 - ((y_mesh - c3_y) / r_y[:, :, :, :, :, :, 2]) ** 2
    c4 = 1 - ((x_mesh - c4_x) / r_x[:, :, :, :, :, :, 3]) ** 2 - ((y_mesh - c4_y) / r_y[:, :, :, :, :, :, 3]) ** 2

    c1 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c1), TF_ZERO)
    c2 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c2), TF_ZERO)
    c3 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c3), TF_ZERO)
    c4 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c4), TF_ZERO)
    return 1 + (erd - TF_ONE_COMPLEX) * (c1 + c2 + c3 + c4)


def build_coupled_rectangular_resonators(norm_param, span_limits, Lx, Ly, erd, x_mesh, y_mesh, sigmoid_coeff):
    # norm_param: A 'tf.Tensor' of shape (2, pixelsX, pixelsY, 4)

    # unpack inputs
    norm_px = norm_param[0:1, :, :, :]
    norm_py = norm_param[1:2, :, :, :]
    batchSize = erd.shape[0]
    span_max = tf.cast(span_limits["max"], dtype=tf.float32)
    span_min = tf.cast(span_limits["min"], dtype=tf.float32)

    # Nanopost centers.
    c1_x = -Lx / 4
    c1_y = -Ly / 4
    c2_x = -Lx / 4
    c2_y = Ly / 4
    c3_x = Lx / 4
    c3_y = -Ly / 4
    c4_x = Lx / 4
    c4_y = Ly / 4

    # Nanopost width ranges
    r_x = (norm_px * (span_max - span_min) + span_min) * Lx
    r_y = (norm_py * (span_max - span_min) + span_min) * Ly
    r_x = tf.tile(r_x, multiples=(batchSize, 1, 1, 1))
    r_y = tf.tile(r_y, multiples=(batchSize, 1, 1, 1))
    r_x = r_x[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
    r_y = r_y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]

    c1 = (
        1
        - ((x_mesh - c1_x) / r_x[:, :, :, :, :, :, 0]) ** POWER_EXP
        - ((y_mesh - c1_y) / r_y[:, :, :, :, :, :, 0]) ** POWER_EXP
    )
    c2 = (
        1
        - ((x_mesh - c2_x) / r_x[:, :, :, :, :, :, 1]) ** POWER_EXP
        - ((y_mesh - c2_y) / r_y[:, :, :, :, :, :, 1]) ** POWER_EXP
    )
    c3 = (
        1
        - ((x_mesh - c3_x) / r_x[:, :, :, :, :, :, 2]) ** POWER_EXP
        - ((y_mesh - c3_y) / r_y[:, :, :, :, :, :, 2]) ** POWER_EXP
    )
    c4 = (
        1
        - ((x_mesh - c4_x) / r_x[:, :, :, :, :, :, 3]) ** POWER_EXP
        - ((y_mesh - c4_y) / r_y[:, :, :, :, :, :, 3]) ** POWER_EXP
    )

    c1 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c1), TF_ZERO)
    c2 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c2), TF_ZERO)
    c3 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c3), TF_ZERO)
    c4 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c4), TF_ZERO)
    return 1 + (erd - TF_ONE_COMPLEX) * (c1 + c2 + c3 + c4)


def generate_cell_perm(norm_param, rcwa_parameters):
    """
    Generates permittivity and permeability for a unit cell comprising of structures according to "parameterization_type"
    set in the rcwa_parameters setting dict. 
    
    Args:
        `norm_param` (tf.float): A tensor of shape (d1, pixelsX, pixelsY, d2), where d1 are normalized shape parameters
            for each of the d2 number of structures placed in the cell
        `rcwa_parameters`: A dict of type `rcwa_params` containing simulation and optimization settings.
    
    Returns:
        `tf.float`: A tensor of shape (batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny) specifying the relative permittivity
            distribution of each cell.
        `tf.float`: A tensor of shape (batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny) specifying the relative permeability
            distribution of each cell.
  """

    # Retrieve simulation size parameters
    parameterization_type = rcwa_parameters["parameterization_type"]
    batchSize = rcwa_parameters["batchSize"]
    pixelsX = rcwa_parameters["pixelsX"]
    pixelsY = rcwa_parameters["pixelsY"]
    Nlay = rcwa_parameters["Nlay"]
    Nx = rcwa_parameters["Nx"]
    Ny = rcwa_parameters["Ny"]
    Lx = rcwa_parameters["Lx"]
    Ly = rcwa_parameters["Ly"]
    dtype = rcwa_parameters["dtype"]
    cdtype = rcwa_parameters["cdtype"]

    # Define the cartesian cross section.
    # Convert to tensors and expand and tile to match the simulation shape.
    x_mesh, y_mesh = get_cartesian_grid(Lx, Nx, Ly, Ny)
    y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    y_mesh = tf.tile(y_mesh, multiples=(batchSize, pixelsX, pixelsY, 1, 1, 1))
    x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    x_mesh = tf.tile(x_mesh, multiples=(batchSize, pixelsX, pixelsY, 1, 1, 1))

    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    materials_shape_lay = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    UR = rcwa_parameters["urd"] * tf.ones(materials_shape, dtype=cdtype)

    # Initialize the relative permittivity
    init_function = ALLOWED_PARAMETERIZATION_TYPE[parameterization_type]
    span_limits = rcwa_parameters["span_limits"]
    sigmoid_coeff = rcwa_parameters["sigmoid_coeff"]
    lay_eps_list = rcwa_parameters["lay_eps_list"]

    struct_binary = init_function(norm_param, span_limits, Lx, Ly, x_mesh, y_mesh, sigmoid_coeff, Nlay)
    ER = []
    for i in range(Nlay):
        if struct_binary[i] != None:
            ER.append(lay_eps_list[i] + (rcwa_parameters["erd"] - lay_eps_list[i]) * tf.cast(struct_binary[i], cdtype))
        else:
            ER.append(lay_eps_list[i] * tf.ones(materials_shape_lay, dtype=cdtype))

    ER = tf.concat(ER, axis=3)

    return ER, UR


ALLOWED_PARAMETERIZATION_TYPE = {
    "rectangular_resonators": build_rectangle_resonator,
    "elliptical_resonators": build_elliptical_resonator,
    "cylindrical_nanoposts": build_cylindrical_nanoposts,
    "coupled_elliptical_resonators": build_coupled_elliptical_resonators,
    "coupled_rectangular_resonators": build_coupled_rectangular_resonators,
    "None": None,
}

CELL_SHAPE_DEGREE = {  # See rcwa_params.__get_param_shape()
    "rectangular_resonators": [2, 1],
    "elliptical_resonators": [2, 1],
    "cylindrical_nanoposts": [1, 1],
    "coupled_elliptical_resonators": [2, 4],
    "coupled_rectangular_resonators": [2, 4],
    "None": None,
}

