import numpy as np


def lookup_D2_pol2(phaseTable, transTable, p1_vect, p2_vect, wavelength, use_wavelength, ms_trans, ms_phase):
    # Find the sub-table matching the wavelength requested
    _, w_idx = min((val, idx) for (idx, val) in enumerate(np.abs(use_wavelength - wavelength)))
    sublib_phase = phaseTable[:, :, :, w_idx]
    sublib_trans = transTable[:, :, :, w_idx]
    or_table = sublib_trans * np.exp(1j * sublib_phase)
    or_table = np.hstack(
        (np.expand_dims(or_table[0:1, :, :].flatten(), -1), np.expand_dims(or_table[1:2, :, :].flatten(), -1))
    )

    # Get the two polarization profiles for each metasurface
    ms_trans = np.hstack(
        (np.expand_dims(ms_trans[0:1, :, :].flatten(), -1), np.expand_dims(ms_trans[1:2, :, :].flatten(), -1),)
    )
    ms_phase = np.hstack(
        (np.expand_dims(ms_phase[0:1, :, :].flatten(), -1), np.expand_dims(ms_phase[1:2, :, :].flatten(), -1),)
    )
    target_profile = ms_trans * np.exp(1j * ms_phase)

    # Brute-force look-up for the metasurface
    design_p1 = []
    design_p2 = []
    for cell in range(target_profile.shape[0]):
        _, param_idx = min(
            (val, idx)
            for (idx, val) in enumerate(
                0.5 * np.abs(target_profile[cell, 0] - or_table[:, 0])
                + 0.5 * np.abs(target_profile[cell, 1] - or_table[:, 1])
            )
        )
        design_p1.append(p1_vect[param_idx])
        design_p2.append(p2_vect[param_idx])

    # Define a normalized shape vector for convenience
    design_p1 = np.array(design_p1)
    design_p2 = np.array(design_p2)

    return design_p1, design_p2


def lookup_D1_pol1(phaseTable, transTable, p1_vect, wavelength, use_wavelength, ms_trans, ms_phase):

    # Find the sub-table matching the wavelength requested
    _, w_idx = min((val, idx) for (idx, val) in enumerate(np.abs(use_wavelength - wavelength)))
    sublib_phase = phaseTable[w_idx, :]
    sublib_trans = transTable[w_idx, :]
    or_table = sublib_trans * np.exp(1j * sublib_phase)

    # Get the target profile
    target_profile = ms_trans.flatten() * np.exp(1j * ms_phase.flatten())

    # minimize via look-up table
    design_p1 = []
    for cell in range(target_profile.shape[0]):
        _, param_idx = min((val, idx) for (idx, val) in enumerate(np.abs(target_profile[cell] - or_table[:])))
        design_p1.append(p1_vect[param_idx])

    # Define a normalized shape vector for convenience
    design_p1 = np.array(design_p1)

    return design_p1
