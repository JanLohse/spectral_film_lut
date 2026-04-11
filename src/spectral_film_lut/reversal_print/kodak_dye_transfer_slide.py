"""Kodak Dye Transfer for Slides"""

from dataclasses import replace

from spectral_film_lut.reversal_print.kodak_dye_transfer_kodachrome import (
    KODAK_DYE_TRANSFER_KODACHROME,
)

KODAK_DYE_TRANSFER_SLIDE = replace(
    KODAK_DYE_TRANSFER_KODACHROME,
    name="Kodak Dye Transfer for Slides",
)
"""Kodak Dye Transfer for Slides"""

# TODO: integrate
# separation_neg = Kodak5222Dev6()
# sensitivity = separation_neg.sensitivity
# filters = xp.stack([WRATTEN["29"], WRATTEN["61"], WRATTEN["47"]])
# self.sensitivity = sensitivity * filters.T
#
# log_exposure_matrix = xp.array(list(curve.keys()), dtype=default_dtype)
# density_curve_matrix = xp.array(list(curve.values()), dtype=default_dtype)
# log_H_ref_mat = xp.interp(
#     xp.asarray(self.lad[0]), density_curve_matrix, log_exposure_matrix
# )
# separation_curve = separation_neg.density_curve[0]
# separation_exposure = separation_neg.log_exposure[0]
# slope = (separation_curve[-1] - separation_curve[-2]) / (
#         separation_exposure[-1] - separation_exposure[-2]
# )
# separation_curve = xp.append(separation_curve, separation_curve[-1] + slope * 1)
# separation_exposure = xp.append(
#     separation_exposure, separation_exposure[-1] + 1
# )
# density_curve = xp.interp(
#     log_H_ref_mat - separation_curve + separation_neg.d_ref,
#     log_exposure_matrix,
#     density_curve_matrix,
# )
#
# TODO: highlight_mask
#
# self.log_exposure = [separation_exposure] * 3
# self.density_curve = [density_curve * scale for scale in density_measurements]
#
# self.calibrate()
