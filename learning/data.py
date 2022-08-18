import numpy as np
from nengo.utils.numpy import rmse
from scipy.optimize import curve_fit, minimize_scalar

sjostrom_x = np.array([0.1, 10, 20, 40, 50])
sjostrom_prepost = np.array([-0.04, 0.14, 0.29, 0.53, 0.56])
sjostrom_preposterr = np.array([0.05, 0.1, 0.14, 0.11, 0.26])
sjostrom_postpre = np.array([-0.29, -0.41, -0.34, 0.56, 0.75])
sjostrom_postpreerr = np.array([0.08, 0.11, 0.1, 0.32, 0.19])

# Ignore the 0.1 Hz case (too slow)
sjostrom_x = sjostrom_x[1:]
sjostrom_prepost = sjostrom_prepost[1:]
sjostrom_preposterr = sjostrom_preposterr[1:]
sjostrom_postpre = sjostrom_postpre[1:]
sjostrom_postpreerr = sjostrom_postpreerr[1:]
stdp_freqdep_prepost = np.array([
    3.71392743e-06, 9.40111624e-06, 2.43064947e-05, 3.34130531e-05,
])
stdp_freqdep_postpre = np.array([
    -5.44757319e-07, 6.86053886e-07, 1.71790439e-05, 3.40464669e-05,
])

bipoo_x = np.array([
    -0.0998, -0.0774, -0.0728, -0.0654, -0.0616, -0.0614, -0.0424,
    -0.0234, -0.0232, -0.0184, -0.0172, -0.0140, -0.0127, -0.0082,
    -0.0051, -0.0049, -0.0045, -0.0044, -0.0038, -0.0027, -0.0027,
    -0.0026,  0.0018,  0.0042,  0.0052,  0.0063,  0.0070,  0.0073,
    0.0074,  0.0078,  0.0079,  0.0080,  0.0081,  0.0167,  0.0168,
    0.0252,  0.0260,  0.0350,  0.0559,  0.0765,  0.0851,  0.0945,
]) * -1
bipoo_y = np.array([
    -0.0173, -0.1468,  0.0301,  0.0068, -0.0804, -0.0073, -0.1377,
    -0.0928, -0.1510, -0.2365, -0.1377, -0.0463, -0.3204, -0.2564,
    -0.3577, -0.2190, -0.1327, -0.4366,  0.7625,  0.0027, -0.1593,
    0.2319,  0.9070,  0.3008,  1.0324,  0.2136,  0.3348,  0.9626,
    0.8480,  0.4926,  0.0193,  0.7334,  0.1945,  0.3564,  0.4386,
    0.1463,  0.0259,  0.1804, -0.1111,  0.0980, -0.0447,  0.0284,
])
stdp_pair_y = np.array([
    6.11395746e-10, 8.67202229e-10, 1.29638246e-09, 1.90947890e-09,
    2.78954153e-09, 4.05639505e-09, 5.88304145e-09, 8.51935882e-09,
    1.23263453e-08, 1.78256030e-08, 2.57708465e-08, 3.72512414e-08,
    5.38407408e-08, 7.78138987e-08, 9.94671724e-08, 1.43749436e-07,
    2.07743262e-07, 3.00223388e-07, 4.33870612e-07, 6.27010586e-07,
    9.06126623e-07, -3.47637188e-07, -2.90157191e-07, -2.42181212e-07,
    -2.02137812e-07, -1.68715378e-07, -1.40819170e-07, -1.17535455e-07,
    -1.04193421e-07, -8.69655820e-08, -7.25862762e-08, -6.05845135e-08,
    -5.05671794e-08, -4.22061591e-08, -3.52275895e-08, -2.94028894e-08,
    -2.45412723e-08, -2.04834971e-08, -1.70966522e-08, -1.42698019e-08,
    -1.19103531e-08, -1.02683698e-08,
])

wang_x = np.array([-0.0885, 0.02, 0.0837])
wang_y = np.array([-0.003, 0.21, 0.06])
wang_yerr = np.array([0.03, 0.04, 0.04])
stdp_quadruplet_y = np.array([
    4.88292671e-07, 4.59280031e-07, 4.56800840e-07, 4.54315492e-07,
    4.52041389e-07, 4.50402619e-07, 4.50197781e-07, 4.52901391e-07,
    4.61205625e-07, 4.79994974e-07, 5.18101498e-07, 5.91468934e-07,
    7.28861415e-07, 1.11215096e-06, 1.67198434e-06, 2.69828594e-06,
    4.55767232e-06, 7.92243478e-06, 1.40082540e-05, 2.26787412e-05,
    3.87952383e-05, 2.58320448e-05, 1.84026971e-05, 1.35039086e-05,
    1.01658969e-05, 7.81952104e-06, 5.83820321e-06, 4.52472242e-06,
    3.66091783e-06, 2.99735152e-06, 2.48245526e-06, 2.07995195e-06,
    1.76360998e-06, 1.51401653e-06, 1.31653386e-06, 1.15996320e-06,
    1.03564223e-06, 9.36815960e-07, 8.58185460e-07, 8.30716659e-07,
])

wang_prepostpre_x = [(0.005, -0.005),
                     (0.01, -0.01),
                     (0.015, -0.005),
                     (0.005, -0.015)]
wang_prepostpre_y = np.array([-0.01, 0.03, 0.01, 0.24])
wang_prepostpre_yerr = np.array([0.04, 0.04, 0.03, 0.06])
stdp_prepostpre_y = np.array([
    4.27230803e-07, 3.06996927e-07, 5.26777740e-07, 6.93428552e-08,
])

wang_postprepost_x = [(-0.005, 0.005),
                      (-0.01, 0.01),
                      (-0.005, 0.015),
                      (-0.015, 0.005)]
wang_postprepost_y = np.array([0.33, 0.34, 0.22, 0.29])
wang_postprepost_yerr = np.array([0.04, 0.04, 0.08, 0.05])
stdp_postprepost_y = np.array([
    2.36156923e-05, 1.19947948e-05, 1.64309580e-05, 8.20517622e-06,
])


def exponential_fit(t_diffs, w_diffs, verbose=False):

    def pre(x, amp, tau):
        amp = abs(amp)
        return amp * np.exp(x / tau)

    def post(x, amp, tau):
        amp = abs(amp) * -1
        return amp * np.exp(-x / tau)

    def fit_subset(func, ix, guess):
        popt, _ = curve_fit(func, t_diffs[ix], w_diffs[ix], p0=guess)

        if verbose:
            print("%s: amp=%f, tau=%f" % (func.__name__,) + popt)
            print("error: %f" % np.sum(
                (w_diffs[ix] - func(t_diffs[ix], *popt)) ** 2, axis=0))
        return popt

    popt_pre = fit_subset(pre, t_diffs < 0, guess=[0.8, 0.01])
    popt_post = fit_subset(post, t_diffs > 0, guess=[0.8, 0.01])

    fit_x = np.linspace(-0.12, 0.12, 201)
    fit_y = np.zeros_like(fit_x)
    fit_y[fit_x < 0] = pre(fit_x[fit_x < 0], *popt_pre)
    fit_y[fit_x > 0] = post(fit_x[fit_x > 0], *popt_post)
    fit_y[fit_x == 0] = np.nan

    return fit_x, fit_y


def gini_index(vector):
    # Max sparsity = 1 (single 1 in the vector)
    v = np.sort(np.abs(vector))
    n = v.shape[0]
    k = np.arange(n) + 1
    l1norm = np.sum(v)
    summation = np.sum((v / l1norm) * ((n - k + 0.5) / n))
    return 1 - 2 * summation


def sparsity(array):
    # Iterate over first dimension
    array = np.reshape(array, (array.shape[0], -1))
    return np.hstack([gini_index(c) for c in array])


def scale(orig, target):
    """Rescale orig to match target."""

    def objective(scale):
        return rmse(orig * scale, target)

    res = minimize_scalar(objective)
    return res.x


def nearest_idx(array, val):
    return (np.abs(array - val)).argmin()


def nearest(array, val):
    return array[nearest_idx(array, val)]
