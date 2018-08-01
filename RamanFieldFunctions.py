import numpy as np


def field_t(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t):

    return A_R * np.exp(-(t - t0_R) ** 2 / (2. * width_R ** 2)) * (np.cos(w_R * t) + np.cos((w_R + w_v) * t)) \
           + A_EE * np.exp(-(t - t0_EE) ** 2 / (2. * width_EE ** 2)) * (np.cos(freq * t)) + 0j


def d_dt_field_t(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t):

    return A_R \
           * np.exp(-(t - t0_R) ** 2 / width_R ** 2) \
           * (
                    (np.cos(w_R * t) + np.cos((w_R + w_v) * t)) * (-(t - t0_R) / (width_R ** 2))
                    - (w_R + w_v) * np.sin((w_R + w_v) * t)
                    - w_R * np.sin(w_R * t)
            )\
           + A_EE \
           * np.exp(-(t - t0_EE) ** 2 / (2. * width_EE ** 2)) \
           * (
                    (np.cos(freq * t)) * (-(t - t0_EE) / (width_EE ** 2))
                    - freq * np.sin(freq * t)
            ) + 0j


def grad_field_t_A_R(width_R, t0_R, w_R, w_v, t):

    return np.exp(-(t - t0_R) ** 2 / width_R ** 2) * (np.cos(w_R * t) + np.cos((w_R + w_v) * t)) + 0j


def grad_field_t_width_R(A_R, width_R, t0_R, w_R, w_v, t):

    return A_R \
           * np.exp(-(t - t0_R) ** 2 / width_R ** 2) \
           * (
                    (np.cos(w_R * t) + np.cos((w_R + w_v) * t)) * ((t - t0_R) ** 2 / (width_R ** 3))
            ) + 0j


def grad_field_t_A_EE(width_EE, t0_EE, freq, t):

    return np.exp(-(t - t0_EE) ** 2 / (2. * width_EE ** 2)) * (np.cos(freq * t)) + 0j


def grad_field_t_width_EE(A_EE, width_EE, t0_EE, freq, t):

    return A_EE \
           * np.exp(-(t - t0_EE) ** 2 / width_EE ** 2) \
           * (
                   (np.cos(freq * t)) * ((t - t0_EE) ** 2 / (width_EE ** 3))
            ) + 0j


def d_dt_A_R(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t):

    return d_dt_field_t(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t) \
           / grad_field_t_A_R(width_R, t0_R, w_R, w_v, t)


def d_dt_width_R(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t):

    return d_dt_field_t(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t) \
           / grad_field_t_width_R(A_R, width_R, t0_R, w_R, w_v, t)


def d_dt_A_EE(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t):

    return d_dt_field_t(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t) \
           / grad_field_t_A_EE(width_EE, t0_EE, freq, t)


def d_dt_width_EE(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t):

    return d_dt_field_t(A_R, A_EE, width_R, width_EE, t0_R, t0_EE, w_R, w_v, freq, t) \
           / grad_field_t_width_EE(A_EE, width_EE, t0_EE, freq, t)
