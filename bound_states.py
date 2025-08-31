import numpy as np
from Bessel import *

# Variablentransformationen (auch für partielle Ableitungen)
def to_x(y, v0):
    return np.sqrt(v0 - y**2)

def to_x_dy(y, v0):  
    return - y / np.sqrt(v0 - y**2)

def to_x_dv0(y, v0):                                       
    return 0.5 / np.sqrt(v0 - y**2)

# Bedingung für gebundene Zustände
def bound_states(l, y, v0):
    x = to_x(y, v0)
    numerator = dhl(l, 1.j*y)
    denominator = hl(l, 1.j*y)
    right = np.real(1.j * (y/x) * numerator / denominator)
    left = djl(l, x) / jl(l, x)
    return left - right


# partielle Ableitung nach y für Bedingung für gebundene Zustände
def bound_states_dy(l, y, v0):
    x     = to_x(y, v0)        
    dx_dy = -y / x              
    jl_x   = jl(l, x)
    djl_x  = djl(l, x)
    d2jl_x = d2jl(l, x)
    dleft_dx = (d2jl_x * jl_x - djl_x**2) / (jl_x**2)
    left_term = dleft_dx * dx_dy

    hy    = hl(l, 1j*y)
    dhy   = dhl(l, 1j*y)
    d2hy  = d2hl(l, 1j*y)
    B     = dhy / hy
    num    = 1j*d2hy * hy - dhy * 1j*dhy
    dB_dy  = num / (hy**2)

    A      = 1j * y / x
    dA_dy  = 1j*(1/x) + 1j*y*(y/(x**3))

    right_term = np.real(dA_dy * B + A * dB_dy)
    return left_term - right_term



# partielle Ableitung nach v0 für Bedingung für gebundene Zustände
def bound_states_dv0(l, y, v0):
    x = to_x(y, v0)
    dx_dv0 = 1/(2*x) 

    jl_x = jl(l, x)
    djl_x = djl(l, x)
    d2jl_x = d2jl(l, x)
    dleft_dx = (d2jl_x * jl_x - djl_x**2) / (jl_x**2)
    left_term = dleft_dx * dx_dv0

    hy = hl(l, 1.j*y)
    dhy = dhl(l, 1.j*y)
    quot = dhy / hy
    d_1_over_x_dv0 = -1/(2*x**3)
    right_term = np.real(1j * y * d_1_over_x_dv0 * quot)
    return left_term - right_term

#Funktion zur Berechnung der radialen Wellenfunktion (inside legt fest, ob r<r0 ist (für inside<1 der Fall, da inside=r/r0))
def radial_of_bound_states(l, x, a, b, A=1):
    radial_in = x* A * jl(l, a*x)
    radial_out = x* A * (jl(l, a)/hl(l, 1.j *b)) * hl(l, 1.j * b*x)
    return np.where(x <= 1, radial_in, radial_out)
