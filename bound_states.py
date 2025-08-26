import numpy as np
from Bessel import *

def to_x(y, v0):
    return np.sqrt(v0 - y**2)

def to_x_dy(y, v0):  
    return - y / np.sqrt(v0 - y**2)

def to_x_dv0(y, v0):                                            #!!!!!!!!!!!!!!!!!!!
    return 0.5 / np.sqrt(v0 - y**2)


def bound_states(l, y, v0):
    x = to_x(y, v0)
    numerator = dhl(l, 1.j*y)
    denominator = hl(l, 1.j*y)
    right = np.real(1.j * (y/x) * numerator / denominator)
    left = djl(l, x) / jl(l, x)
    return left - right

    
#def bound_states_dy(l, y, v0):
#    x = to_x(y, v0)
#    dx = to_x_dx(y, v0)
#    jy = jl(l, y)
#    djy = djl(l, y)
#    d2jy = d2jl(l, y)
#    hy = hl(l, y * 1.j)
#    dhy = dhl(l, y * 1.j)
#    d2hy = d2hl(l, y * 1.j)
#    return np.real(d2jy / jy - djy**2/jy**2 - y*d2hy/(x*hy) + y*dhy**2/(x*hy**2) + 1.j * y*dx*dhy/(x**2*hy)- 1.j * dhy/(x*hy))

def bound_states_dy(l, y, v0):
    # 1) x und dx/dy
    x     = to_x(y, v0)        # = sqrt(v0 - y^2)
    dx_dy = -y / x              # ∂x/∂y bei festem v0

    # 2) Linker Anteil: left(x) = djl/jl
    jl_x   = jl(l, x)
    djl_x  = djl(l, x)
    d2jl_x = d2jl(l, x)
    # ∂left/∂x
    dleft_dx = (d2jl_x * jl_x - djl_x**2) / (jl_x**2)
    left_term = dleft_dx * dx_dy

    # 3) Rechter Anteil: right = Re[A(x,y)*B(y)]
    # B(y) = dhl(i*y)/hl(i*y), hängt nur von y
    hy    = hl(l, 1j*y)
    dhy   = dhl(l, 1j*y)
    d2hy  = d2hl(l, 1j*y)
    B     = dhy / hy
    # ∂B/∂y via Quotientenregel und Kettenregel
    num    = 1j*d2hy * hy - dhy * 1j*dhy
    dB_dy  = num / (hy**2)

    # A(y,x) = i*y/x
    A      = 1j * y / x
    # ∂A/∂y = i*(1/x) + i*y * (∂(1/x)/∂y)
    #       = i/x + i*y*(y/x^3)
    dA_dy  = 1j*(1/x) + 1j*y*(y/(x**3))

    right_term = np.real(dA_dy * B + A * dB_dy)

    # 4) Gesamtableitung
    return left_term - right_term



#def bound_states_dv0(l, y, v0):
#    x = to_x(y, v0)
#    dx = to_x_dv0(y, v0)
#    hy = hl(l, 1.j * y)
#    dhy = dhl(l, 1.j * y)
#    return np.real(1.j * y*dhy*dx/(hy*x**2))

def bound_states_dv0(l, y, v0):
    x = to_x(y, v0)
    dx_dv0 = 1/(2*x)  # d(x)/dv0

    # Left-Term Ableitung
    jl_x = jl(l, x)
    djl_x = djl(l, x)
    d2jl_x = d2jl(l, x)
    dleft_dx = (d2jl_x * jl_x - djl_x**2) / (jl_x**2)
    left_term = dleft_dx * dx_dv0

    # Right-Term Ableitung
    hy = hl(l, 1.j*y)
    dhy = dhl(l, 1.j*y)
    quot = dhy / hy
    d_1_over_x_dv0 = -1/(2*x**3)
    right_term = np.real(1j * y * d_1_over_x_dv0 * quot)

    # Gesamtableitung:
    return left_term - right_term

#inside legt fest, ob r<r0 ist (für inside<1 der Fall, da inside=r/r0)
def radial_of_bound_states(l, x, a, b, A=1):
    radial_in = x* A * jl(l, a*x)
    radial_out = x* A * (jl(l, a)/hl(l, 1.j *b)) * hl(l, 1.j * b*x)
    return np.where(x <= 1, radial_in, radial_out)


#explizite Ordnungen ab hier 

def bound_states_l0(y, v0):
    x = np.sqrt(v0 - y**2)
    # Bessel j0 und Ableitung
    jl = np.sin(x)/x
    djl = np.cos(x)/x - np.sin(x)/x**2
    # Hankel h0 und Ableitung, z = 1j*y
    z = 1j*y
    hl = -np.exp(1j*z)/(1j*z)
    dhl = -np.exp(1j*z)/(1j*z)*(1j + 1/z)
    # rechte Seite
    right = np.real(1j * (y/x) * dhl / hl)
    # linke Seite
    left = djl / jl
    return left - right

def bound_states_dv0_l0(y, v0):
    x = np.sqrt(v0 - y**2)
    dx_dv0 = 1/(2*x)
    
    # Besselfunktion und Ableitungen für l=0
    jl = np.sin(x)/x
    djl = np.cos(x)/x - np.sin(x)/x**2
    d2jl = -np.sin(x)/x - 2*np.cos(x)/x**2 + 2*np.sin(x)/x**3

    # Left-Term Ableitung
    dleft_dx = (d2jl * jl - djl**2) / jl**2
    left_term = dleft_dx * dx_dv0

    # Hankelfunktion und Ableitung für l=0, z = 1j*y
    z = 1j*y
    hl = -np.exp(1j*z)/(1j*z)
    dhl = -np.exp(1j*z)/(1j*z) * (1j + 1/z)
    quot = dhl / hl  # vereinfacht sich zu (1j + 1/z)

    d_1_over_x_dv0 = -1/(2*x**3)
    right_term = np.real(1j * y * d_1_over_x_dv0 * quot)

    return left_term - right_term