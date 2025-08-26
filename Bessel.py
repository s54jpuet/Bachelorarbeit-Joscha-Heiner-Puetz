import numpy as np
from scipy.special import spherical_jn, spherical_yn

# sphärische Bessel-Funktion 
def jl(l, x):
    if l==0:
        return np.sin(x)/x
    else:
        return spherical_jn(l, x)

# sphärische Neumann-Funktion
def yl(l, x):
    if l==0:
        return -np.cos(x)/x
    else:
        return spherical_yn(l, x)

# 1. Ableitung sphärische Bessel-Funktion 
def djl(l, x):
    if l==0:
        return (x * np.cos(x) - np.sin(x)) / x**2
    else:
        return spherical_jn(l, x, derivative=True)

# 1. Ableitung sphärische Neumann-Funktion
def dyl(l, x):
    if l==0:
        return (np.sin(x) /x + np.cos(x) / x**2)
    else:
        return spherical_yn(l, x, derivative=True)

# 2. Ableitung sphärische Bessel-Funktion 
def d2jl(l, x):
    if l == 0:
        return ( (2 - x**2) * np.sin(x) - 2*x*np.cos(x) ) / x**3
    else:
        return -djl(l+1, x) + (l/x)*djl(l, x) - (l/x**2)*jl(l, x)

# 2. Ableitung sphärische Neumann-Funktion
def d2yl(l, x):
    if l == 0:
        return (-2 * np.sin(x) / x**2) + (np.cos(x) / x) - (2 * np.cos(x) / x**3)
    else:
        return -dyl(l+1, x) + (l/x)*dyl(l, x) - (l/x**2)*yl(l, x)

# sphärische Hankel-Funktion
def hl(l, x):
    return jl(l, x) + 1.j* yl(l,x)

# 1. Ableitung sphärische Hankel-Funktion
def dhl(l, x):
    return djl(l, x) + 1.j* dyl(l,x)

# 2. Ableitung sphärische Hankel-Funktion
def d2hl(l, x):
    return d2jl(l, x) + 1.j* d2yl(l,x)


