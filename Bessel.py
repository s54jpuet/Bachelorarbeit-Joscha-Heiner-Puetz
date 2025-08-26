import numpy as np
from scipy.special import spherical_jn, spherical_yn

def jl(l, x):
    if l==0:
        return np.sin(x)/x
    else:
        return spherical_jn(l, x)
    
def yl(l, x):
    if l==0:
        return -np.cos(x)/x
    else:
        return spherical_yn(l, x)

def djl(l, x):
    if l==0:
        return (x * np.cos(x) - np.sin(x)) / x**2
    else:
        return spherical_jn(l, x, derivative=True)
    
def dyl(l, x):
    if l==0:
        return (np.sin(x) /x + np.cos(x) / x**2)
    else:
        return spherical_yn(l, x, derivative=True)

def d2jl(l, x):
    if l == 0:
        return ( (2 - x**2) * np.sin(x) - 2*x*np.cos(x) ) / x**3
    else:
        return -djl(l+1, x) + (l/x)*djl(l, x) - (l/x**2)*jl(l, x)

def d2yl(l, x):
    if l == 0:
        return (-2 * np.sin(x) / x**2) + (np.cos(x) / x) - (2 * np.cos(x) / x**3)
    else:
        return -dyl(l+1, x) + (l/x)*dyl(l, x) - (l/x**2)*yl(l, x)

#Hankel usw.

def hl(l, x):
    return jl(l, x) + 1.j* yl(l,x)


def dhl(l, x):
    return djl(l, x) + 1.j* dyl(l,x)

def d2hl(l, x):
    return d2jl(l, x) + 1.j* d2yl(l,x)


