from Bessel import *


def to_x(y, v0):
	return np.sqrt(y**2 + v0)

def denominator(y, v0, l):
    x = to_x(y, v0)
    j_l_x = jl(l, x)
    j_l_prime_x = djl(l, x)
    n_l_y = yl(l, y)
    n_l_prime_y = dyl(l, y)
    return y * n_l_prime_y - (x * j_l_prime_x / j_l_x) * n_l_y


def numerator(y, v0, l):
    x = to_x(y, v0)
    j_l_x = jl(l, x)
    j_l_prime_x = djl(l, x)
    j_l_y = jl(l, y)
    j_l_prime_y = djl(l, y)
    return (x * j_l_prime_x / j_l_x) * j_l_y - y * j_l_prime_y

def to_x_dx(y, v0):  
    return y / np.sqrt(y * y + v0)


def to_x_dv0(y, v0):  
    return 0.5 / np.sqrt(y * y + v0)


#def denominator_dy(y, v0, l):  
#     g = to_x(y, v0)
#     g_prime = to_x_dx(y, v0)
#     jn = jl(l, g)
#     jn_prime = djl(l, g)
#     jn_prime_2 = d2jl(l, g)
#     yn = yl(l, y)
#     yn_prime = dyl(l, y)
#     yn_prime_2 = d2yl(l, y)
#
#     f1 = g * g_prime * yn * jn * jn_prime_2
#     f2 = g * g_prime * yn * jn_prime * jn_prime
#     f3 = (g * yn_prime + g_prime * yn) * jn * jn_prime
#     f4 = (-y * yn_prime_2 - yn_prime) * jn * jn
#     return -1.0 * (f1 - f2 + f3 + f4) / (jn * jn)


def denominator_dy(y, v0, l):
    x = to_x(y, v0)
    gx = to_x_dx(y, v0)                  # dx/dy

    j  = jl(l, x)
    jp = djl(l, x)
    jpp= d2jl(l, x)

    n  = yl(l, y)
    np = dyl(l, y)
    npp= d2yl(l, y)

    alpha      = x * jp / j
    alpha_prime= (jp/j) + x*(jpp/j) - x*(jp/j)**2   # d/dx [x jp/j]

    Fy  = y*npp + (1.0 - alpha)*np - alpha_prime*gx*n
    return Fy

def denominator_dv0(y, v0, l):  
     g = to_x(y, v0)
     g_prime = to_x_dv0(y, v0)
     jn = jl(l, g)
     jn_prime = djl(l, g)
     jn_prime_2 = d2jl(l, g)
     yn = yl(l, y)

     f1 = g * jn * jn_prime_2
     f2 = g * jn_prime * jn_prime
     f3 = jn * jn_prime
     return -1.0 * yn * g_prime * (f1 - f2 + f3) / (jn * jn)


def radial_of_resonance(l, x, a, b, A=1):
    denom = jl(l, b) * dyl(l, b) * b - djl(l, b) * yl(l, b) * b

    B = (jl(l, a) * dyl(l, b) * b - a * djl(l, a) * yl(l, b)) / denom
    C = (a * djl(l, a) * jl(l, b) - jl(l, a) * djl(l, b) * b) / denom           #Koeffizienten der Wellenfunktion abhängig von A

    radial_in = x* A * jl(l, a*x)
    radial_out = x* A * (B * jl(l, b*x) + C * yl(l, b*x))           #x ist hier r/r0, wonach geplottet wird. a entspricht dem x und b dem y aus dem Konturplot
    return np.where(x <= 1, radial_in, radial_out)
