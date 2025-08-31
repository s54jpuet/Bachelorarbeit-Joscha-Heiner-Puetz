from scipy.optimize import root_scalar
import numpy as np

TOLERANCE = 1e-12
MAX_ITERATIONS = 200

def roots_y(f, fy, y, min_v0, max_v0, step):  #fy ist partielle Ableitung der eingebenen Funktion
    roots = []
    v0 = min_v0
    lastvalue = f(y, v0)
    while v0 <= max_v0:
        if type(step) is float:
            s = step
        else:
            s = step(v0)
        v0 += s
        value = f(y, v0)
        if lastvalue * value < 0:
            dvalue = fy(y, v0)
            if dvalue * value > 0:
                sol = root_scalar(
                    lambda v0: f(y, v0),
                    bracket= [v0 - s, v0],
                    method='brentq',
                    xtol = TOLERANCE
                )

                if sol.converged:
                    roots.append(sol.root)
        lastvalue = value
    return roots


def find_close_root(f, fx, fy, initial_x, initial_y):
    '''
    x ist im Koordinatensystem nach rechts, y nach oben (nicht die verwendeten Substitutionen aus der Bachelorarbeit)
    '''
    x = initial_x
    y = initial_y
    found = False
    for _ in range(MAX_ITERATIONS):
        denom = f(x, y)
        dx = fx(x, y)
        dy = fy(x, y)
        length_squared = dx * dx + dy * dy
        x -= denom / length_squared * dx
        y -= denom / length_squared * dy

        if np.abs(denom) < 1e-8:
            found = True
            break
    if not found:
        print("could not find close root")        
    return (x, y, found)
