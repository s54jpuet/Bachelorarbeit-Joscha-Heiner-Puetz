from scipy.optimize import root_scalar
import numpy as np

TOLERANCE = 1e-12
MAX_ITERATIONS = 200

def roots_y(f, fy, y, min_v0, max_v0, step):  #fy ist Ableitung 
    roots = []
    v0 = min_v0
    lastvalue = f(y, v0)
    #print("root search")
    while v0 <= max_v0:
        if type(step) is float:
            s = step
        else:
            s = step(v0)
        v0 += s
        value = f(y, v0)
        #print(f"{value}")
        if lastvalue * value < 0:
            dvalue = fy(y, v0)
            #print(f"possible root in [{v0 - step}, {v0}] with f={value}, df={dvalue}")
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
    x ist im koordinatensystem nach rechts, y nach oben
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

        if np.abs(denom) < TOLERANCE:
            found = True
            break
    if not found:
        print("could not find close root")        
    return (x, y, found)


def find_roots(f, df, min_x, max_x, step):  #fy ist Ableitung 
    roots = []
    x = min_x
    lastvalue = f(x)
    print("root search")
    while x <= max_x:
        x += step
        value = f(x)
        print(f"{value}")
        if lastvalue * value < 0:
            dvalue = df(x)
            print(f"possible root in [{x - step}, {x}] with f={value}, df={dvalue}")
            if dvalue * value < 0:
                sol = root_scalar(
                    f,
                    bracket= [x - step, x],
                    method='brentq',
                    xtol = TOLERANCE
                )

                if sol.converged:
                    roots.append(sol.root)
        lastvalue = value
    return roots