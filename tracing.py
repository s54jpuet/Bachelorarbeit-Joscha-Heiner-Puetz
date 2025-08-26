import numpy as np
from roots_y import *

MAX_ITERATIONS = 10000000
WRITE_DELTA = 0.05
MOVE_DELTA =  0.05
TOLERANCE = 1e-12


def trace_root(initial_x, initial_y, f, fx, fy, distance, max_iterations = MAX_ITERATIONS, write_delta = WRITE_DELTA, move_delta = MOVE_DELTA):
    """
    Verfolgt eine Nullstellenkurve durch Bewegung orthogonal zum Gradienten und sucht dabei Schritt für Schritt nächstgelegene Fortsetzungen
    """
    rootsx = [initial_x]
    rootsy = [initial_y]
    moved_dist = 0.0
    iter_count = 0
    last_x = initial_x
    x,  y = initial_x, initial_y
    while moved_dist < distance:
        dx = fx(x, y)
        dy = fy(x, y)
        length = np.sqrt(dx * dx + dy * dy)
        if length < TOLERANCE or np.abs(dy) < TOLERANCE:
            print("This seems to be a local extremum. We cannot move anywhere with confidence!")
            break
        dx, dy = dx / length, dy / length

        sign = np.sign(dy)
        move_x = sign * move_delta * dy
        move_y = -sign * move_delta * dx

        #xf = lambda y: f(x + move_x, y)
        #xdf = lambda y: fy(x + move_x, y)
        #print(f"moving from ({x}, {y}) to (nx, ny) = ({x + move_x}, {y + move_y}): f(nx, ny) = {f(x+move_x, y+move_y)}")
        new_x, new_y, found = find_close_root(f, fx, fy, x + move_x, y + move_y)
        if found:
            x, y = new_x, new_y
        else:
            print(f"Aborted trying to find a root at ({x}, {y}) because no close root could be found.")
            break
        delta = x - last_x
        iter_count += 1

        if delta >= write_delta:
            moved_dist += delta
            last_x = x
            rootsx.append(x)
            rootsy.append(y)
            iter_count = 0
        elif iter_count > max_iterations:
            print(f"Aborted root search at x={x} due to reaching the max iteration count of {max_iterations}")
            break
    return np.array(rootsx), np.array(rootsy)
