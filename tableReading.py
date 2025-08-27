import os
import os.path
import numpy as np

def f_name(typ, l, i):
    return f"tables/l={l}/trace_{typ}_{i}.csv"

def read_line(typ, l, i):
    '''
    Bestimmte Linie i in Konturplot von bestimmtem l für typ bs oder rs als 3 Arrays mit x, y und v0
    '''
    try:
        filename = f_name(typ, l, i)
        data = np.loadtxt(filename, delimiter = ",", skiprows=1)
        data = np.transpose(data)
        return data[0], data[1], data[2]
    
    except Exception as e: 
        print(f"Fehler beim Lesen der Datei Typ: {typ} mit l={l} und i={i}")
        raise e
    
#read_line("Resonanzen", 0, 1)

def read_order(typ, l):
    '''
    Alle Linien in Konturplot von bestimmtem l für typ bs oder rs als 3 Arrays mit x, y und v0
    '''
    i = 0
    x = []
    y = []
    v0 = []
    while os.path.isfile(f_name(typ, l, i)):
        n_x, n_y, n_v0 = read_line(typ, l, i)
        x.extend(n_x)
        y.extend(n_y)
        v0.extend(n_v0)
        i = i+1
    return np.array(x), np.array(y), np.array(v0)

#read_order("Resonanzen", 0)


def read_order_Resonanzen(l):
    return read_order("Resonanzen", l)

def read_order_Bound_States(l):
    return read_order("Bound States", l)

def read_line_Resonanzen(l, i):
    return read_line("Resonanzen", l, i)

def read_line_Bound_States(l, i):
    return read_line("Bound States", l, i)

def read_line_Bound_States2(l, i):                           
    return read_line("Bs_mit_trace_root", l, i)

def check_min_v0(typ, l, v0_max):
    '''
    Überprüfe, ob minimaler v0 Wert für bs oder rs kleiner ist als v0_max
    '''
    try:
        filename = f_name(typ, l, 0)
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        data = np.transpose(data)
        data2 = data[2]
        min_val = np.min(data2)
        print(min_val)
        if min_val < v0_max:
            #print("true")
            return True
        else:
            #print("false")
            return False

    except Exception as e:
        print(f"Fehler beim Lesen der Datei Typ: {typ} mit l={l} und i={i}")
        raise e

#check_min_v0("Resonanzen", 4, 100)

def find_ymin_for_min_v0(l):
    '''
    Finde den kleinsten v0 Wert für "Resonanzen" einer Ordnung
    '''
    try:
        filename = f_name("Resonanzen", l, 0)
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        # Spalte 2 enthält v0, Spalte 1 enthält y
        idx_min = np.argmin(data[:, 2])
        y_min   = data[idx_min, 1]
        print(y_min)
        return y_min
    except Exception as e:
        print(f"Fehler beim Lesen der Datei Typ: {typ} mit l={l} und i={i}")
        raise e

#find_ymin_for_min_v0(4)


def check_max_l(typ, v0_max):
    """
    Findet das maximale l, für das der minimale v0 kleiner als v0_max ist.
    """
    l = 0
    while True:
        result = check_min_v0(typ, l, v0_max)
        if not result:
            return l  # Das letzte l, das in v0 liegt
        l += 1

#check_max_l("Resonanzen", 20)


def find_xy_for_v0(typ, l, v0_input):
    x, y, v0 = read_order(typ, l)
    if len(x) != len(y) and len(y) != len(v0):
        print(f"Fehler: Die Länge der Arraysx, y, v0 ist verschieden für l={l}!")
    x_match = []
    y_match = []

    # Alle Crossings entsprechend v0 suchen
    for i in range(1, len(v0)):
        if v0[i-1] < v0_input and v0[i] >= v0_input:
            x_match.append(x[i])
            y_match.append(y[i])
    return x_match, y_match

#find_xy_for_v0("Resonanzen", 0, 100)