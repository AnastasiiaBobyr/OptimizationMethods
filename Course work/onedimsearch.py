import numpy as np

def svenn(f, x0, S0, delta_param):
    x = np.array([x0])
    S = np.array([S0])
    
    lambd = [0]
    delta =  delta_param * np.linalg.norm(x[0]) / np.linalg.norm(S[0]) 
    
    #choose direction and find first value
    f1 = f(x[0] + (lambd[0] - delta) * S[0])
    f2 = f(x[0] + lambd[0] * S[0])
    f3 = f(x[0] + (lambd[0] + delta) * S[0])
    
    f_values = [f2]
    
    if f1 >= f2 >= f3:
        delta = delta
        lambd.append(lambd[0] + delta)
        f_values.append(f3)
    elif f1 <= f2 <= f3:
        delta = - delta
        lambd.append(lambd[0] + delta)
        f_values.append(f1)
    elif f1 >= f2 <= f3:
        a = lambd[0] - delta
        b = lambd[0] + delta
        f_values.append(f1)
        f_values.append(f3)
        return a, b
    else:
        print('Function is not unimodal')

    x = np.append(x, [x[0] + lambd[1] * S[0]], axis=0)

    #find all other values except first and last
    i = 1
    while f_values[i] <= f_values[i - 1]:
        lambd.append(lambd[i] + (2 ** i) * delta)
        x = np.append(x, [x[0] + lambd[i + 1] * S[0]], axis=0)
        f_values.append(f(x[0] + lambd[i + 1] * S[0]))
        i += 1
        
    #find last value
    lambd.append(lambd[i - 1] + (lambd[i] - lambd[i - 1]) / 2)
    x = np.append(x, [x[0] + lambd[i + 1] * S[0]], axis=0)
    f_values.append(f(x[0] + lambd[i + 1] * S[0]))
    
    #sort lambdas
    lambda_and_f = dict(zip(lambd, f_values))
    lambda_sorted = sorted(lambda_and_f)
    f_values_sorted = []
    for key in lambda_sorted:
        f_values_sorted.append(lambda_and_f[key])

    m = min(f_values_sorted)
    a = lambda_sorted[f_values_sorted.index(m) - 1]
    b = lambda_sorted[f_values_sorted.index(m) + 1]
    
    
    return a, b


def golden_ratio(f, a, b, x0, S0, epsilon):
    
    x = np.array([x0])
    S = np.array([S0])
    L = b - a
    
    x1 = a + 0.382 * L
    x2 = a + 0.618 * L

    f1 = f(x[0] + x1 * S[0])
    f2 = f(x[0] + x2 * S[0])
    
    while L >= epsilon:
        if f1 <= f2:
            a = a
            b = x2
            L = b - a
            x2 = x1
            f2 = f1
            x1 = a + 0.382 * L
            f1 = f(x[0] + x1 * S[0])
        elif f1 >= f2:
            a = x1
            b = b
            L = b - a
            x1 = x2
            f1 = f2
            x2 = a + 0.618 * L
            f2 = f(x[0] + x2 * S[0])

    lambda_top = (a + b) / 2
    return lambda_top


def dsk_powell(f, a, b, xw, S0, epsilon):
    xq = np.array([xw])
    S = np.array([S0])
    x1 = a
    x2 = (a + b) / 2
    x3 = b

    f1 = f(xq[0] + x1 * S[0])
    f2 = f(xq[0] + x2 * S[0])
    f3 = f(xq[0] + x3 * S[0])
    
    x_star = x2 + (abs(x2 - x1) * (f1 - f3) / (2 * (f1 - 2 * f2 + f3)))
    f_star = f(xq[0] + x_star * S[0])
    
    x = [x1, x2, x_star, x3]
    f_values = [f1, f2, f_star, f3]
    
    x_and_f = dict(zip(x, f_values))
    x_sorted = sorted(x_and_f)
    f_values_sorted = []
    for key in x_sorted:
        f_values_sorted.append(x_and_f[key])
        
    x_min = x2
    f_min = f_values_sorted[x_sorted.index(x2)]
    f_star = f_values_sorted[x_sorted.index(x_star)]

    while abs(x_min - x_star) >= epsilon and abs(f_min - f_star) >= epsilon:

        f_min = min(f_values_sorted)

        x1 = x_sorted[f_values_sorted.index(f_min) - 1]
        x2 = x_sorted[f_values_sorted.index(f_min)]
        x3 = x_sorted[f_values_sorted.index(f_min) + 1]
        x_min = x2
        f1 = f_values_sorted[f_values_sorted.index(f_min) - 1]
        f2 = f_min
        f3 = f_values_sorted[f_values_sorted.index(f_min) + 1] 

        if abs(x3 - x2) == abs(x2 - x1):
            x_star = x2 + (abs(x2 - x1) * (f3 - f1)) / (2 * (f1 - 2 * f2 + f3))
        else:
            a1 = (f2 - f1) / (x2 - x1)
            a2 = (1 / (x3 - x2)) * ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1))
            x_star = ((x1 + x2) / 2) - (a1 / (2 * a2))
        
        if abs(x_min - x_star) >= epsilon:
            f_star = f(xq[0] + x_star * S[0])
        else: f_star = f2
            
        x = [x1, x2, x_star, x3]
        f_values = [f1, f2, f_star, f3]

        x_and_f = dict(zip(x, f_values))
        x_sorted = sorted(x_and_f)
        f_values_sorted = []
        for key in x_sorted:
            f_values_sorted.append(x_and_f[key])
        
    return x_star