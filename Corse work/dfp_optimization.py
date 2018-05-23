import numpy as np
from onedimsearch import svenn
from onedimsearch import golden_ratio
from onedimsearch import dsk_powell


def dfp(x0, f, epsilon, h, grad_f, one_dim, epsilon_onedim, svenn_param):

    def delta_x(i):
        return (x[i+1] - x[i])[None,:]

    def delta_g(i):
        return (grad[i+1] - grad[i])[None,:]
    
    x = np.array([x0])
    A0 = np.array([[1,0], [0,1]])
    A = np.array([A0])

    grad = [grad_f(x[0])]
    S0 = -grad[0]
    S = np.array([S0])

    a, b = svenn(f, x[0], S[0], svenn_param)
    l1 = one_dim(f, a, b, x[0], S[0], epsilon_onedim)
    lambda_ = [l1]

    x1 = x[0] + lambda_[0] * S[0]
    x = np.append(x, [x1], axis=0)

    i = 0
    f_values = [f(x[0]), f(x[1])]

    while (abs(f_values[i+1] - f_values[i]) >= epsilon or (np.linalg.norm(x[i+1] - x[i]) / np.linalg.norm(x[i])) >= epsilon): 

        grad.append(grad_f(x[i+1]))

        delta_xi, delta_gi = delta_x(i), delta_g(i)
        Acor = np.dot(delta_xi.T, delta_xi) / np.dot(delta_xi, delta_gi.T) - np.dot(np.dot(np.dot(A[i], delta_gi.T), delta_gi), A[i]) / np.dot(np.dot(delta_gi, A[i]), delta_gi.T)

        A = np.append(A, [A[i] + Acor], axis=0)
        S = np.append(S, [- np.dot(A[i+1], grad[i+1])], axis=0)

        a, b = svenn(f, x[i+1], S[i+1], svenn_param)
        l1 = one_dim(f, a, b, x[i+1], S[i+1], epsilon_onedim)

        #restart
        if l1 <= 0.001: 
            A[i+1] = A0
            S[i+1] = - grad[i+1]
            a, b = svenn(f, x[i+1], S[i+1], svenn_param)
            l1 = one_dim(f, a, b, x[i+1], S[i+1], epsilon_onedim)

        lambda_.append(l1)

        x_next = x[i+1] + lambda_[i+1] * S[i+1]
        x = np.append(x, [x_next], axis=0)
        f_values.append(f(x[i+2]))
        i += 1
    # print(x)
        
    return x[-1]

