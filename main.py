import numpy as np
import pyipopt
import datetime
from functions import *


bnds, x0 = get_bounds_and_initialx()
nvar = len(x0)
print('number of variables:', nvar)
x_L = np.zeros(nvar)
x_U = np.zeros(nvar)
for i in range(nvar):
    x_L[i] = -1e10 if bnds[i][0] is None else bnds[i][0]
    x_U[i] = 1e10 if bnds[i][1] is None else bnds[i][1]
# constraints
neq = len(eq_constr(x0))+len(compr_eq(x0))
ncon = neq+len(ineq_constr(x0))
nnzh = S*(n_dem*Nt+n_links*Nx*4+n_links*Nx*4)


def eval_g(x):
    return np.append(np.append(eq_constr(x), compr_eq(x)), ineq_constr(x))


def eval_jac_g(x, flag):
    ret = np.append(np.append(eq_constr_jac(x), compr_eq_jac(x), axis=0), ineq_constr_jac(x), axis=0)
    if flag:
        print('hi', np.sum(ret))
        print('nnzrj', np.count_nonzero(ret))
        return ret.flatten()
    else:
        print('warning', np.shape(np.trim_zeros(ret.flatten())))
        return np.trim_zeros(ret.flatten())


nnzj = np.count_nonzero(eval_jac_g(x0, True))
print(nnzj)
g_L = np.zeros((len(eval_g(x0))))
g_U = np.zeros_like(g_L)
g_U[neq:] = 1e10
nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, avg_cost, avg_cost_grad, eval_g, eval_jac_g)


nlp.num_option('bound_relax_factor', 0.1)
nlp.str_option("mu_strategy", "adaptive")
nlp.str_option("derivative_test", "first-order")
nlp.str_option('warm_start_init_point', 'yes')
nlp.str_option('linear_solver', 'mumps')
print(datetime.datetime.now(), ": Going to call solve")
x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
nlp.close()
print(x)




