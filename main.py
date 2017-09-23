import numpy as np
import datetime
from scipy.optimize import minimize
from functions import *


bnds, x0 = get_bounds_and_initialx()
print(len(x0))
con1 = {'type': 'ineq', 'fun': lambda x: -eq_constr(x)**2}
con2 = {'type': 'ineq', 'fun': lambda x: ineq_constr(x)}
con3 = {'type': 'ineq', 'fun': lambda x: -compr_eq(x)**2}


def bnds_constr(x):
    ret = []
    for i in range(len(x)):
        if bnds[i][0] is not None:
            ret.append(x[i]-bnds[i][0])
        if bnds[i][1] is not None:
            ret.append(bnds[i][1]-x[i])
    return np.array(ret)
con4 = {'type': 'ineq', 'fun': bnds_constr}
cons = [con1, con2, con3, con4]
options = {'maxiter': 10000}
print(datetime.datetime.now())
sol = minimize(avg_cost, x0, constraints=cons, method='COBYLA', options=options)
print(sol.success)
print(sol.message)
#print(sol.nit)