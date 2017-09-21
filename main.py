import numpy as np
from scipy.optimize import minimize
from functions import *


bnds, x0 = get_bounds_and_initialx()
con1 = {'type': 'eq', 'fun': eq_constr}
con2 = {'type': 'ineq', 'fun': ineq_constr}
cons = [con1, con2]
sol = minimize(avg_cost, x0, bounds=bnds, constraints=cons)
print(sol.success)
print(sol.message)
print(sol.nit)