import numpy as np
from scipy.optimize import minimize
from functions import *


bnds, x0 = get_bounds_and_initialx()
print(len(x0))
con1 = {'type': 'eq', 'fun': lambda x: eq_constr(x)**2}
con2 = {'type': 'ineq', 'fun': ineq_constr}
cons = [con1, con2]
sol = minimize(lambda x: avg_cost(x)**2, x0, bounds=bnds, constraints=cons)
print(sol.success)
print(sol.message)
print(sol.nit)