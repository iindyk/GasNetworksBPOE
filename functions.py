#  contains all functions used for optimization
import numpy as np
from data import*


def parse_x(x):
    # node pressure - [bar]
    p = np.reshape(x[:S*n_nodes*Nt], (S, n_nodes, Nt))
    # compressor boost - [bar]
    dp = np.reshape(x[S*n_nodes*Nt:S*n_nodes*Nt+S*(n_links-2)*Nt], (S, n_links-2, Nt))
    # flow in pipe - [scmx10-4/hr]
    fin = np.reshape(x[S*Nt*(n_nodes+n_links-2):S*Nt*(n_nodes+2*n_links-2)], (S, n_links, Nt))
    # flow out pipe - [scmx10-4/hr]
    fout = np.reshape(x[S*Nt*(n_nodes+2*n_links-2):S*Nt*(n_nodes+3*n_links-2)], (S, n_links, Nt))
    # supply flow - [scmx10-4/hr]
    s = np.reshape(x[S*Nt*(n_nodes+3*n_links-2):S*Nt*(n_nodes+3*n_links-2+n_sup)], (S, n_sup, Nt))
    # demand flow - [scmx10-4/hr]
    dem = np.reshape(x[S*Nt*(n_nodes+3*n_links-2+n_sup):S*Nt*(n_nodes+3*n_links-2+n_sup+n_dem)], (S, n_dem, Nt))
    # compressor power [kW]
    pw = np.reshape(x[S*Nt*(n_nodes+3*n_links-2+n_sup+n_dem):S*Nt*(n_nodes+4*n_links-4+n_sup+n_dem)], (S, n_links-2, Nt))
    # auxiliary variable
    slack = np.reshape(x[S*Nt*(n_nodes+4*n_links-4+n_sup+n_dem):S*Nt*(n_nodes+4*n_links-4+n_sup+n_dem+n_links*Nx)],
                       (S, n_links, Nt, Nx))
    # link pressure profile - [bar]
    px = np.reshape(x[S*Nt*(n_nodes+4*n_links-4+n_sup+n_dem+n_links*Nx):S*Nt*(n_nodes+4*n_links-4+n_sup+n_dem+2*n_links*Nx)],
                    (S, n_links, Nt, Nx))
    # link flow profile - [scmx10-4/hr]
    fx = np.reshape(x[S*Nt*(n_nodes+4*n_links-4+n_sup+n_dem+2*n_links*Nx):S*Nt*(n_nodes+4*n_links-4+n_sup+n_dem+3*n_links*Nx)],
                    (S, n_links, Nt, Nx))
    return p, dp, fin, fout, s, dem, pw, slack, px, fx


def get_bounds_and_initialx():
    x0 = np.ones(S*Nt*(n_nodes+4*n_links-4+n_sup+n_dem+3*n_links*Nx))
    bnds = []
    # node pressure - [bar]
    pmax_arr = np.ones((S, n_nodes, Nt))
    pmin_arr = np.ones((S, n_nodes, Nt))
    for i in range(n_nodes):
        pmax_arr[:, i, :] = pmax[i]
        pmin_arr[:, i, :] = pmin[i]
    pmax_arr = np.reshape(pmax_arr, S*n_nodes*Nt)
    pmin_arr = np.reshape(pmin_arr, S*n_nodes*Nt)
    x0[:S * n_nodes * Nt] = 50.0
    for i in range(S*n_nodes*Nt):
        bnds.append((pmin_arr[i], pmax_arr[i]))
    # compressor boost - [bar]
    x0[S * n_nodes * Nt:S * n_nodes * Nt + S * (n_links - 2) * Nt] = 10.0
    for i in range(S * (n_links - 2) * Nt):
        bnds.append((0.0, 100.0))
    # flow in pipe - [scmx10-4/hr]
    x0[S * Nt * (n_nodes + n_links - 2):S * Nt * (n_nodes + 2 * n_links - 2)] = 100.0
    for i in range(S*n_links*Nt):
        bnds.append((1.0, 500.0))
    # flow out pipe - [scmx10-4/hr]
    x0[S * Nt * (n_nodes + 2 * n_links - 2):S * Nt * (n_nodes + 3 * n_links - 2)] = 100.0
    for i in range(S*n_links*Nt):
        bnds.append((1.0, 500.0))
    # supply flow - [scmx10-4/hr]
    x0[S * Nt * (n_nodes + 3 * n_links - 2):S * Nt * (n_nodes + 3 * n_links - 2 + n_sup)] = 10.0
