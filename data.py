import numpy as np


# link data
n_links = 12
LINK = np.array(['l1','l2','l3','l4','l5','l6','l7','l8','l9','l10','l11','l12'])
lstartloc = np.array(['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n11','n12'])
lendloc = np.array(['n2','n3','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13'])
ldiam = np.array([920.0, 920.0, 920.0, 920.0, 920.0, 920.0, 920.0, 920.0, 920.0, 920.0, 920.0, 920.0])
llength = np.array([300.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 300.0])
ltype = np.array(['p','a','a','a','a','a','a','a','a','a','a','p'])

# node data
n_nodes = 13
NODE = np.array(['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13'])
pmin = np.array([57.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 39.0])
pmax = np.array([70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 41.0])

# supply data
n_sup = 1
SUP = np.array([1])
sloc = np.array(['n1'])
smin = np.array([0.0])
smax = np.array([30.0])

# demand data
n_dem = 1
DEM = np.array([1])
dloc = np.array(['n13'])
d = np.array([10])

# scaling factors
rhon = 0.72                                  # density of air at normal conditions - [kg/m3]
ffac = (1e+6*rhon)/(24*3600)                 # from scmx10-6/day to kg/s
ffac2 = (3600)/(1e+4*rhon)                   # from kg/s to scmx10-4/hr
pfac = 1e+5                                  # from bar to Pa
pfac2 = 1e-5                                 # from Pa to bar
dfac = 1e-3                                  # from mm to m
lfac =1e+3                                   # from km to m

# convert units for input data
ldiam = ldiam*dfac                    # from  mm to m
llength = llength*lfac                # from  km to m
smin = smin*ffac*ffac2                # from scmx106/day to kg/s and then to scmx10-4/hr
smax = smax*ffac*ffac2                # from scmx106/day to kg/s and then to scmx10-4/hr
d = d*ffac*ffac2                      # from scmx106/day to kg/s and then to scmx10-4/hr
pmin = pmin*pfac*pfac2                # from bar to Pascals and then to bar
pmax = pmax*pfac*pfac2                # from bar to Pascals and then to bar

eps = 0.025                                             # pipe rugosity - [mm]
lam = np.array([
    (2*np.log10(3.7*ldiam[i]/(eps*dfac)))**(-2.0)
                for i in range(len(LINK))])             # friction coefficient
A = (1/4)*np.pi*(ldiam**2)                              # pipe transveral area - [m^2]
Cp = 2.34                                               # heat capacity @ constant pressure [kJ/kg-K]
Cv = 1.85                                               # hat capacity @ constant volume [kJ/kg-K]
gam = Cp/Cv                                             # expansion coefficient [-]
z = 0.80                                                # gas compressibility  - []
R = 8314.0                                              # universal gas constant [J/kgmol-K]
Tgas = 293.15                                           # reference temperature [K]
M = 18.0                                                # gas molar mass [kg/kgmol]
nu2 = gam*z*R*Tgas/M                                    # gas speed of sound [m/s]
om = (gam-1)/gam                                        # aux constant [-]
c1 = (pfac2/ffac2)*(nu2/A)
c2 = A*(ffac2/pfac2)
c3 = np.array([A[i]*(pfac2/ffac2)*(8*lam[i]*nu2)/(np.pi*np.pi*(ldiam[i]**5)) for i in range(len(LINK))])
c4 = (1/ffac2)*(Cp*Tgas)

TF = 24*3600                          # horizon time - [s]
Nt = 48                               # number of temporal grid points
TIME = np.arange(1, Nt+1)             # set of temporal grid points
TIMEm = np.arange(1, Nt)              # set of temporal grid points minus 1
TDEC = 20                             # decision time step
Nx = 10                               # number of spatial grid points
DIS = np.arange(1, Nx+1)              # set of spatial grid points
S = 3                                 # number of scenarios
lambd = 0.9                           # trade-off exp value and cvar
dt = TF/Nt                            # temporal grid spacing - [s]
dx = llength/(Nx-1)                   # spatial grid spacing - [m]

# cost factors
ce = 0.1         # cost of compression [$/kWh]
cd = 1e6         # demand tracking cost [-]
cT = 1e6         # terminal constraint cost [-]
cs = 0           # supply cost [$/scmx10-4]

# stochastic demand
stochd = np.ones((S, n_dem, Nt))  # todo
