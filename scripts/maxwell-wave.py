# maxwell-wave.py
# Attempt at time-dependent solver of maxwell eqs in form given by James Cook (really scalar wave eq)
# time-stepper was based loosely on the one in:
# https://www.firedrakeproject.org/demos/DG_advection.py.html

from firedrake import *
import math
import time
import numpy

meshres = 64
mesh = PeriodicSquareMesh(meshres, meshres, 1.0, quadrilateral=True)
V1 = FunctionSpace(mesh, "CG", 2)

# time parameters (used 2.0 / 1000 for order 2 elt side 1/64)
T = 2.0
timeres = 1000
t = Constant(0.0)
dt = T/timeres

# model parameters
k_init = 2.0*2.0*pi+0.0  # wavenumber for initial data

x, y = SpatialCoordinate(mesh)
f1 = Function(V1) # f1 is wave field at step-1
f2 = Function(V1)  # f2 is wave field at step
v1 = TestFunction(V1)

# wave init data - determines how the wave will propagate
f1.interpolate(sin(k_init*(y-dt)))
f2.interpolate(sin(k_init*(y))) 

# TRIALCODE output init data for checking
#File("maxwell_wave_init.pvd").write(f1, f2)
#quit()

f_trial = TrialFunction(V1)
a = (f_trial*v1)*dx

L1= (2*f2*v1 - (dt**2)*inner(grad(f2),grad(v1)) - f1*v1)*dx

outfile = File("maxwell-wave.pvd")

cnt=0
start = time.time()

f_new = Function(V1)

params = {'ksp_type': 'preonly', "pc_type": "lu"}  # block Jacobi here gives numerical artifacts, so don't!
prob1 = LinearVariationalProblem(a, L1, f_new)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)

t = 0.0
step = 0
output_freq = 10
while t < T-0.5*dt:
    solv1.solve()
    f1.assign(f2)
    f2.assign(f_new)

    step += 1
    t += dt
    if step % output_freq == 0:
        outfile.write(f1, f_new)
        print("done step")
        print(step)
        print("\n")

end = time.time()
wall_time = end-start

File("maxewll-wave_final.pvd").write(f1, f2)

print("done.")
print("\n")
print("wall time:"+str(wall_time)+"\n")



