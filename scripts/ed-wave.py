# ed-wave
# electromagnetic wave propagation with dielectrics
# using FEEC
# Ed Threlfall 1 August 2024

from firedrake import *
from irksome import Dt, MeshConstant, TimeStepper, RadauIIA, GaussLegendre
import time

mesh = Mesh("square_circ.msh")

circle_eps = 10.0  # dielectric const in circle

x,y = SpatialCoordinate(mesh)

T = 8.0
timeres = 80

t = Constant(0.0)
dt = Constant(T/timeres)

# parameters for irksome
# don't use RadauIIA ones - much dissipation
butcher_tableau = GaussLegendre(1)

# (at least) two nice choices for storing the vector (I expect DG will work also but that needs num flux implementing so it's a bit more work ...) 

# 1) Raviart-Thomas edge elements - order should be same as that of space SS below
SV=FunctionSpace(mesh,"RTE",1)  # note element type N1curl is equivalent to RTE

# 2) Brezzi-Douglas-Marini edge elements - order should be one less than that of space SS below
#SV=FunctionSpace(mesh, "BDME",1)  # note element type N2curl is equivalent to BDME

# ... there is also the "obvious" choice of CG function space, which does not work well:
#SV=VectorFunctionSpace(mesh, "CG", 1)

SS=FunctionSpace(mesh,"CG",1)  # order should be 1, 2, 3, ...
V = SV*SS

# mixed formulation with B_z a scalar in the plane and E a vector in the plane
EB = Function(V)
e, b = TestFunctions(V)

E, B = split(EB)

# spatial dependence of dielectric constant - must be elementwise constant and able to be discontinuous between elements
SS2 = FunctionSpace(mesh,"DG",0)
eps = Function(SS2)
eps_rule = conditional(le(x*x+y*y,0.0625), circle_eps, 1.0)  # circle
eps.interpolate(eps_rule)

#a=( inner(sigma, tau) - inner(u*eps, grad(tau)) +inner(grad((sigma-dot(grad(eps),u))/eps),v)+inner(curl(u),curl(v))) *dx

# need to set initial data e.g. wave condition
k = 30
sigma_longt = 0.1
offset = -0.7
sigma_trans = 0.25
EB.sub(0).interpolate(as_vector([0.0,sin(k*x)*exp(-((x-offset)**2/(2*(sigma_longt**2))))*exp(-((y)**2/(2*(sigma_trans**2))))]))
EB.sub(1).interpolate(sin(k*x)*exp(-((x-offset)**2/(2*(sigma_longt**2))))*exp(-((y)**2/(2*(sigma_trans**2)))))

# TRIALCODE output init data
File("ed_wave_init.pvd").write(EB.sub(0), EB.sub(1))
#quit()

F = inner(Dt(E),e)*dx + Dt(B)*b*dx \
  + (grad(E[1])[0]-grad(E[0])[1])*b*dx \
  + (-grad(B)[1]/eps)*e[0]*dx + (grad(B)[0]/eps)*e[1]*dx

# this is intended to be direct solver
linparams = {"mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu"}

bc0 = DirichletBC(V.sub(0), as_vector([0,0]), "on_boundary")

stepper = TimeStepper(F, butcher_tableau, t, dt, EB, solver_parameters=linparams, bcs=[bc0])

outfile = File("ed-wave.pvd")

cnt=0
start = time.time()

while float(t) < float(T):
    if (float(t) + float(dt)) >= T:
        dt.assign(T - float(t))
    if(cnt % 1 == 0):
       print("outputting data ...\n")
       Es, Bs = EB.split()
       outfile.write(Es, Bs)
    cnt=cnt+1
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(float(t), float(dt))

end = time.time()
wall_time = end-start

print("done.")
print("\n")
print("wall time:"+str(wall_time)+"\n")


