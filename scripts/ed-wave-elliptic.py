# ed-wave-elliptic
# electromagnetic wave propagation with dielectric
# using FEEC
# extra: optionally do a scan over wavenumber and see the eigenmodes appear
# extra: attempt at finding eigenmodes of the square at end of file (currently disabled)
# Ed Threlfall 7 August 2024

from firedrake import *
import time

#mesh = RectangleMesh(200, 200, 1.0, 1.0, -1.0, -1.0)  # for trying out plane waves
mesh = Mesh("square_circ.msh")

circle_eps = 10.0  # dielectric const in circle SET TO 1.0 FOR PLANE WAVE
theta = 0*pi/180   # plane wave incidence angle to positive x-direction
kmag = 3.0+0.0     # in eps=1 case avoid sqrt(m^2+n^2) for integers m,n or get pollution from eigenmode!
kx = (2*pi*kmag/2)*cos(theta)
ky = (2*pi*kmag/2)*sin(theta)

x,y = SpatialCoordinate(mesh)

# 1) Raviart-Thomas edge elements - order should be same as that of space SS below
SV=FunctionSpace(mesh,"RTE",1)  # note element type N1curl is equivalent to RTE

# 2) Brezzi-Douglas-Marini edge elements - order should be one less than that of space SS below
#SV=FunctionSpace(mesh, "BDME",1)  # note element type N2curl is equivalent to BDME

 # ... there is also the "obvious" choice of CG function space, which does not work well:
#SV=VectorFunctionSpace(mesh, "CG", 1)

SS=FunctionSpace(mesh,"CG",1)  # order should be 1, 2, 3, ...
V = SV*SS

# mixed formulation with B_z scalar in the plane and E a vector in the plane, u is E, sigma is div D
usigma = Function(V)
v, tau = TestFunctions(V)

u, sigma = split(usigma)

# spatial dependence of dielectric constant - must be elementwise constant and able to be discontinuous between elements
SS2 = FunctionSpace(mesh,"DG",0)
eps = Function(SS2)
eps_rule = conditional(le(x*x+y*y,0.0625), circle_eps, 1.0)  # circle
eps.interpolate(eps_rule)

# boundaries: 13 is lhs, then go round clockwise in ordering
# this boundary condition gives a pure plane wave in the eps=1 case
# it's a BC on (curl E) dot (v cross n)
# it's the surface term from integrating curl curl E dot v -> - curl E dot curl v
# with RTE elements, cannot use the DirichletBC object generally (perp component of E can't be specified that way)
a = ( inner(sigma, tau) - div(eps*u)*tau +sigma*div(v)+inner(curl(u),curl(v))-(kx*kx+ky*ky)*eps*inner(u,v)) *dx \
  - ((cos(theta)*kx+sin(theta)*ky)*cos(kx*x+ky*y)*(-v[1]))*ds(13) \
  - ((cos(theta)*kx+sin(theta)*ky)*cos(kx*x+ky*y)*(-v[0]))*ds(14) \
  - ((cos(theta)*kx+sin(theta)*ky)*cos(kx*x+ky*y)*( v[1]))*ds(15) \
  - ((cos(theta)*kx+sin(theta)*ky)*cos(kx*x+ky*y)*( v[0]))*ds(16) \

# this is intended to be direct solver
linparams = {"mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu"}

outfile = File("ed-wave-elliptic.pvd")

start = time.time()

solve(a==0, usigma, solver_parameters=linparams)

outfile.write(usigma.sub(0), usigma.sub(1))

end = time.time()
wall_time = end-start

print("done.")
print("\n")
print("wall time:"+str(wall_time)+"\n")

quit()

# there follows code to scan in wavenumber and output the solutions
# this is like shooting method, it shows the eigenfunctions as resonances

animfile = File("ed-wave-elliptic_anim.pvd")
for i in range (1,1000):
   kmag = float(i)/50
   kx = (2*pi*kmag/2)*cos(theta)
   ky = (2*pi*kmag/2)*sin(theta)
   a = ( inner(sigma, tau) - div(eps*u)*tau +sigma*div(v)+inner(curl(u),curl(v))-(kx*kx+ky*ky)*eps*inner(u,v)) *dx \
     - ((cos(theta)*kx+sin(theta)*ky)*cos(kx*x+ky*y)*(-v[1]))*ds(13) \
     - ((cos(theta)*kx+sin(theta)*ky)*cos(kx*x+ky*y)*(-v[0]))*ds(14) \
     - ((cos(theta)*kx+sin(theta)*ky)*cos(kx*x+ky*y)*( v[1]))*ds(15) \
     - ((cos(theta)*kx+sin(theta)*ky)*cos(kx*x+ky*y)*( v[0]))*ds(16)
   solve(a==0, usigma, solver_parameters=linparams)
   animfile.write(usigma.sub(0), usigma.sub(1))
   print("run "+str(i))

quit()

# there follows an attempt to obtain the harmonics of a similar system
# does not currently work as I don't know how to include the weak bc (ds term gives errors when call assemble)
# nonsense eigenvalues / functions without BCs

from firedrake.petsc import PETSc
try:
	from slepc4py import SLEPc
except ImportError:
		import sys
		warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
		sys.exit(0)

u2 = TrialFunction(SV)

v2 = TestFunction(SV)

# attempt to get eigenvalues in D. Arnold SIAM textbook "Finite Element Exterior Calculus" (p.9)
a2 = ( inner(curl(u2),curl(v2))) *dx \
  - (1*(-v2[1]))*ds   # SETTING WEAK BC THIS WAY GIVES ERROR: for Arnold example I want E cross n = 0 on the bdy
m = inner(u2,v2)*dx \

petsc_a = assemble(a2).M.handle
petsc_m = assemble(m).M.handle

num_eigenvalues=6

opts = PETSc.Options()
opts.setValue("eps_gen_hermitian", None)
opts.setValue("st_pc_factor_shift_type", "NONZERO")
opts.setValue("eps_type", "krylovschur")
opts.setValue("eps_smallest_real", None)
opts.setValue("eps_tol", 1e-10)

es = SLEPc.EPS().create(comm=COMM_WORLD)
es.setDimensions(num_eigenvalues)
es.setOperators(petsc_a, petsc_m)
es.setFromOptions()

st=es.getST()
st.setType(SLEPc.ST.Type.SINVERT)
es.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

es.solve()

nconv = es.getConverged()
print("number of converged eigenvalues:")
print(nconv)
vr, vi = petsc_a.getVecs()
lam = es.getEigenpair(0, vr, vi)  #mode 0 spans cohomology space if is zero eigenvalue - choose mode here

import numpy as np
npa1 = np.array(vr.getSize())
npa1 = vr

cnt = SV.dof_dset.size
npa2 = npa1[0:cnt]

eigenmode=Function(SV)
eigenmode.vector()[:] = npa2

File("ed-wave-elliptic_eigenmode_u.pvd").write(eigenmode)

print("eigenvalue value:")
print(lam)

purified = Function(SV)
purified.interpolate(usigma.sub(0) - inner(usigma.sub(0), eigenmode)*eigenmode)
File("ed-wave-purified.pvd").write(purified)

for i in range (0, nconv):
   lam1 = es.getEigenpair(i, vr, vi)
   print("eigenvalue " +str(i)+" value:")
   print(lam1)
