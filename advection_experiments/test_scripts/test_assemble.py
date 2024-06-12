import firedrake as fd

# 1. Create a mesh and define a function space
mesh = fd.UnitSquareMesh(10, 10)  # Create a 10x10 unit square mesh
V = fd.FunctionSpace(mesh, "P", 1)  # P1 elements

# 2. Define the problem data
u = fd.TrialFunction(V)
v = fd.TestFunction(V)
f = fd.Function(V)  # Source term
f.dat.data[:] = 1.
#f.interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])"))  # Example source function

kappa = 1.0  # Diffusion coefficient (constant in this example)

# Define the bilinear and linear forms
a = kappa * fd.dot(fd.grad(u), fd.grad(v)) * fd.dx  # Bilinear form
L = f * v * fd.dx  # Linear form

# 3. Assemble the stiffness matrix (A) and right-hand side vector (b)
u_sol = fd.Function(V)  # Solution function
bc = fd.DirichletBC(V, 0.0, "on_boundary") 
A = fd.assemble(a, bcs=[bc])
b = fd.assemble(L, bcs=[bc])


# 5. Solve the linear system A * u_sol = b
fd.solve(A, u_sol, b)

# Output the solution (for example, save to a file or visualize)
#File("solution.pvd").write(u_sol)