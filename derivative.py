import sympy as sp

x = sp.symbols('x')

f = (sp.exp(x))**2

dx_f = sp.diff(f, x)

print(dx_f)