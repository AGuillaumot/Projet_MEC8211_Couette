import sympy as sp

# Symboles
x, y, mu = sp.symbols('x y mu')

# Solution MMS 
u_expr = sp.sin(sp.pi*y)
v_expr = 0
p_expr = sp.cos(sp.pi*x)

# Sources
f_x = sp.diff(p_expr, x) - mu * (sp.diff(u_expr, x, x) + sp.diff(u_expr, y, y))
f_y = sp.diff(p_expr, y)  

# Conversion en fonctions Python
u_MMS = sp.lambdify((x, y), u_expr, 'numpy')
v_MMS = sp.lambdify((x, y), v_expr, 'numpy')
p_MMS = sp.lambdify((x, y), p_expr, 'numpy')
source_mms = sp.lambdify((x, y, mu), (f_x, f_y), 'numpy')
