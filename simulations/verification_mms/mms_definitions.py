import sympy as sp

# Symboles
x, y, mu, rho = sp.symbols('x y mu rho')

# Solution MMS 
u_expr = sp.sin(sp.pi*x)*sp.cos(sp.pi*y)
v_expr = -sp.cos(sp.pi*x)*sp.sin(sp.pi*y)
p_expr = x+y

# Sources
grad_u_expr = [sp.diff(u_expr, x),sp.diff(u_expr, y)]
grad_v_expr = [sp.diff(v_expr, x),sp.diff(v_expr, y)]
grad_p_expr = [sp.diff(p_expr, x),sp.diff(p_expr, y)]

f_x = sp.diff(p_expr, x) \
    - mu * (sp.diff(u_expr, x, x) + sp.diff(u_expr, y, y)) \
    + sp.diff(rho * u_expr**2, x) + sp.diff(rho * u_expr * v_expr, y)

f_y = sp.diff(p_expr, y) \
    - mu * (sp.diff(v_expr, x, x) + sp.diff(v_expr, y, y)) \
    + sp.diff(rho * u_expr * v_expr, x) + sp.diff(rho * v_expr**2, y)
# Conversion en fonctions Python
u_MMS = sp.lambdify((x, y), u_expr, 'numpy')
v_MMS = sp.lambdify((x, y), v_expr, 'numpy')
p_MMS = sp.lambdify((x, y), p_expr, 'numpy')
source_mms = sp.lambdify((x, y, mu, rho), (f_x, f_y), 'numpy')

grad_u_MMS = sp.lambdify((x, y), grad_u_expr, 'numpy')
grad_v_MMS = sp.lambdify((x, y), grad_v_expr, 'numpy')
grad_p_MMS = sp.lambdify((x, y), grad_p_expr, 'numpy')