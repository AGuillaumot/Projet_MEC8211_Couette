import sympy as sp

# Symboles
x, y, mu, rho = sp.symbols('x y mu rho')

# Solution MMS 
u_expr = x**2 * (1 - x)**2 * y * (1 - y)
v_expr = -x * (1 - x) * y**2 * (1 - y)**2
p_expr = x + y

# Sources
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
