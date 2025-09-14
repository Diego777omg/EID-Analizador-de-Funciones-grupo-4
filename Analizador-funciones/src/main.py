import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math
import threading

import sympy as sp
from sympy import S
from sympy.calculus.util import function_range

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

x = sp.symbols('x')

def safe_sympify(expr_str):
    """Intenta convertir un string a expresión sympy, con manejo de errores"""
    try:
        expr = sp.sympify(expr_str, evaluate=True)
        return expr, None
    except Exception as e:
        return None, str(e)

def find_denominator_singularities(expr):
    """Devuelve las soluciones reales que hacen el denominador 0"""
    try:
        num, den = sp.fraction(sp.simplify(expr))
        if den == 1:
            return []
        sols = sp.solve(sp.Eq(den, 0), x)
        real_sols = [s for s in sols if sp.im(s) == 0]
        return list(map(sp.simplify, real_sols))
    except Exception:
        return []

def find_log_constraints(expr):
    """Busca argumentos de log y devuelve las restricciones (arg>0)"""
    constraints = []
    for f in expr.atoms(sp.log):
        args = f.args
        if args:
            constraints.append(sp.StrictGreaterThan(args[0], 0))
    return constraints

def find_even_root_problems(expr):
    """Detecta raíces de índice par con radicando simbólico y devuelve expresiones que deben >=0"""
    constraints = []
    for p in expr.atoms(sp.Pow):
        base, exp = p.as_base_exp()
        if exp.is_Rational:
            den = exp.q
            if den % 2 == 0: 
                constraints.append(sp.StrictGreaterThan(base, -sp.oo))  
    return constraints

def compute_domain(expr):
    """Calcula un dominio aproximado: Reales menos singularidades detectadas y aplica restricciones simples"""
    domain_exceptions = set()
    den_sing = find_denominator_singularities(expr)
    for s in den_sing:
        domain_exceptions.add(sp.simplify(s))
    logs = find_log_constraints(expr)
    roots = find_even_root_problems(expr)
    desc = "Dominio: todos los reales"
    if domain_exceptions or logs or roots:
        parts = []
        if domain_exceptions:
            parts.append("excepto x = " + ", ".join([str(sp.N(v)) for v in domain_exceptions]))
        if logs:
            parts.append("y además los argumentos de log deben ser > 0")
        if roots:
            parts.append("y cuidar raíces de índice par (radicandos >= 0)")
        desc = "Dominio: todos los reales, " + "; ".join(parts)
    return desc, list(domain_exceptions), logs, roots

def compute_intersections(expr):
    """Calcula intersecciones con ejes: f(0) y soluciones de f(x)=0 (reales)"""
    try:
        y_at_0 = sp.simplify(expr.subs(x, 0))
        sol = sp.solve(sp.Eq(expr, 0), x)
        real_roots = [s for s in sol if sp.im(s) == 0]
        return y_at_0, real_roots
    except Exception as e:
        return None, []

def compute_range(expr):
    """Intenta calcular el recorrido simbólicamente; si falla, hace muestreo numérico"""
    try:
        rng = function_range(expr, x, S.Reals)
        return str(rng)
    except Exception:
        xs = []
        ys = []
        for val in list(range(-50, 51, 2)):
            try:
                fv = float(sp.N(expr.subs(x, val)))
                if math.isfinite(fv):
                    xs.append(val); ys.append(fv)
            except Exception:
                continue
        if ys:
            return f"aprox [{min(ys):.4g}, {max(ys):.4g}] (muestreo)"
        else:
            return "No se pudo estimar el recorrido"

def evaluate_point(expr, x_val):
    """Devuelve pasos y resultado de evaluar f(x_val)"""
    steps = []
    try:
        steps.append(f"Expresión original: {sp.srepr(expr)}")
        substituted = expr.subs(x, x_val)
        steps.append(f"Sustituyendo x = {x_val} -> {sp.simplify(substituted)}")
        numeric = float(sp.N(substituted))
        steps.append(f"Cálculo numérico: f({x_val}) = {numeric}")
        return steps, numeric, None
    except Exception as e:
        return ["Error en evaluación: " + str(e)], None, str(e)

def sample_function(expr, domain_exceptions, x_min=-10, x_max=10, n_points=800):
    """Genera listas de x,y evitando singularidades. n_points es paso aproxim"""
    xs = []
    ys = []
    step = (x_max - x_min) / max(1, n_points)
    cur = x_min
    while cur <= x_max:
        skip = False
        for s in domain_exceptions:
            try:
                sval = float(sp.N(s))
                if abs(cur - sval) < 1e-2:
                    skip = True
                    break
            except Exception:
                continue
        if not skip:
            try:
                yv = sp.N(expr.subs(x, cur))
                if yv.is_real:
                    yvf = float(yv)
                    if math.isfinite(yvf):
                        xs.append(cur)
                        ys.append(yvf)
            except Exception:
                pass
        cur += step
    return xs, ys

class AnalizadorGUI:
    def __init__(self, root):
        self.root = root
        root.title("Analizador de funciones - Proyecto")
        root.geometry("1100x700")

        frm = ttk.Frame(root, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(frm)
        top.pack(fill=tk.X, pady=6)
        ttk.Label(top, text="Función f(x):").grid(row=0, column=0, sticky=tk.W)
        self.func_entry = ttk.Entry(top, width=60)
        self.func_entry.grid(row=0, column=1, padx=6)
        ttk.Label(top, text="x (opcional):").grid(row=0, column=2, sticky=tk.W, padx=(10,0))
        self.x_entry = ttk.Entry(top, width=12)
        self.x_entry.grid(row=0, column=3, padx=6)

        btn_frame = ttk.Frame(frm)
        btn_frame.pack(fill=tk.X, pady=6)
        self.analyze_btn = ttk.Button(btn_frame, text="Analizar", command=self.on_analyze)
        self.analyze_btn.pack(side=tk.LEFT, padx=4)
        self.eval_btn = ttk.Button(btn_frame, text="Evaluar punto", command=self.on_evaluate)
        self.eval_btn.pack(side=tk.LEFT, padx=4)
        self.plot_btn = ttk.Button(btn_frame, text="Graficar", command=self.on_plot)
        self.plot_btn.pack(side=tk.LEFT, padx=4)
        self.clear_btn = ttk.Button(btn_frame, text="Limpiar", command=self.on_clear)
        self.clear_btn.pack(side=tk.LEFT, padx=4)

        self.text = scrolledtext.ScrolledText(frm, width=60, height=25)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,6))

        self.fig = Figure(figsize=(7,5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frm)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.parsed_expr = None
        self.domain_exceptions = []
        self.x_eval_point = None

    def log(self, *msgs):
        for m in msgs:
            self.text.insert(tk.END, str(m) + "\n")
        self.text.see(tk.END)

    def on_clear(self):
        self.func_entry.delete(0, tk.END)
        self.x_entry.delete(0, tk.END)
        self.text.delete('1.0', tk.END)
        self.ax.clear()
        self.canvas.draw()
        self.parsed_expr = None
        self.domain_exceptions = []

    def on_analyze(self):
        expr_str = self.func_entry.get().strip()
        if not expr_str:
            messagebox.showwarning("Entrada vacía", "Ingrese una función en la caja 'Función f(x)'.")
            return
        expr, err = safe_sympify(expr_str)
        self.text.delete('1.0', tk.END)
        if err:
            self.log("Error al parsear la función:", err)
            return
        self.parsed_expr = sp.simplify(expr)
        self.log("Función parseada:", str(self.parsed_expr))
        domain_desc, den_excs, logs, roots = compute_domain(self.parsed_expr)
        self.domain_exceptions = den_excs
        self.log("Dominio (descripción):", domain_desc)
        if den_excs:
            self.log("Singularidades detectadas:", ", ".join([str(sp.N(d)) for d in den_excs]))
        y0, x_roots = compute_intersections(self.parsed_expr)
        self.log(f"Intersección con eje Y: f(0) = {sp.simplify(y0)}")
        if x_roots:
            self.log("Intersecciones con eje X (reales):", ", ".join([str(sp.simplify(r)) for r in x_roots]))
        else:
            self.log("Intersecciones con eje X: no se detectaron raíces reales.")
        rng = compute_range(self.parsed_expr)
        self.log("Recorrido (simbol./aprox):", rng)
        self.log("Análisis completado.")

    def on_evaluate(self):
        if self.parsed_expr is None:
            self.on_analyze()
            if self.parsed_expr is None:
                return
        x_text = self.x_entry.get().strip()
        if not x_text:
            messagebox.showwarning("Entrada vacía", "Ingrese un valor numérico para x.")
            return
        try:
            x_val = float(x_text)
        except Exception:
            messagebox.showerror("Error", "Valor de x no es numérico.")
            return
        steps, numeric, err = evaluate_point(self.parsed_expr, x_val)
        for s in steps:
            self.log(s)
        if err:
            self.log("Error en evaluación:", err)
        else:
            self.log(f"Punto evaluado: ({x_val}, {numeric})")
            self.x_eval_point = (x_val, numeric)
            self.on_plot()

    def on_plot(self):
        if self.parsed_expr is None:
            self.on_analyze()
            if self.parsed_expr is None:
                return
        self.ax.clear()
        xs, ys = sample_function(self.parsed_expr, self.domain_exceptions, x_min=-10, x_max=10, n_points=800)
        if not xs:
            self.log("No hay puntos muestreables para graficar (posible dominio muy restringido).")
            return
        self.ax.plot(xs, ys, label=f"f(x) = {sp.pretty(self.parsed_expr)}")
        y0, x_roots = compute_intersections(self.parsed_expr)
        try:
            y0f = float(sp.N(y0))
            self.ax.scatter([0], [y0f], marker='o', s=50, label=f"f(0)={y0f}")
        except Exception:
            pass
        for r in x_roots:
            try:
                rv = float(sp.N(r))
                self.ax.scatter([rv], [0], marker='x', s=50, label=f"root {rv}")
            except Exception:
                continue
        for s in self.domain_exceptions:
            try:
                sval = float(sp.N(s))
                self.ax.axvline(sval, linestyle='--', label=f"x={sval}")
            except Exception:
                continue
        if getattr(self, 'x_eval_point', None):
            xv, yv = self.x_eval_point
            try:
                self.ax.scatter([xv], [yv], marker='o', s=80, color='red', label=f"Punto evaluado ({xv:.3g},{yv:.3g})")
            except Exception:
                pass
        self.ax.set_title("Gráfica de la función")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)
        self.ax.legend(loc='best', fontsize='small')
        self.canvas.draw()
        self.log("Gráfica generada.")

def main():
    root = tk.Tk()
    app = AnalizadorGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
