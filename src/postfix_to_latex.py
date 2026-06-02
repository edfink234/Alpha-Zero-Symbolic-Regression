#!/usr/bin/env python3
import glob
import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import system

def is_operator(token):
    return is_binary_operator(token) or is_unary_operator(token)

def is_binary_operator(token):
    return token in {'+', '-', '*', '/', '^', 'MYCDOT'}

def is_unary_operator(token):
    return token in {
        "cos", "exp", "sqrt", "sin", "asin", "arcsin", "log", "tanh",
        "acos", "arccos", "~", "ln", "MYBRACKETSQRT", "tan",
        "MYCOS", "MYSIN", "MYTAN"
    }

def rpn_to_infix(rpn_expression):
    stack = []
    if isinstance(rpn_expression, str):
        rpn_expression = rpn_expression.split()

    for token in rpn_expression:
        if not is_operator(token):
            stack.append(token)

        elif is_unary_operator(token):
            operand = stack.pop()

            if token in {"ln"}:
                result = f"log({operand})"
            elif token in {"MYBRACKETSQRT"}:
                result = f"sqrt({operand})"
            elif token == "~":
                result = f"(-{operand})"
            elif token in {"MYCOS"}:
                result = f"cos({operand})"
            elif token in {"MYSIN"}:
                result = f"sin({operand})"
            elif token in {"MYTAN"}:
                result = f"tan({operand})"
            else:
                result = f"{token}({operand})"

            stack.append(result)

        else:
            right_operand = stack.pop()
            left_operand = stack.pop()

            op = token
            if op == "^":
                op = "**"
            elif op == "MYCDOT":
                op = "*"

            result = f"({left_operand} {op} {right_operand})"
            stack.append(result)

    return stack[-1]

def natural_key(filename):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", filename)]

def count_expressions_better_than(filename, best_mse):
    count = 0

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                _, mse = line.rsplit(",", 1)
                mse = float(mse.strip())
            except ValueError:
                continue

            if mse < best_mse:
                count += 1
            else:
                break

    return count

class SympyDagEvaluator:
    """
    Build a DAG from a SymPy Expr by hash-consing subexpressions,
    then evaluate bottom-up on NumPy arrays/scalars.

    Node format:
        (op, data, children)

    where:
        op       : string like "const", "symbol", "add", "mul", ...
        data     : constant value or symbol name or None
        children : tuple of child node ids
    """

    def __init__(self, expr):
        self.expr = expr
        self.nodes = []
        self.root = -1

        # SymPy expr -> node id
        self._intern_expr = {}

        # (op, data, children) -> node id
        self._intern_node = {}

        self.root = self._build(expr)

    def _intern(self, op, data, children):
        key = (op, data, children)
        if key in self._intern_node:
            return self._intern_node[key]
        idx = len(self.nodes)
        self.nodes.append((op, data, children))
        self._intern_node[key] = idx
        return idx

    def _build(self, expr):
        if expr in self._intern_expr:
            return self._intern_expr[expr]

        if expr.is_Symbol:
            node_id = self._intern("symbol", str(expr), ())

        elif expr.is_Integer:
            node_id = self._intern("const", float(int(expr)), ())

        elif expr.is_Rational:
            node_id = self._intern("const", float(expr), ())

        elif expr.is_Float:
            node_id = self._intern("const", float(expr), ())

        elif expr.is_Number:
            node_id = self._intern("const", float(expr.evalf()), ())

        else:
            args = tuple(self._build(a) for a in expr.args)

            if expr.func is sp.Add:
                node_id = self._intern("add", None, args)

            elif expr.func is sp.Mul:
                node_id = self._intern("mul", None, args)

            elif expr.func is sp.Pow:
                node_id = self._intern("pow", None, args)

            elif expr.func is sp.sin:
                node_id = self._intern("sin", None, args)

            elif expr.func is sp.cos:
                node_id = self._intern("cos", None, args)

            elif expr.func is sp.exp:
                node_id = self._intern("exp", None, args)

            elif expr.func is sp.log:
                node_id = self._intern("log", None, args)

            elif expr.func is sp.sqrt:
                node_id = self._intern("sqrt", None, args)

            elif expr.func is sp.tanh:
                node_id = self._intern("tanh", None, args)

            elif expr.func.__name__ == "sech":
                node_id = self._intern("sech", None, args)

            elif expr.func is sp.asin:
                node_id = self._intern("asin", None, args)

            elif expr.func is sp.acos:
                node_id = self._intern("acos", None, args)

            elif expr.func is sp.Abs:
                node_id = self._intern("abs", None, args)

            else:
                raise NotImplementedError(
                    "Unsupported SymPy node: func=%r expr=%r" % (expr.func, expr)
                )

        self._intern_expr[expr] = node_id
        return node_id

    def evaluate(self, env, shape=None, dtype=np.float64):
        """
        env maps symbol names to numpy arrays/scalars, e.g.
            {"r": r_vals, "theta": theta_vals}

        shape is only needed if expression is constant and you want an array result.
        """
        values = [None] * len(self.nodes)

        for i, node in enumerate(self.nodes):
            op, data, children = node

            if op == "symbol":
                if data not in env:
                    raise KeyError("Missing value for symbol '%s'" % data)
                values[i] = env[data]

            elif op == "const":
                c = np.array(data, dtype=dtype)
                if shape is None:
                    values[i] = c
                else:
                    values[i] = np.full(shape, c, dtype=dtype)

            else:
                ch = [values[j] for j in children]

                if op == "add":
                    out = ch[0]
                    for x in ch[1:]:
                        out = out + x
                    values[i] = out

                elif op == "mul":
                    out = ch[0]
                    for x in ch[1:]:
                        out = out * x
                    values[i] = out

                elif op == "pow":
                    base, expo = ch
                    values[i] = np.power(base, expo)

                elif op == "sin":
                    values[i] = np.sin(ch[0])

                elif op == "cos":
                    values[i] = np.cos(ch[0])

                elif op == "exp":
                    values[i] = np.exp(ch[0])

                elif op == "log":
                    values[i] = np.log(ch[0])

                elif op == "sqrt":
                    values[i] = np.sqrt(ch[0])

                elif op == "tanh":
                    values[i] = np.tanh(ch[0])

                elif op == "sech":
                    values[i] = sech_stable(ch[0])

                elif op == "asin":
                    values[i] = np.arcsin(ch[0])

                elif op == "acos":
                    values[i] = np.arccos(ch[0])

                elif op == "abs":
                    values[i] = np.abs(ch[0])

                else:
                    raise RuntimeError("Unknown op '%s'" % op)

        return values[self.root]


def canonical_node_signatures(dag, *, ignore_const_values=True, ignore_param_names=True):
    sigs = [None] * len(dag.nodes)

    def rec(i):
        if sigs[i] is not None:
            return sigs[i]

        op, data, children = dag.nodes[i]
        child_sigs = tuple(rec(j) for j in children)

        if op == "const" and ignore_const_values:
            data_use = "CONST"
        elif op == "symbol" and ignore_param_names and str(data).startswith("a"):
            data_use = "PARAM"
        else:
            data_use = data

        sigs[i] = (op, data_use, child_sigs)
        return sigs[i]

    for i in range(len(dag.nodes)):
        rec(i)

    return set(sigs)

def canonical_node_signature_map(dag, *, ignore_const_values=True, ignore_param_names=True):
    sigs = [None] * len(dag.nodes)

    def rec(i):
        if sigs[i] is not None:
            return sigs[i]

        op, data, children = dag.nodes[i]
        child_sigs = tuple(rec(j) for j in children)

        if op == "const" and ignore_const_values:
            data_use = "CONST"
        elif op == "symbol" and ignore_param_names and str(data).startswith("a"):
            data_use = "PARAM"
        else:
            data_use = data

        sigs[i] = (op, data_use, child_sigs)
        return sigs[i]

    for i in range(len(dag.nodes)):
        rec(i)

    return {sig: i for i, sig in enumerate(sigs)}, sigs

def dag_subexpr_size(sig):
    op, data, children = sig
    return 1 + sum(dag_subexpr_size(c) for c in children)

def dag_sig_to_sympy(sig):
    op, data, children = sig
    args = [dag_sig_to_sympy(c) for c in children]

    if op == "symbol":
        return sp.Symbol(str(data))
    if op == "const":
        return sp.Symbol("C") if data == "CONST" else sp.Float(data)

    if op == "add":
        return sp.Add(*args)
    if op == "mul":
        return sp.Mul(*args)
    if op == "pow":
        return sp.Pow(*args)
    if op == "sin":
        return sp.sin(args[0])
    if op == "cos":
        return sp.cos(args[0])
    if op == "exp":
        return sp.exp(args[0])
    if op == "log":
        return sp.log(args[0])
    if op == "sqrt":
        return sp.sqrt(args[0])
    if op == "tanh":
        return sp.tanh(args[0])
    if op == "sech":
        return sp.sech(args[0])
    if op == "asin":
        return sp.asin(args[0])
    if op == "acos":
        return sp.acos(args[0])
    if op == "abs":
        return sp.Abs(args[0])

    raise ValueError(f"Unknown op: {op}")


def dag_jaccard(expr1, expr2, **kwargs):
    A = canonical_node_signatures(SympyDagEvaluator(expr1), **kwargs)
    B = canonical_node_signatures(SympyDagEvaluator(expr2), **kwargs)
    return 1.0 if not A and not B else len(A & B) / len(A | B)


def main():
    files = sorted(glob.glob("../benchmark_results/benchmark_*_individuals.txt"), key=natural_key)
    best_est_rule_mses = [
        3.51, 3.54, 4.03,
        12.5, 14.9, 4.83,
        2.89e8, 2.93e8, 2.92e8,
        5.41, 7.77, 2.09,
        18.2, 20.7, 16.5,
        0.317, 0.389, 0.379,
        547.0, 491.0, 409.0,
        14.9, 9.05, 6.97,
        5.56e3, 6.01e3, 3.34e3,
        5.67e3, 5.86e3, 3.77e3,
    ]
    best_exprs = []
    labels = []

    if len(files) != len(best_est_rule_mses):
        raise ValueError(
            f"Expected {len(best_est_rule_mses)} benchmark files, found {len(files)}"
        )
        
    variable_names = {
        "delta_w_t_k_ada_delta", "gamma", "beta_2", "t", "epsilon",
        "m_t_k", "v_t_k", "eta", "theta", "value", "d_ij",
        "m_t_k_hat", "g_t_k", "velocity_k", "expt_weight_squared_k",
        "gradient_k", "d_ij_nest", "delta_w_t_k", "expt_grad_squared_k",
        "v_t_k_hat", "w_k", "beta_1"
    }

    latex_names = {
        "delta_w_t_k_ada_delta": r"\Delta w^{\text{A}\hspace{-.018cm}\text{dadelta}}_{j, m, t=\tau}",
        "delta_w_t_k": r"\Delta w_{j, m, t=\tau}",
        "m_t_k": r"\mu_{j,m,t=\tau}",
        "m_t_k_hat": r"\widehat{\mu}_{j,m,t=\tau}",
        "v_t_k": r"\nu_{j,m,t=\tau}",
        "v_t_k_hat": r"\widehat{\nu}_{j,m,t=\tau}",
        "g_t_k": r"g_{j, m, t=\tau}",
        "gradient_k": r"\sigma_{\hspace{-.05cm}g^{2}_{j,m}}",
        "velocity_k": r"v_{j, m, t=\tau}",
        "expt_weight_squared_k": r"E\left[\Delta w_{j, m}^2\right]_{t=\tau}",
        "expt_grad_squared_k": r"E\left[g_{j, m}^2\right]_{t=\tau}",
        "w_k": r"w_{j,m,t=\tau}",
        "d_ij": r"d_{j}",
        "d_ij_nest": r"d_{j}^{\mathrm{Nesterov}}",
        "beta_1": r"\beta_1",
        "beta_2": r"\beta_2",
        "gamma": r"\gamma",
        "epsilon": r"\epsilon",
        "eta": r"\eta",
        "theta": r"\theta",
        "t": r"t",
        "value": r"y_{j}",
    }

    local_dict = {
        name: sp.Symbol(name, latex_name=latex)
        for name, latex in latex_names.items()
    }

    local_dict.update({
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "arcsin": sp.asin,
        "acos": sp.acos,
        "arccos": sp.acos,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "tanh": sp.tanh,
    })

    html_sections = []
    
    total_est_mse_improved_cases = 0.0
    total_found_mse_improved_cases = 0.0
    num_improved_cases = 0
    appendix_equations = []
    for i, filename in enumerate(files, start=1):
        best_est_mse = best_est_rule_mses[i - 1]
        num_better = count_expressions_better_than(filename, best_est_mse)
        
        with open(filename, "r") as f:
            first_line = f.readline().strip()

        if not first_line:
            continue

        postfix_rule, mse = first_line.rsplit(",", 1)
        postfix_rule = postfix_rule.strip()
        mse = float(mse.strip())
        if mse < best_est_mse:
            num_improved_cases += 1
            total_est_mse_improved_cases += best_est_mse
            total_found_mse_improved_cases += mse
        infix = rpn_to_infix(postfix_rule)
        print(f"Benchmark {i}")
        print("================")
        print(f"infix = {infix}")
        print(f"MSE = {mse}")
        print(f"Best Est. Rule MSE = {best_est_mse}")
        print(f"# expressions better than Best Est. Rule = {num_better}")
        print("===============\n")

        expr = parse_expr(
            infix,
            local_dict=local_dict,
            transformations=standard_transformations,
            evaluate=False,
        )
        best_exprs.append(expr)
        labels.append(f"B{i}")

        symbol_names = {
            local_dict[name]: latex
            for name, latex in latex_names.items()
        }
        
        substitutions = {
            local_dict["eta"]: sp.Float(0.5),
            local_dict["theta"]: sp.Float(0.01),
            local_dict["gamma"]: sp.Float(0.9),
            local_dict["epsilon"]: sp.Float(1e-8),
            local_dict["beta_1"]: sp.Float(0.9),
            local_dict["beta_2"]: sp.Float(0.999),
        }

        expr = expr.subs(substitutions)

        expr = expr.evalf(4)
        latex_code = sp.latex(expr, symbol_names=symbol_names, full_prec=False)
        mse_code = f"{mse:.4f}"
        appendix_equations.append(f"""
\\subsection*{{Benchmark {i}}}
\\begin{{equation}}
\\label{{eq:benchmark-{i}}}
{latex_code}, \\quad \\text{{MSE = }} {mse_code}
\\end{{equation}}
""")
        if num_improved_cases > 0:
            accumulated_percent_improvement = (
                (total_est_mse_improved_cases - total_found_mse_improved_cases)
                / total_est_mse_improved_cases
            ) * 100.0
        else:
            accumulated_percent_improvement = 0.0


        html_sections.append(f"""
<h2 id="benchmark-{i}">Benchmark {i}</h2>

<div class="rule-row">
    <div class="equation">
    \\[
    {latex_code}
    \\]
    </div>

    <div class="mse">
        MSE: {mse_code}
    </div>
</div>
""")
    
    toc = "\n".join(
        f'<li><a href="#benchmark-{i}">Benchmark {i}</a></li>'
        for i in range(1, len(html_sections) + 1)
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Benchmark Weight Update Rules</title>

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    line-height: 1.6;
}}

h1 {{
    margin-bottom: 20px;
}}

h2 {{
    margin-top: 50px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 5px;
}}

.rule-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 30px;
    margin-top: 20px;
    margin-bottom: 20px;
}}

.equation {{
    flex: 1;
    font-size: 1.2em;
    overflow-x: auto;
}}

.mse {{
    min-width: 140px;
    text-align: right;
    font-weight: bold;
    white-space: nowrap;
}}

ul {{
    margin-bottom: 40px;
}}
</style>
</head>

<body>

<h1>Benchmark Weight-Update Rules</h1>

<h2>Table of Contents</h2>
<ul>
{toc}
</ul>

{''.join(html_sections)}

</body>
</html>
"""
    # ---- DAG-Jaccard heatmap across best found benchmark rules ----
    n = len(best_exprs)
    M = np.eye(n)

    for a in range(n):
        for b in range(a + 1, n):
            sim = dag_jaccard(
                best_exprs[a],
                best_exprs[b],
                ignore_const_values=True,
                ignore_param_names=False,
            )
            M[a, b] = sim
            M[b, a] = sim

    labels = [f"B{i}" for i in range(1, n + 1)]

    # Sort by medoid: rule with largest total similarity to all others
    offdiag_sums = M.sum(axis=1) - 1.0
    medoid_pos = int(np.argmax(offdiag_sums))
    order = np.argsort(-M[medoid_pos, :])

    M_sorted = M[np.ix_(order, order)]
    labels_sorted = [labels[i] for i in order]

    df_sorted = pd.DataFrame(M_sorted, index=labels_sorted, columns=labels_sorted)
    df_sorted.to_csv("../data_files/benchmark_rule_dag_jaccard_sorted.csv")

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(M_sorted, vmin=0, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels_sorted, rotation=90)
    ax.set_yticklabels(labels_sorted)

    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{M_sorted[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
            )

    ax.set_title(
        "DAG-Jaccard Similarity of Best Found Weight-Update Rules\n"
        f"sorted by medoid {labels[medoid_pos]}"
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    heatmap_path = "../images/benchmark_rule_dag_jaccard_sorted_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)

    print(f"medoid = {labels[medoid_pos]}")
    print(f"Saved sorted DAG-Jaccard matrix to: benchmark_rule_dag_jaccard_sorted.csv")
    print(f"Saved sorted DAG-Jaccard heatmap to: {heatmap_path}")
    system(f"open {heatmap_path}")

    output_path = Path("../benchmark_rules.html")

    with open(output_path, "w") as f:
        f.write(html)

    appendix_tex = r"""
\appendix
\section{Discovered Weight-Update Equations}
\label{app:discovered-update-equations}

This appendix lists the best discovered symbolic update equation for each benchmark experiment.
""" + "\n".join(appendix_equations)

    appendix_path = Path("../benchmark_rules_appendix.tex")

    with open(appendix_path, "w") as f:
        f.write(appendix_tex)

    print(f"Wrote LaTeX appendix to: {appendix_path.resolve()}")
        
    print("Summary")
    print("=======")
    print(f"# improved cases = {num_improved_cases}")
    print(f"Total Best Est. MSE over improved cases = {total_est_mse_improved_cases}")
    print(f"Total Best Found MSE over improved cases = {total_found_mse_improved_cases}")
    print(f"Accumulated percent improvement = {accumulated_percent_improvement:.2f}%")
    print("===============\n")
    
    print(f"Wrote HTML to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
