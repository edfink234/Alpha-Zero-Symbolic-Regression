#!/usr/bin/env python3
import glob
import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from pathlib import Path

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

def main():
    files = sorted(glob.glob("benchmark_*_individuals.txt"), key=natural_key)
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

    output_path = Path("benchmark_rules.html")

    with open(output_path, "w") as f:
        f.write(html)

    
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
