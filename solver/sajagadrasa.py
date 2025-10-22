

import math
import pandas as pd
try:
    import pulp
except Exception as e:
    print("PuLP not installed. Install via 'python -m pip install pulp' to run optimization.")
    pulp = None

# Data (sama seperti report)
produk = [
    {"Produk":"Kopi", "d_per_hari":60, "S":75000, "h":2500, "L":7},
    {"Produk":"Smoothies", "d_per_hari":40, "S":60000, "h":2000, "L":5},
    {"Produk":"Brownies", "d_per_hari":30, "S":50000, "h":1800, "L":4},
    {"Produk":"Donat", "d_per_hari":35, "S":45000, "h":1600, "L":3},
    {"Produk":"Tea", "d_per_hari":45, "S":40000, "h":1200, "L":2},
    {"Produk":"Toast", "d_per_hari":25, "S":30000, "h":1000, "L":2},
]

df = pd.DataFrame(produk)
df["D_per_tahun"] = df["d_per_hari"] * 365

# Analytic EOQ
df["Q_eoq_analytic"] = ((2 * df["D_per_tahun"] * df["S"]) / df["h"])**0.5
df["TC_analytic"] = (df["D_per_tahun"] / df["Q_eoq_analytic"]) * df["S"] + (df["Q_eoq_analytic"] / 2) * df["h"]
df["SS20pct"] = 0.2 * df["d_per_hari"] * df["L"]
df["ROP_SS20"] = df["d_per_hari"] * df["L"] + df["SS20pct"]

# Save analytic results
df_out = df[["Produk","D_per_tahun","S","h","Q_eoq_analytic","TC_analytic","SS20pct","ROP_SS20"]].copy()
df_out.columns = ["Produk","D(unit/year)","S(Rp)","h(Rp/unit/year)","Q_EOQ_Analytic","TC_Analytic(Rp/year)","SS_20pct(unit)","ROP_SS20(unit)"]
df_out.to_excel("results_from_python_analytic.xlsx", index=False)
print("Analytic results written to results_from_python_analytic.xlsx")

# ---------- PuLP discrete-choice MILP (memilih kandidat Q integer) ----------
def solve_pulp_discrete(warehouse_capacity=None, pct_range=0.5, n_steps=21):
    """
    Pendekatan diskret:
    - Untuk tiap produk buat kandidat Q integer di sekitar EOQ analitik (±pct_range)
    - Gunakan variabel biner y_{i,q} untuk memilih satu kandidat Q per produk
    - Objective linear: sum_i sum_q y_{i,q} * (D_i*S_i/q + q/2*h_i)
    - Optional capacity constraint: sum_i (Q_i/2) <= warehouse_capacity
    """
    if pulp is None:
        print("PuLP not available. Skipping optimization.")
        return None

    prob = pulp.LpProblem("EOQ_DiscreteChoice", pulp.LpMinimize)

    # 1) buat kandidat Q per produk
    candidate_Q = {}
    for i, row in df.iterrows():
        D = row["D_per_tahun"]
        S = row["S"]
        h = row["h"]
        q_star = ((2 * D * S) / h) ** 0.5  # analytic EOQ
        # rentang kandidat sekitar ±pct_range dari q_star
        low = max(1, int(math.floor(q_star * (1 - pct_range))))
        high = max(low + 1, int(math.ceil(q_star * (1 + pct_range))))
        # buat grid integer; jika rentang kecil, ambil semua integer; else sampling n_steps
        if high - low + 1 <= n_steps:
            candidates = list(range(low, high + 1))
        else:
            step = (high - low) / (n_steps - 1)
            vals = [int(round(low + k * step)) for k in range(n_steps)]
            candidates = sorted(set([max(1, v) for v in vals]))
        candidate_Q[row["Produk"]] = candidates

    # 2) variabel biner y_{produk,q}
    y = {}
    for produk, candidates in candidate_Q.items():
        for q in candidates:
            name = f"y_{produk.replace(' ','_')}_{q}"
            y[(produk, q)] = pulp.LpVariable(name, cat="Binary")

    # 3) constraint: pilih tepat 1 kandidat per produk
    for produk, candidates in candidate_Q.items():
        prob += pulp.lpSum([y[(produk, q)] for q in candidates]) == 1, f"ChooseOne_{produk.replace(' ','_')}"

    # 4) objective linear: sum over konstanta * y
    obj_terms = []
    for i, row in df.iterrows():
        produk = row["Produk"]
        D = row["D_per_tahun"]
        S = row["S"]
        h = row["h"]
        for q in candidate_Q[produk]:
            cost_const = (D * S) / q + (q / 2.0) * h
            obj_terms.append(cost_const * y[(produk, q)])
    prob += pulp.lpSum(obj_terms)

    # 5) optional capacity constraint: sum_i (Q_i/2) <= capacity
    if warehouse_capacity is not None:
        prob += pulp.lpSum([ (q / 2.0) * y[(produk, q)]
                             for produk, candidates in candidate_Q.items()
                             for q in candidates ]) <= warehouse_capacity, "StorageCapacity"

    # 6) solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 7) kumpulkan hasil
    results = []
    for produk, candidates in candidate_Q.items():
        chosen_q = None
        for q in candidates:
            val = pulp.value(y[(produk, q)])
            if val is not None and val > 0.5:
                chosen_q = int(q)
                break
        if chosen_q is None:
            # fallback: pilih kandidat dengan nilai y terbesar kalau tidak ada yg =1
            best = max(candidates, key=lambda q: pulp.value(y[(produk, q)]) if pulp.value(y[(produk, q)]) is not None else -1)
            chosen_q = int(best)
        row = df[df["Produk"] == produk].iloc[0]
        D = row["D_per_tahun"]; S = row["S"]; h = row["h"]
        TC = (D / chosen_q) * S + (chosen_q / 2.0) * h
        results.append({"Produk": produk, "Q_choice": chosen_q, "TC_choice": TC})
    resdf = pd.DataFrame(results)
    resdf.to_excel("results_from_pulp_discrete.xlsx", index=False)
    print("PuLP discrete optimization results written to results_from_pulp_discrete.xlsx")
    return resdf

# ---------- Pemanggilan contoh ----------
# Jika ingin tanpa capacity constraint:
solve_pulp_discrete(warehouse_capacity=None, pct_range=0.5, n_steps=21)

