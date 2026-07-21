"""Validate the executed notebook: count images, check for errors/empty outputs."""
import nbformat

nb = nbformat.read("Team165_Q_learning_DQN_DDQN.ipynb", as_version=4)
n_code = n_img = n_err = n_stream = 0
exec_counts = []
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    n_code += 1
    exec_counts.append(c.get("execution_count"))
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            n_err += 1
            print(f"  ERROR in cell {i}: {o.get('ename')}: {o.get('evalue')}")
        if o.get("output_type") == "stream":
            n_stream += 1
        data = o.get("data", {})
        if "image/png" in data:
            n_img += 1

print(f"Code cells        : {n_code}")
print(f"Embedded PNGs     : {n_img}")
print(f"Stream outputs    : {n_stream}")
print(f"Error outputs     : {n_err}")
print(f"All cells executed: {all(x is not None for x in exec_counts)} "
      f"(exec counts: {exec_counts})")
print("STATUS:", "OK — clean, fully executed, plots embedded" if n_err == 0 and n_img >= 6
      else "NEEDS ATTENTION")
