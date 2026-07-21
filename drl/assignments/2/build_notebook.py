"""Build a Jupyter notebook (.ipynb) from a `# %%` percent-format source file.

Splits `assignment_src.py` on cell markers:
  * "# %%"            -> code cell
  * "# %% [markdown]" -> markdown cell (subsequent lines are un-commented)
and writes an nbformat v4 notebook. Keeping the authored source as a plain
Python file avoids all JSON-escaping pitfalls and keeps the code editable.
"""
import sys
import nbformat as nbf

SRC = "assignment_src.py"
OUT = "Team165_Q_learning_DQN_DDQN.ipynb"


def strip_md(line: str) -> str:
    """Remove the leading '# ' (or '#') that comments a markdown line."""
    if line.startswith("# "):
        return line[2:]
    if line == "#":
        return ""
    return line


def build():
    with open(SRC, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    cells = []
    cur_type = None          # 'code' or 'markdown'
    buf = []

    def flush():
        """Emit the accumulated buffer as a notebook cell."""
        if cur_type is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            return
        if cur_type == "markdown":
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(text))

    for line in lines:
        stripped = line.strip()
        if stripped == "# %% [markdown]":
            flush(); cur_type, buf = "markdown", []
        elif stripped == "# %%":
            flush(); cur_type, buf = "code", []
        else:
            if cur_type == "markdown":
                buf.append(strip_md(line))
            elif cur_type == "code":
                buf.append(line)
            # lines before the first marker are ignored
    flush()

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (venv_drl)",
            "language": "python",
            "name": "venv_drl",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT} with {len(cells)} cells "
          f"({sum(c['cell_type']=='code' for c in cells)} code, "
          f"{sum(c['cell_type']=='markdown' for c in cells)} markdown).")


if __name__ == "__main__":
    build()
