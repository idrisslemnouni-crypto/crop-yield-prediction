"""Validate saved notebook structure and reject recorded execution errors."""

from pathlib import Path

import nbformat

for path in Path("notebooks").glob("*.ipynb"):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if cell.execution_count is None:
                raise ValueError(f"Unexecuted cell in {path}")
            if any(output.output_type == "error" for output in cell.outputs):
                raise ValueError(f"Recorded execution error in {path}")
    print(f"Validated executed notebook: {path}")
