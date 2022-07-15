import fenics as fn
from pathlib import Path
import random
import string
import os
import subprocess

import meshio


def save_msh_to_file(mesh: meshio.Mesh) -> str:
    """Convert a mesh in (py)gmsh format and save it as fenics mesh"""
    N = 10
    name = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))

    Path("temp").mkdir(exist_ok=True)
    mesh.write(f"temp/{name}.msh", file_format="gmsh22")
    subprocess.run(
        ["gmsh", "-2", "-format", "msh2", f"temp/{name}.msh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    subprocess.run(
        ["dolfin-convert", f"temp/{name}.msh", f"temp/{name}.xml"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    os.remove(f"temp/{name}.msh")
    return f"temp/{name}.xml"


def convert_msh_to_fenics(mesh: meshio.Mesh) -> fn.Mesh:
    """Convert a mesh in (py)gmsh format to a fenics mesh"""
    filename = save_msh_to_file(mesh)
    res = fn.Mesh(filename)
    os.remove(filename)

    return res
