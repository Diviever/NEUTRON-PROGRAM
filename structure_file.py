"""
structure_file.py — material library and structure files (no GUI code)

    load_library(folder)        materials/<Name>.txt  ->  {name: (potential µeV, ε complex)}
    sheet_names(path)           [(sheet name, number of layer rows), ...] of an .xlsx file
    read_layers(path, sheet)    [(material name, thickness nm), ...] from columns A and B of one sheet
    to_periodic(rows)           any layer sequence -> materials in table order + thickness of each in every period
    periodic_materials(...)     those rows -> (list of Material in user mode, number of periods N)
    read_thickness_file(...)    thicknesses of ONE material from a dropped file (for its card)

Every structure becomes periodic: N periods of the materials in table order, each material with its own
thickness in every period. Example (top first):  Ni 30 nm, Ti 4 nm, Ti 5 nm
    -> neighbouring equal rows merged: Ni 30, Ti 9   -> materials [Ni, Ti], thicknesses Ni [30], Ti [9], N = 1
A material that is missing from a period gets thickness 0 there (that layer is left out of the stack),
so any aperiodic sequence is reproduced exactly.

Library file format, one file per material, the file name is the material name (materials/Ni.txt):
    row 1:  neutron potential V in µeV                          e.g.  0.2410
    row 2:  X-ray permittivity ε as "real imaginary"             e.g.  0.999951363 9.9055e-07
    row 3…: free comments (source, density, X-ray energy, …)

Structure file (.xlsx): column A = material name, column B = thickness in nm.
Rows whose column B is not a number (headers such as "Column2" or "Thickness") and empty rows are skipped.
"""
from pathlib import Path

import numpy as np

from calculation import Material

LIBRARY_DIR = Path(__file__).resolve().parent / "materials"


def parse_number(text):
    """Number from a cell or a text ('8,90' and '8.90' both work); None if it is not a number."""
    try:
        return float(str(text).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def parse_complex(text):
    """ε from 'real imaginary' (two numbers) or from '0.9999+1e-6i' / '0.9999+1e-6j'."""
    parts = text.replace(",", ".").split()
    if len(parts) == 2:
        return complex(float(parts[0]), float(parts[1]))
    return complex(text.replace(" ", "").replace("i", "j").replace(",", "."))


def load_library(folder=LIBRARY_DIR):
    """{material name: (potential, epsilon)} from all .txt files of the folder ({} if it does not exist)."""
    library = {}
    for file in sorted(Path(folder).glob("*.txt")):
        rows = [line.strip() for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        library[file.stem] = (float(rows[0].replace(",", ".")), parse_complex(rows[1]))
    return library


def find_in_library(name, library):
    """Library entry for a material name, ignoring upper/lower case; None if missing."""
    for key, value in library.items():
        if key.lower() == str(name).strip().lower():
            return value
    return None


def read_layers(path, sheet):
    """[(material name, thickness in nm), ...] from columns A and B of one sheet, in file order."""
    import openpyxl                                      # needed only for structure files
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = []
    for row in workbook[sheet].iter_rows(values_only=True):
        if len(row) < 2 or row[0] is None:
            continue
        thickness = parse_number(row[1])
        if thickness is not None:
            rows.append((str(row[0]).strip(), thickness))
    workbook.close()
    return rows


def sheet_names(path):
    """[(sheet name, number of layer rows), ...] for every sheet of the file."""
    import openpyxl
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    names = workbook.sheetnames
    workbook.close()
    return [(name, len(read_layers(path, name))) for name in names]


def to_periodic(rows):
    """
    rows: [(name, thickness), ...] from the top down.
    Returns (names, arrays, merged):
        names   materials in order of first appearance (the table order)
        arrays  {name: [thickness in period 1, period 2, ...]}, top period first, 0 where the material is missing
        merged  how many neighbouring rows of the same material were added together
    A new period starts whenever a material does not come after the previous one in the table order.
    """
    layers, merged = [], 0
    for name, d in rows:                                    # 1. merge neighbouring rows of the same material
        if layers and layers[-1][0].lower() == name.lower():
            layers[-1] = (layers[-1][0], layers[-1][1] + d)
            merged += 1
        else:
            layers.append((name, d))
    names = []                                              # 2. table order = order of first appearance
    for name, _ in layers:
        if name.lower() not in [n.lower() for n in names]:
            names.append(name)
    position = {n.lower(): i for i, n in enumerate(names)}
    periods, last = [], len(names)                          # 3. fill the periods
    for name, d in layers:
        i = position[name.lower()]
        if i <= last:                                       # not after the previous material: new period
            periods.append([0.0] * len(names))
        periods[-1][i] = d
        last = i
    arrays = {n: [period[i] for period in periods] for i, n in enumerate(names)}
    return names, arrays, merged


def periodic_materials(rows, library, sigma, first_row_is_top=True):
    """
    Rows of a structure file -> (materials, N, merged): one Material per distinct name, in user mode with its
    thickness in every period (nm -> Å), potential and ε from the library, roughness sigma.
    Raises KeyError listing all names that have no library file.
    """
    missing = sorted({name for name, _ in rows if find_in_library(name, library) is None})
    if missing:
        raise KeyError(", ".join(missing))
    ordered = rows if first_row_is_top else list(reversed(rows))
    names, arrays, merged = to_periodic(ordered)
    materials = []
    for name in names:
        potential, epsilon = find_in_library(name, library)
        material = Material(name, potential, epsilon=epsilon, sigma=sigma)
        material.user = 10.0 * np.array(arrays[name])
        material.mode = "user"
        materials.append(material)
    return materials, len(arrays[names[0]]), merged


def read_numbers(path):
    """Numbers of column A of the first sheet (a plain list of thicknesses)."""
    import openpyxl
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    values = [parse_number(row[0]) for row in workbook.worksheets[0].iter_rows(values_only=True) if row]
    workbook.close()
    return [v for v in values if v is not None]


def read_thickness_file(path, name, first_row_is_top=True):
    """
    Thicknesses (nm, top period first) of ONE material, for its card:
      - a structure file (material | thickness): the periodic thickness array of `name` from the first sheet
        that has layer rows (the same conversion as for the whole structure);
      - otherwise a plain column A of numbers, one thickness per period.
    """
    for sheet, n_rows in sheet_names(path):
        if n_rows == 0:
            continue
        rows = read_layers(path, sheet)
        names, arrays, _ = to_periodic(rows if first_row_is_top else list(reversed(rows)))
        for key in arrays:
            if key.lower() == name.strip().lower():
                return arrays[key]
        raise KeyError(f"'{name}' is not in sheet '{sheet}' (it has {', '.join(names)})")
    values = read_numbers(path)
    if not values:
        raise ValueError("no thicknesses found in the file")
    return values if first_row_is_top else values[::-1]
