"""
structure_file.py — material library and structure files (no GUI code)

    load_library(folder)        materials/<Name>.txt  ->  {name: (potential µeV, ε complex)}
    sheet_names(path)           [(sheet name, number of layer rows), ...] of an .xlsx file
    read_layers(path, sheet)    [(material name, thickness nm), ...] from columns A and B of one sheet
    layers_to_materials(...)    those rows -> list of Material objects, one per layer, from the top down

Library file format, one file per material, the file name is the material name (materials/Ni.txt):
    row 1:  neutron potential V in µeV                          e.g.  0.2410
    row 2:  X-ray permittivity ε as "real imaginary"             e.g.  0.999951363 9.9055e-07
    row 3…: free comments (source, density, X-ray energy, …)

Structure file (.xlsx): column A = material name, column B = thickness in nm.
Rows whose column B is not a number (headers such as "Column2" or "Thickness") and empty rows are skipped.
"""
from pathlib import Path

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


def layers_to_materials(rows, library, sigma, first_row_is_top=True):
    """
    One Material per row, ordered from the top (ambient side) to the substrate, as build_stack expects
    with N = 1. Thickness nm -> Å; potential and ε from the library; every layer gets the roughness sigma.
    Raises KeyError listing all names that have no library file.
    """
    missing = sorted({name for name, _ in rows if find_in_library(name, library) is None})
    if missing:
        raise KeyError(", ".join(missing))
    ordered = rows if first_row_is_top else list(reversed(rows))
    materials = []
    for name, thickness_nm in ordered:
        potential, epsilon = find_in_library(name, library)
        d = 10.0 * thickness_nm
        materials.append(Material(name, potential, linear=(d, d), epsilon=epsilon, sigma=sigma))
    return materials
