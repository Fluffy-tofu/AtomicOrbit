"""
AtomicOrbit: Electron Orbital Visualization Tool

This program visualizes the orbitals of electrons in an atom based on their quantum numbers.
It provides both 3D interactive plots and command-line interface for easy use.

Author: Fluffy-Tofu
Version: 1.1.0
"""

import argparse
import sys
from src.atomicorbit.visualization.interactive_3d_plot import plot_orbital_3d, plot_multiple_orbitals, plot_multiple_orbitals_single_plot

# ASCII Logo
LOGO = r"""
 █████╗ ████████╗ ██████╗ ███╗   ███╗██╗ ██████╗ ██████╗ ██████╗ ██████╗ ██╗████████╗
██╔══██╗╚══██╔══╝██╔═══██╗████╗ ████║██║██╔════╝██╔═══██╗██╔══██╗██╔══██╗██║╚══██╔══╝
███████║   ██║   ██║   ██║██╔████╔██║██║██║     ██║   ██║██████╔╝██████╔╝██║   ██║   
██╔══██║   ██║   ██║   ██║██║╚██╔╝██║██║██║     ██║   ██║██╔══██╗██╔══██╗██║   ██║   
██║  ██║   ██║   ╚██████╔╝██║ ╚═╝ ██║██║╚██████╗╚██████╔╝██║  ██║██████╔╝██║   ██║   
╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝   ╚═╝   
                                                                                                              
"""

# Element to atomic number mapping
ELEMENTS = {
    "hydrogen": 1, "H": 1,
    "helium": 2, "He": 2,
    "lithium": 3, "Li": 3,
    "beryllium": 4, "Be": 4,
    "boron": 5, "B": 5,
    "carbon": 6, "C": 6,
    "nitrogen": 7, "N": 7,
    "oxygen": 8, "O": 8,
    "fluorine": 9, "F": 9,
    "neon": 10, "Ne": 10,
    "sodium": 11, "Na": 11,
    "magnesium": 12, "Mg": 12,
    "aluminum": 13, "Al": 13,
    "silicon": 14, "Si": 14,
    "phosphorus": 15, "P": 15,
    "sulfur": 16, "S": 16,
    "chlorine": 17, "Cl": 17,
    "argon": 18, "Ar": 18,
    "potassium": 19, "K": 19,
    "calcium": 20, "Ca": 20,
    "scandium": 21, "Sc": 21,
    "titanium": 22, "Ti": 22,
    "vanadium": 23, "V": 23,
    "chromium": 24, "Cr": 24,
    "manganese": 25, "Mn": 25,
    "iron": 26, "Fe": 26,
    "cobalt": 27, "Co": 27,
    "nickel": 28, "Ni": 28,
    "copper": 29, "Cu": 29,
    "zinc": 30, "Zn": 30,
    "gallium": 31, "Ga": 31,
    "germanium": 32, "Ge": 32,
    "arsenic": 33, "As": 33,
    "selenium": 34, "Se": 34,
    "bromine": 35, "Br": 35,
    "krypton": 36, "Kr": 36,
    "rubidium": 37, "Rb": 37,
    "strontium": 38, "Sr": 38,
    "yttrium": 39, "Y": 39,
    "zirconium": 40, "Zr": 40,
    "niobium": 41, "Nb": 41,
    "molybdenum": 42, "Mo": 42,
    "technetium": 43, "Tc": 43,
    "ruthenium": 44, "Ru": 44,
    "rhodium": 45, "Rh": 45,
    "palladium": 46, "Pd": 46,
    "silver": 47, "Ag": 47,
    "cadmium": 48, "Cd": 48,
    "indium": 49, "In": 49,
    "tin": 50, "Sn": 50,
    "antimony": 51, "Sb": 51,
    "tellurium": 52, "Te": 52,
    "iodine": 53, "I": 53,
    "xenon": 54, "Xe": 54,
    "cesium": 55, "Cs": 55,
    "barium": 56, "Ba": 56,
    "lanthanum": 57, "La": 57,
    "cerium": 58, "Ce": 58,
    "praseodymium": 59, "Pr": 59,
    "neodymium": 60, "Nd": 60
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="atomicorbit",
        description="Visualize electron orbitals based on quantum numbers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example usage:\n"
               "  atomicorbit -p single -n 2 -l 1 -m 0\n"
               "  atomicorbit -a carbon -p all\n"
               "  atomicorbit -a C -p multiple"
    )

    parser.add_argument('-p', '--plot', choices=['single', 'multiple', 'all'],
                        required=True, help="Type of plot to generate")

    # Only require atom/electrons for multiple/all plots
    atom_group = parser.add_mutually_exclusive_group()
    atom_group.add_argument('-e', '--electrons', type=int, help="Number of electrons (1-60)")
    atom_group.add_argument('-a', '--atom', type=str, help="Name or symbol of the atom (e.g., 'hydrogen' or 'H')")

    # Arguments for single plot
    parser.add_argument('-n', type=int, help="Principal quantum number (required for single plot)")
    parser.add_argument('-l', type=int, help="Azimuthal quantum number (required for single plot)")
    parser.add_argument('-m', type=int, help="Magnetic quantum number (required for single plot)")

    return parser.parse_args()


def validate_args(args):
    if args.plot == 'single':
        if args.n is None or args.l is None or args.m is None:
            raise ValueError("For single plot, only -n, -l, and -m are required.\n"
                             "Example: atomicorbit -p single -n 2 -l 1 -m 0")
        if args.electrons is not None or args.atom is not None:
            raise ValueError("Single plot type doesn't require -e/--electrons or -a/--atom flags.\n"
                             "Example: atomicorbit -p single -n 2 -l 1 -m 0")
    else:  # multiple or all plots
        if args.electrons is None and args.atom is None:
            raise ValueError(f"For {args.plot} plot, either -e/--electrons or -a/--atom must be provided.\n"
                             f"Example: atomicorbit -a carbon -p {args.plot}")
        if args.n is not None or args.l is not None or args.m is not None:
            raise ValueError(f"For {args.plot} plot, -n, -l, and -m should not be provided.\n"
                             f"Example: atomicorbit -a carbon -p {args.plot}")


def get_electron_count(args):
    if args.electrons is not None:
        if args.electrons < 1 or args.electrons > 60:
            raise ValueError("Invalid number of electrons. Please enter a number between 1 and 60.")
        return args.electrons
    elif args.atom:
        atom_input = args.atom.lower()
        for name, number in ELEMENTS.items():
            if atom_input == name.lower() or atom_input == str(number).lower():
                return number
        raise ValueError(f"Unknown atom: {args.atom}. Please enter a valid atom name or symbol.")
    else:
        raise ValueError("Either --electrons or --atom must be provided.")


def main():
    print(LOGO)
    print("Welcome to AtomicOrbit: Electron Orbital Visualization Tool")
    print("Version 1.1.0\n")

    try:
        args = parse_arguments()
        validate_args(args)

        if args.plot == 'single':
            fig = plot_orbital_3d(n=args.n, l=args.l, m=args.m,
                                  title=f"Orbital: n={args.n}, l={args.l}, m={args.m}")
            output_file = f"orbital_n{args.n}_l{args.l}_m{args.m}.html"
        else:
            electron_count = get_electron_count(args)
            element_name = args.atom if args.atom else f"{electron_count} electrons"
            print(f"Visualizing orbitals for: {element_name.capitalize()}")

            if args.plot == 'multiple':
                fig = plot_multiple_orbitals(electron_count=electron_count)
                output_file = f"{element_name}_multiple_orbitals.html"
            else:  # args.plot == 'all'
                fig = plot_multiple_orbitals_single_plot(electron_count=electron_count)
                output_file = f"{element_name}_all_orbitals.html"

        fig.write_html(output_file)
        print(f"Plot generated successfully. Open '{output_file}' in your browser to view.")
        print("Opening the plot in your browser...")
        fig.show()

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nFor help, use: atomicorbit -h or atomicorbit --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
