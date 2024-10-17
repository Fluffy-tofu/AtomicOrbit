# AtomicOrbit: Electron Orbital Visualization Tool

AtomicOrbit is a powerful Python tool for visualizing electron orbitals in atoms based on their quantum numbers. It provides both 3D interactive plots and a command-line interface for easy use.

![AtomicOrbit Logo](https://github.com/Fluffy-tofu/AtomicOrbit/blob/main/images/logo_atomicorbit_bigger.png)

## Features

- Generate 3D visualizations of electron orbitals
- Support for single orbital, multiple orbitals, and all orbitals plot types
- Interactive 3D plots using Plotly
- Command-line interface for easy use
- Support for atoms up to atomic number 60

## Installation

To install AtomicOrbit, follow these steps:

1. Ensure you have Python 3.7 or higher installed on your system.
2. Clone this repository:
   ```
   git clone https://github.com/your-username/AtomicOrbit.git
   cd AtomicOrbit
   ```
3. Install the package using pip:
   ```
   pip install .
   ```

## Usage

AtomicOrbit can be used via the command line. Here are some example commands:

1. Visualize a single orbital:
   ```
   atomicorbit -p single -n 2 -l 1 -m 0
   ```

2. Visualize all orbitals for carbon in one plot:
   ```
   atomicorbit -a carbon -p all
   ```

3. Visualize every orbital for an atom with 29 electrons (copper) seperatly:
   ```
   atomicorbit -e 29 -p multiple
   ```

For more options and information, use the help command:
```
atomicorbit -h
```

## Examples

Here are some example visualizations generated by AtomicOrbit:



Single orbital (2p):
![Single Orbital](https://github.com/Fluffy-tofu/AtomicOrbit/blob/main/images/Bildschirmfoto%202024-10-17%20um%2007.51.23.png)

2. All orbitals for Copper:
![All Orbitals](https://github.com/Fluffy-tofu/AtomicOrbit/blob/main/images/Bildschirmfoto%202024-10-17%20um%2007.55.20.png)

Every orbital in separate plots for Neodymium:
<img src="https://github.com/Fluffy-tofu/AtomicOrbit/blob/main/images/Bildschirmfoto%202024-10-17%20um%2008.22.47.png" alt="Multiple Orbitals" height="400">

## Contributing

Contributions to AtomicOrbit are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors who have helped with the development of AtomicOrbit.
- Special thanks to the scientific community for their research on electron orbitals and quantum mechanics.

## Contact

If you have any questions or feedback, please open an issue on the GitHub repository.

Happy orbital visualization!
