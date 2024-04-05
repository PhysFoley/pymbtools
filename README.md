# pymbtools: Tools for simulating CG lipid membranes in ESPResSo MD

It is assumed the reader has already compiled espressomd and is familiar with its basic usage. If not, see
[installation](https://espressomd.github.io/doc4.2.0/installation.html) and a brief [tutorial](https://espressomd.github.io/tutorials4.2.0/lennard_jones/lennard_jones.html).

Detailed information about the Cooke lipid model can be found in the following papers:
* Original 3-bead model: [Cooke & Deserno JCP 2005](https://doi.org/10.1063/1.2135785)
* Flip-fixed 4-bead model: [Foley & Deserno JCTC 2020](https://doi.org/10.1021/acs.jctc.0c00862)

`Lipid.py` and `mbtools.py` comprise the "pymbtools" suite of tools. `Lipid.py` contains the definiton for the `Lipid` class and `mbtools.py` is a collection of functions for assembling lipid bilayers in various geometries, along with a few rudimentary analysis functions and a method for bonding together particles into large-scale rigid structures.

The template directory contains 5 example scripts that run short simulations demonstrating the functionality of pymbtools.

* `flat_3bead.py` : The simplest case. A flat, 3-bead (original) Cooke model membrane.

* `flat_flipfixed.py` : Still just a flat periodic membrane, but with the updated (4-bead) Cooke model which suppresses flip-flop. This can be use to simulate asymmetric membranes.

* `buckle.py` : 4-bead flipfixed Cooke membrane buckled along the x-direction of the simulation box.

* `stickytape.py` : This shows how to run a simulation of a membrane strip with open edges patched up with "stickytape" (see [Foley & Deserno JCP 2024](https://doi.org/10.1063/5.0189771)). Additionally, *this file includes code for making 4-bead Cooke lipids with varying aspect ratios using an angle for tapering in order to open up the field of curvature-asymmetric membranes*. This version constructs the stickytapes the hard way, manually placing each individual bond in a thought-out way to form a rigid cross-linked structure.

* `stickytape_autobond.py` : This simulation is very similar to the previous one, but makes use of mbtools' springBondStructure() to automatically bond together all stickytape beads within a particular cutoff distance of one another, making the task of creating a rigid structure much easier.
