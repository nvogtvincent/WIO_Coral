# Sea the Connectivity
## Introduction
Many marine invertebrates undergo a planktonic larval phase. Larvae can drift for significant distances through ocean currents (sometimes assisted by swimming), permitting long-distance dispersal. As a result, whilst many organisms are distributed across spatially isolated habitats (such as hydrothermal vents and coral reefs), these habitats may be *connected* in a network through larval dispersal. Understanding these connections is extremely important for understanding population dynamics, resilience, and biogeography. These connections are commonly displayed as *connectivity matrices*, but these can be non-intuitive to interpret spatially.

Sea the Connectivity is a web application developed to visualise these ecological networks as graphs in a clear and intuitive manner. Individual populations are represented as nodes (circles), with the size representing the likelihood that a larva from that population successfully settles anywhere. Hovering over a population node reveals all of the outgoing connections from that node, with bold red lines representing the strongest connections (the highest likelihood of dispersal). Clicking on a node displays the population name, and other useful statistics associated with that population.

## Example dataset: coral connectivity in the southwest Indian Ocean (SECoW)
This dataset is based on the SECoW dispersal model (Simulating Ecosystem Connectivity with WINDS) built on OceanParcels, and high-resolution surface currents from the (south)West INDian ocean Simulation (WINDS). We simulated virtual spawning events during the northwest monsoon between c. 8000 2x2 km reef cells across the southwest Indian Ocean, using larval biological parameters for the broadcasting coral *P daedalea*. These simulations are described in further detail [in this preprint](https://doi.org/10.5194/egusphere-2023-778). 

## How to use
To use Sea the Connectivity, just download and extract the repository, and open `index.html` in your web browser.
