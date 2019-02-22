# Graptor
The **Gr**az **Ap**plication for **To**mographic **R**econstruction (Graptor) is a software tool developed to allow for efficient, high quality tomographic reconstruction from Radon-transform data. It originates from a project concerning Scanning Transmission Electron Tomography of the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en) together with the  [Austrian Centre for Electron Microscopy and Nanoanalysis](https://www.felmi-zfe.at), though it is not limited to such applications. A special feature of this toolbox is the coupling in multi-channel tomography (originally developed for joint HAADF-EDX reconstruction), yielding superior reconstruction quality for data sets with complementing information. The reconstruction algorithm bases on an iterative procedure that minimizes the Tikhonov functional associated with the tomographic reconstruction problem and multi-channel total generalized variation regularization. The code features a powerful OpenCL/GPU implementation, resulting in high reconstruction speed, and a Graphical User Interface allowing for easy use.

## Highlights
* High quality tomography reconstructions.
* Easy to use graphical user interface.
* Fast reconstruction due to custom OpenCL/GPU-implementation.
* Preprocessing options to ensure data fits the framework.
 
## Requirements
The code is written for Python 2.7 though it also works in Python 3. No dedicated installation is needed for the program, simply download the code and get started. Be sure to have the following Python modules installed, most of which should be standard.

* tkinter
* [pyopencl](https://pypi.org/project/pyopencl/)
* [argparse](https://pypi.org/project/argparse/)
* [numpy](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [matplotlib](https://pypi.org/project/matplotlib/) (with tkagg backend)
* [mrcfile](https://pypi.org/project/mrcfile/)
* subprocess
* queue
* threading
* shlex
* shutil
* [h5py](https://pypi.org/project/h5py/)

Particularly, correctly installing and configuring PyOpenCL might take some time, as dependent on the used platform/GPU, suitable drivers must be installed.

## Getting started
To start the Graphical User interface, run `GUI.py` inside the Graptor folder (e.g. in a terminal via `python GUI.py` or similarly from an Python development environment). 
We refer to the manual for precise instructions on how to use the GUI. It is adviced to run the examples in the manual with the phantom test data in order to get a grasp of the relevant functions and options.

Additionally, the script `Reconstruction_coupled.py` is provided for using the reconstruction algorithm inside a terminal. You can find help via `python Reconstruction_coupled.py --help` concerning possible parameters as well as an example for the call.

## Known issues

* There appears to be an issue with the automatic splitting in case of insufficient GPU memory under Windows.

## Authors

* **Richard Huber** richard.huber@uni-graz.at
* **Martin Holler** martin.holler@uni-graz.at 
* **Kristian Bredies** kristian.bredies@uni-graz.at

All authors are affiliated with the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en).

## Publications

* R. Huber, G. Haberfehlner, M. Holler, G. Kothleitner and K. Bredies. Total Generalized Variation regularization for multi-modal electron tomography. To appear in *Nanoscale*, 2019.

## Acknowledgements

The development of this software was partially supported by the following projects:

* *Regularization Graphs for Variational Imaging*, funded by the Austrian Science Fund (FWF), grant P-29192,

* *Lifting-based regularization for dynamic image data*, funded by the Austrian Science Fund (FWF), grant J-4112,

* *International Research Training Group IGDK 1754 Optimization and Numerical Analysis for Partial Differential Equations with Nonsmooth
Structures*, funded by the German Research Council (DFG) and the Austrian Science Fund (FWF), grant W-1244.

We wish to express special thanks to Georg Haberfehlner and Gerald Kothleitner from the [Austrian Centre for Electron Microscopy and Nanoanalysis](https://www.felmi-zfe.at) who offered us much insight into application of electron tomography, and aside from many other intellectual contributions to this program, designed the synthetic data set in the `example` folder.

## License

This project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.
