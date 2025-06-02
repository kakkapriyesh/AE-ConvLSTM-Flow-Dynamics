# AE-ConvLSTM-Flow-Dynamics (Pytorch)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6994987.svg)](https://doi.org/10.5281/zenodo.6994987)

Document: Sequence to sequence AE-ConvLSTM network for modelling the dynamics of PDE systems] https://arxiv.org/abs/2208.07315.

Please cite the above paper if this is useful to you. 

This repository presents an attempt to learn solutions to the Navier–Stokes equations using physics constraints, without relying on data. The core model used is AE-ConvLSTM—an extended version of the autoencoder ConvLSTM network adapted from https://github.com/jhhuang96/ConvLSTM-PyTorch. The network has been modified to enable learning over long time horizons (~100+ time steps). Its architecture is illustrated below:

<figure>
  <p align="center">
  <img width="600" height="300"
  src="https://github.com/kakkapriyesh/AE-ConvLSTM_Flow_Dynamics/blob/main/AE-ConvLSTM.PNG">
  <figcaption>
    <p align="center">
    Red represents classical convolutional layer, blue represents ConvLSTM layer and yellow represents de-convolutional layer.
   </p>
    </figcaption>
   </p>
</figure>
The AE-ConvLSTM modules are unrolled in time, with hidden states passed between them as shown below:

<figure>
  <p align="center">
  <img width="600" height="300"
  src="https://github.com/kakkapriyesh/AE-ConvLSTM_Flow_Dynamics/blob/main/AE-ConvLSTM_Rollout.PNG">
  <figcaption>
    <p align="center">
    Hidden states being passed from one AE-ConvLSTM Module's encoder to another (Module is rolled out). r is the Module sequence number, and $l$ is number of prediction in a single Module
     </p>
    </figcaption>
   </p>
</figure>

The network was initially trained in a data-driven setting on canonical fluid dynamics problems such as the viscous Burgers equation, lid-driven cavity flow, flow past a cylinder, and the vorticity-dissipation formulation of Navier–Stokes equations. It demonstrated the ability to predict long time-series with few forward passes, mitigating vanishing gradient issues in sequence models.

Subsequently, the network was trained solely using the governing equations—i.e., with no access to data—through a physics-constrained learning approach. The model was able to capture the 2D viscous Burgers dynamics reasonably well, as illustrated below. Training was conducted for only 0.35 seconds of simulated time, and the rest is extrapolated:

<figure>
  <p align="center">
  <img width="250" height="550"
  src="https://github.com/kakkapriyesh/AE-ConvLSTM_Flow_Dynamics/blob/main/Burgers_PC.gif">
  <figcaption>
    <p align="center">
             First row is the target, second is prediction and the last is $L_{1}$ error (mostly at discontinuities).
      </p>
    </figcaption>
   </p>
</figure>

This result was achieved by incorporating finite difference discretizations of the PDEs directly into the loss function. Some of the finite difference and image plotting routines were adapted from https://github.com/cics-nd/ar-pde-cnn, as noted in the source files.

The physics_constraint.py file contains loss functions for the lid-driven cavity and the vorticity-dissipation formulation (periodic BCs for Navier–Stokes). These were early experiments in enforcing physics constraints on unsteady Navier–Stokes simulations. However, training did not converge. The underlying challenge lies in the rapidly shifting manifolds during time evolution for coupled fields (e.g., pressure and velocity), which evolve at different rates and magnitudes. This leads to poor optimization landscapes with numerous local minima when using gradient descent.

All experiments were conducted using 4 NVIDIA Tesla V100 GPUs.
