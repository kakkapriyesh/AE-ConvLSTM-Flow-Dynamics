# AE-ConvLSTM-Flow-Dynamics (Pytorch)
[![DOI](https://zenodo.org/badge/510851150.svg)](https://zenodo.org/badge/latestdoi/510851150)

Document: Sequence to sequence AE-ConvLSTM network for modelling the dynamics of PDE systems] https://arxiv.org/abs/2208.07315.

Please cite the above paper if this is useful to you. 

This repo is the work done in an attempt to learn the Navier-Stokes problem using physics constraints without data. The network used was AE-ConvLSTM, an extended version of the auto-encoder ConvLSTM network by https://github.com/jhhuang96/ConvLSTM-PyTorch. The modified network enables learning a long chain of time steps (~100 +). The network structure looks something like this:

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




The network is tested on data-driven cases: i.e., training the network using data and testing it on unseen data. Various attempts were viscous burgers equation, lid-driven cavity, flow past cylinder, and vorticity dissipation formulation of N-S. The network can predict long time-series in few neural networks passes hence helping the problem of vanishing gradients. Later the network was trained using governing equations only and no data (physics constraint method). The network captured 2-D viscous burgers reasonably well, as shown where training is done for 0.35 sec, and the rest is extrapolation. 

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

This was possible by using finite difference discretization of the equation in the loss function, and a few bits of code (finite difference and image plotting) was taken from https://github.com/cics-nd/ar-pde-cnn. I have commented on the source of input data files in the code. Further, the physics constraint file contains loss function for lid-driven cavity and vorticity-dissipation formulation (periodic BC with Navier-Stokes), which was my attempt to make physics constraint work for unsteady Navier-Stokes solution. However, I could not do it. Time evolution changes manifolds drastically for coupled fields having different magnitude and evolution(i.e., pressure and velocity in N-S equations) , leading to local minima when using gradient descent for training.   

4 Tesla V100s where used for training. 
