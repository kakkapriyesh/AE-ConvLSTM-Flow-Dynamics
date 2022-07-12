# AE-ConvLSTM_Flow_Dynamics

This repo is the work done in an attempt to learn Navier-Stokes problem using physics constraint without data. The network used was AE-ConvLSTM which is an extended version of the auto-encode convlstm network by https://github.com/jhhuang96/ConvLSTM-PyTorch. The modified network enables the learning of long chain of time steps (~100 +). The network structure looks something like this:

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



The document explaining ConvLSTM and the training of the network along with results (Please cite !) : (yet to be posted).

The network is tested on data-driven cases: That is, training the network using data and the testing it on unseen data. Various attempts were viscous burgers equation, lid-driven cavity, flow past cylinder and vorticity dissipation formulation of N-S. The network can predict long time-series in few network passes hence helping the problem of vanishing gradients. Later the network was trained using governing equations only and no data (physics constraint method). The network captured 2-D viscous burgers fairly well as shown where training is done for 0.35 sec and the rest is extrapolation. 

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

This was possible by using finite difference discretization of the equation in the loss function and few bits of code (finite difference and image plotting) was taken from https://github.com/cics-nd/ar-pde-cnn. I have commented the source of data files in the code. Further, the physics constraint file contains loss function for lid-driven cavity and vorticity-dissipation formulation (periodic BC with Navier-Stokes), which was my attempt to make physics constraint work for unsteady Navier-Stokes solution but I couldnt do it :(, as it seems that time evolution changes manifolds drastically leading to local minima when using gradient decent for training.   
