# AE-ConvLSTM_Flow_Dynamics

This repo is the work done in an attempt to learn Navier-Stokes problem using physics constraint without data. The network used was AE-ConvLSTM which is an extended version of the auto-encode convlstm network by https://github.com/jhhuang96/ConvLSTM-PyTorch. The modified network enables the learning of long chain of time steps (~100 +). The network structure looks something like this:

<figure>
  <p align="center">
  <img width="600" height="300"
  src="https://github.com/kakkapriyesh/AE-ConvLSTM_Flow_Dynamics/blob/main/AE-ConvLSTM.PNG">
  <figcaption>
    Red represents classical convolutional layer, blue represents ConvLSTM layer and yellow represents de-convolutional layer.
    </figcaption>
   </p>
</figure>

<figure>
  <p align="center">
  <img width="600" height="300"
  src="https://github.com/kakkapriyesh/AE-ConvLSTM_Flow_Dynamics/blob/main/AE-ConvLSTM_Rollout.PNG">
  <figcaption>
    Hidden states being passed from one AE-ConvLSTM Module's encoder to another (Module is rolled out). r is the Module sequence number, and $l$ is number of prediction in a single Module
    </figcaption>
   </p>
</figure>



The document explaining ConvLSTM and the training of the network along with results (Please cite !) : (yet to be posted).

The network is tested on data-driven cases: That is, training the network using data and the testing it on unseen data. Various attempts were viscous burgers equation, lid-driven cavity, flow past cylinder and vorticity dissipation formulation of N-S. The network can predict long time-series in few network passes hence helping the problem of vanishing gradients. Later the network was trained using governing equations only and no data (physics constraint method). The network captured 2-D viscous burgers fairly well as show in this gif:

