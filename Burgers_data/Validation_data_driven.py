import os.path
from network.Net import ED, Encoder, Decoder
from network.Config import convlstm_encoder_params, convlstm_decoder_params #parameters imported
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
#from tqdm import tqdm #used for progress bar
import numpy as np
import argparse
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from torch import from_numpy, tensor
#from earlystopping import EarlyStopping
os.getcwd()
import time
import matplotlib
matplotlib.use('agg')

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size',
                    default=1,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=50, type=int, help='sum of epochs')
args = parser.parse_args()
import torch, gc
gc.collect()
torch.cuda.empty_cache()
TIMESTAMP = "short"
tsteps=9   #number of forward passes to be done through neural network
CKPT='checkpoint_122_0.000874.pth'
PL=True  # Plot Loss
########################################################################################
x_1 = np.linspace(0.1,1,91)
random_seed = 199
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)  #if more than 1 gpu card
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   #For reproducibility

save_dir = './Burgers_valid/' + TIMESTAMP
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
TD_I = np.load('data/dataset_Uinput.npy')  #1,19,5,2,64,64
TD_T = np.load('data/dataset_Uoutput.npy')

IS=0  # simulation to test
TD_I = torch.from_numpy(TD_I)
TD_T = torch.from_numpy(TD_T[IS])

from pylab import*

load_dir='./Burg_seq10/' + TIMESTAMP

import matplotlib.pyplot as plt
    
train = []
epochs = []
file_name = save_dir
if PL==True:
               
    f = open(os.path.join(load_dir)+'/avg_train_loss.txt','r')
    q=0
    for row in f:
        row = row.split(' ')
        train.append(float(row[0]))
        epochs.append(q)
        q=q+1
   
    fig, ax = plt.subplots()

    plt.plot(epochs,train, color = 'g', label = 'Train loss')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
   
    ax.set_ylabel('Train loss', fontsize = 12)
    ax.set_xlabel('Epochs', fontsize = 12)

    plt.title('Train loss', fontsize = 20)
    plt.legend()
    #plt.show()
    plt.savefig(file_name + "/train_loss.png", bbox_inches='tight')
    plt.close()
    
    valid = []
    epochs = []

    f = open(os.path.join(load_dir)+'/avg_train_loss.txt','r')

    q=0
    for row in f:
        row = row.split(' ')
        valid.append(float(row[0]))
        epochs.append(q)
        q=q+1
    #valid.reverse()
    plt.plot(epochs,valid, color = 'g', label = 'Valid loss')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.ylabel('Valid loss', fontsize = 12)
    plt.xlabel('Epochs', fontsize = 12)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.title('Valid loss', fontsize = 20)
    plt.legend()
    #plt.show()
    plt.savefig(file_name + "/valid_loss.png", bbox_inches='tight')
    plt.close()
encoder_params = convlstm_encoder_params
decoder_params = convlstm_decoder_params

encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
net = ED(encoder, decoder)

device = torch.device("cuda")


net = nn.DataParallel(net)
net.to(device)

print (os.path.join(load_dir)+'/valid_test_loss_log.txt')

print('==> loading existing model')
model_info = torch.load(os.path.join(load_dir,CKPT+'.tar'))
net.load_state_dict(model_info['state_dict'])
lossfunction = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net.parameters())
optimizer.load_state_dict(model_info['optimizer'])
cur_epoch = model_info['epoch'] + 1
valid_losses = []

hidden_state_prev=None
with torch.no_grad():
    net.eval()

    loss_log=[]
    TD_I=TD_I[IS]
    Z = np.zeros((TD_T.shape[0], args.frames_input, 1, 64, 64))
    Z = torch.from_numpy(Z).float().to(device)
    TD_T = TD_T.float().to(device)
    inputs_Test=TD_I[0,:,:,:]
    inputs_Test = torch.unsqueeze(inputs_Test, 0) #1,10,1,64,64
    inputs = inputs_Test.float().to(device)
    inputs = torch.unsqueeze(inputs, 2)
    TD_T = torch.unsqueeze(TD_T, 2)
    loss=0.0
    states=None

    

    
    time1=np.zeros((tsteps))
  
    t0=time.time()
    for j in range(0,tsteps):
                

                pred,states = net(inputs,states)
              
                
                t2=time.time()
                
               
                output = pred
                inputs = output.float().to(device)
                Z[j, :, :, :, :] = pred
          
                loss=loss+lossfunction(pred, torch.unsqueeze(TD_T[j],0))
                time1[j]=t2-t0
                loss_log.append(loss)
                

    
    t1=time.time()           
    print("loss summation is ",loss)   
    mpl.rcParams['font.family'] = ['times new roman']  # default is sans-serif
    rc('text', usetex=False)
    cmap = "plasma"
    

    fig, ax = plt.subplots()

    plt.plot(time1, color = 'r', label = 'Time for prediction')
   
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    ax.set_ylabel('Time', fontsize = 12)
    ax.set_xlabel('Steps', fontsize = 12)

    plt.title('Time for prediction', fontsize = 20)
    plt.legend()
   
    plt.savefig(file_name + "/time.png", bbox_inches='tight')
    plt.close()

    Z=Z.data.cpu()
    Z=Z.numpy()


    q = 0
  


    mpl.rcParams['font.family'] = ['times new roman']  # default is sans-serif
    rc('text', usetex=False)
    cmap_error = "viridis"
    cmap = "plasma"
    target=TD_T.cpu().numpy()
    prediction=Z
    error = np.abs(target- prediction)
    mse = np.mean((np.abs(target- prediction))**2)
  
    Q=0
    error_loss=np.zeros((Z.shape[0]*Z.shape[1]))
    if Q==0:

        for t in range(0,Z.shape[0]):
            for j in range(Z.shape[1]):

                fig, ax = plt.subplots(3, 1, figsize=(2,5))
             
                error_loss[Q]=np.mean((np.abs(target[t, j, 0]- prediction[t, j, 0]))**2)


                c_max = np.max(np.array([target[t, j, 0], prediction[t, j, 0]]))
                c_min = np.min(np.array([target[t, j, 0], prediction[t, j, 0]]))
                ax[0].imshow(target[t, j, 0], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto',
                                    extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)
                ax[1].imshow(prediction[t, j, 0], interpolation='nearest', cmap=cmap, origin='lower',
                                        aspect='auto', extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)

                ax[2].imshow(error[t, j, 0], interpolation='nearest', cmap=cmap_error, origin='lower',
                                        aspect='auto', extent=[0, 1, 0, 1])
                c_max_error = np.max(error[t, j, 0])
                c_min_error = np.min(error[t, j, 0])

                p0 = ax[0].get_position().get_points().flatten()
                p1 = ax[1].get_position().get_points().flatten()
                ax_cbar = fig.add_axes([p1[2] + 0.015, p1[1], 0.025, p0[3] - p1[1]])
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min, c_max, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]

                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical',
                                                 ticks=ticks)
                cbar.set_ticklabels(tickLabels)

                p3 = ax[2].get_position().get_points().flatten()
                ax_cbar1 = fig.add_axes([p3[2] + 0.015, p3[1], 0.025, p3[3] - p3[1]])
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min_error, c_max_error, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                cbar = mpl.colorbar.ColorbarBase(ax_cbar1, cmap=plt.get_cmap(cmap_error), orientation='vertical',
                                                 ticks=ticks)
                cbar.set_ticklabels(tickLabels)


                ax[2].set_xticks([0, 0.25, 0.5, 0.75, 1.0])



                ax[2].set_xlabel('x', fontsize=14)


                ax[0].set_title('Time(s)={:.04f}'.format(2*Q*0.005), fontsize=14)
                ax[1].set_xticklabels([])
                ax[0].set_xticklabels([])


                ax[-1].set_xlabel('x', fontsize=14)

     
               
                plt.savefig(file_name + "Burg_seq1_time_{:.04f}.png".format(2*Q*0.005), bbox_inches='tight')
                plt.close()
                Q=Q+1
                print("time step printing",2*Q*0.005)
                
               
    with open("valid_test_loss_log.txt","wt") as f:
                
                for i in loss_log:
                    print(i,"for the image",q, file=f)
                    q=q+1
                print("Sum of valid loss is ",loss, "time for prediction is", t1-t0,file=f)
    ## Making the text file for valid loss plot"
    import shutil
    target=os.path.join(load_dir)+'/valid_test_loss_log.txt'
    og_path=os.path.join(os.getcwd())+'/valid_test_loss_log.txt'
   
    

    shutil.copyfile(og_path, target)

    
    fig, ax = plt.subplots()

    plt.plot(x_1[:-1],error_loss, color = 'b', label = 'MSE loss at each time step')
    #plt.plot(loss_log, color = 'p', label = 'loss at each prediction')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2,-2))
    # ax.set_major_formatter(ScalarFormatter())
    #
    #ax.ticklabel_format(useMathText=True)
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%0.5f'))
    ax.set_ylabel('MSE loss', fontsize = 12)
    ax.set_xlabel('Time Steps (s)', fontsize = 12)
    # plt.xscale()
    plt.title('MSE loss at each time step', fontsize = 20)
    plt.legend()
    #plt.show()
    file_name = save_dir
    plt.savefig(file_name + "/MSE_loss_avg.png", bbox_inches='tight')
    plt.close()