from network.Net import ED, Encoder, Decoder
from network.Config import convlstm_encoder_params, convlstm_decoder_params #parameters imported
import torch
from utils.utils import mkdirs, toNumpy, toTuple
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm #used for progress bar
import numpy as np
import argparse
from network.burger2DFiniteDifference import Burger2DIntegrate
import os
from utils.burgerLoader2D import BurgerLoader
from torch import from_numpy, tensor
from network.earlystopping import EarlyStopping
os.getcwd()
import matplotlib
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.utils.data
import torch.profiler
matplotlib.use('agg')
from utils.Image_Series import Image_Series
from torch.profiler import profile, record_function, ProfilerActivity
import time
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=5,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=5,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
parser.add_argument('-ntrain', default=180, type=int, help='no of training sets')
parser.add_argument('-dt', default=0.005, type=float, help='time step')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TSTEP=20 #40 is full time steps, each timestep calculates 0.025 sec with 5 seq
TIMESTAMP = "short"

################## Loading intial data for training data######################
x0 = 0
x1 = 1.0
dx = (x1 - x0)/64
######################################
# usually the inital points required are large and can be genrated from the fenics solver
#fenics github Repo: https://github.com/cics-nd/ar-pde-cnn/tree/master/2D-Burgers-SWAG/solver

# this data is just for test and is found here: 
#https://drive.google.com/drive/folders/1LoZSpNsgnhna-hcqPvUZ_3S-R-MZkiHI?usp=sharing

Uinput_valid=np.load("data_PC/dataset_input_PC_Valid_5_dt05.npy") #200,39,5,2,64,64
Uinput=np.load("data_PC/dataset_input_PC.npy") #200,19,5,2,64,64

burgerLoader = BurgerLoader(dt=0.005)
training_loader = burgerLoader.createTrainingLoader(args.ntrain, 64, batch_size=args.batch_size)
burgerInt = Burger2DIntegrate(dx, nu=0.005, grad_kernels=[3, 3], device=device)  #gives gradients and laplacians in u and v



seq_length=args.frames_output

##############################################################################
random_seed = 199
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)  #if more than 1 gpu card
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   #For reproducibility

save_dir = '/burgers_PC/' + TIMESTAMP
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


class ValidDataset():
    def __init__(self,X,Y):

        self.len = X.shape[0]
        # self.len = 20
        self.x_data = (X)
        self.y_data = (Y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

import matplotlib.pyplot as plt

cur_epoch = 0
encoder_params = convlstm_encoder_params
decoder_params = convlstm_decoder_params  #Obtaining parameters of the net (already imported)
#for bars


def train():
        '''
        main function to run the training
        '''
        encoder = Encoder(encoder_params[0], encoder_params[1]).cuda() #Conv_nets(0)#CLSTM(1) goes to encoder file
        decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
        net = ED(encoder, decoder)  #Net is created
        early_stopping = EarlyStopping(patience=20, verbose=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("no of parameters", pytorch_total_params)
        

        if torch.cuda.device_count() > 0:
            net = nn.DataParallel(net)  #batchsize should be larger than no of GPU for this to work
        net.to(device)


        lossfunction = nn.MSELoss().cuda()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.5,
                                                        patience=2,
                                                        verbose=True)
        

        # to track the training loss as the model trains
        train_losses = []
        # # to track the validation loss as the model trains
        valid_losses = []
        # # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
        loss_aver=0
       
        tsteps=TSTEP
        tstep_current=np.zeros(TSTEP*10)
        countstep=0.0
        for i in range(0,TSTEP*10): # Progressively increase the time step to help stabailize training
            
            if i%10==0:
                countstep=countstep+1
             #   print(i)
            tstep_current[i]=countstep
 
        for epoch in range(cur_epoch, args.epochs + 1):
            
        
                #     ###################
                #     # train the model #
                #     ###################
            tsteps=int(tstep_current[epoch])
            t0 = time.time()
        
            for batch_idx, input in enumerate(training_loader):  #batch_idx is number of iterations
                
                if epoch%10==0:
                    optimizer = optim.Adam(net.parameters(), lr=args.lr)
             
                print("for epoch",epoch, "at index", batch_idx,"for tsteps",tsteps )
           
                input = input.float().to(device)   # B,C,H,W
                input=torch.unsqueeze(input,dim=1).float() # B,1,C,H,W
                dims = torch.ones(len(input.shape))  #4
                
                dims[1] = args.frames_input
                input = input.repeat(toTuple(toNumpy(dims).astype(int))) #B,10,2,64,64
                net.train()
                upred_list=input
                torch.autograd.set_detect_anomaly(True)
                loss=0.0
                
                state_dec=None 
            
                for iter_t in range(tsteps):
      
                    upred, state_dec= net(upred_list[:,-seq_length:], state_dec) 
 
                    upred_list=torch.cat((upred_list,upred),1)
                        
                    for i in range(0,seq_length):

                        ustar = burgerInt.crankNicolson(upred_list[:,(iter_t+1)*seq_length+i],upred_list[:,(iter_t+1)*seq_length+(i-1)], args.dt) #upred,upred0

                        
                        loss = loss+ lossfunction(upred_list[:,(iter_t+1)*seq_length+i], ustar) #channel to channel loss function calc?
                      
                loss_aver = loss.item() / (seq_length*tsteps)
                train_losses.append(loss_aver)
                loss.backward(retain_graph=False)
            
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.1)
                optimizer.step()
                optimizer.zero_grad()
            
                loss=0.0
                             
                state_dec=None
                t1= time.time()
                    
                
                    #####################
                    # validate the model #
                    #####################


            with torch.no_grad():
                    net.eval()
                    state_dec=None
                    loss=0.0
                    tsteps=int(tstep_current[epoch])
                    # for iter_valid in range(0,200):
                    
                    dataset=ValidDataset(Uinput[180:,0,0],Uinput_valid[180:]) #valid 200,19,10,2,64,64
                    validating_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,pin_memory=True)

                    for index, (inputVar, targetVar) in enumerate(validating_loader):
                      
                        input = inputVar.float().to(device) #5,2,64,64 
                        input=torch.unsqueeze(input,dim=1).float() #1,5,2,64,64
                        target = targetVar.float().to(device)  #1,9,10,2,64,64
                        dims = torch.ones(len(input.shape))  #4
                        dims[1] = args.frames_input
                        upred = input.repeat(toTuple(toNumpy(dims).astype(int))) #1,5,2,64,64

                        
                        loss_count=0
                       

                        for i in range(0,tsteps):
                            upred, state_dec = net(upred, state_dec) 

                            loss = loss+lossfunction(upred, target[:,i]) #1,5,2,64,64
                            loss_count=loss_count+1
                        
                        loss_aver = loss.item() / ((tsteps))
                        valid_losses.append(loss_aver)
                        loss= 0.0
                        state_dec=None 
                        t2=time.time()

                            
                        if (epoch+1) % 10 ==0 and index==0:
                                cmap_error = "viridis"
                                cmap = "plasma"
                           
                                target = label=target[:,i].cpu().numpy()
                                prediction = upred.cpu().numpy()
                                error = np.abs(prediction - target) #(1,5,2,64,64)
                                fig, ax = plt.subplots(3, label.shape[1], figsize=(label.shape[1] * 5, 10))
                                fig.subplots_adjust(wspace=0.5)

                                for t in range((label.shape[1])):
                                    for j in range(1):

                                        c_max = np.max(np.array([target[0,t, 0], prediction[0,t, 0]]))
                                        c_min = np.min(np.array([target[0,t, 0], prediction[0,t, 0]]))
                                        ax[3 * j, t].imshow(target[0,t, 0], interpolation='nearest', cmap=cmap, origin='lower',
                                                            aspect='auto',
                                                            extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)
                                        ax[3 * j + 1, t].imshow(prediction[0,t, 0], interpolation='nearest', cmap=cmap,
                                                                origin='lower',
                                                                aspect='auto', extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)

                                        ax[3 * j + 2, t].imshow(error[0,t, 0], interpolation='nearest', cmap=cmap_error,
                                                                origin='lower',
                                                                aspect='auto', extent=[0, 1, 0, 1])
                                        c_max_error = np.max(error[0,t, 0])
                                        c_min_error = np.min(error[0,t, 0])

                                        p0 = ax[3 * j, t].get_position().get_points().flatten()
                                        p1 = ax[3 * j + 1, t].get_position().get_points().flatten()
                                        ax_cbar = fig.add_axes([p1[2] + 0.0075, p1[1], 0.005, p0[3] - p1[1]])
                                        ticks = np.linspace(0, 1, 5)
                                        tickLabels = np.linspace(c_min, c_max, 5)
                                        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]

                                        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical',
                                                                        ticks=ticks)
                                        cbar.set_ticklabels(tickLabels)

                                        p0 = ax[3 * j + 2, t].get_position().get_points().flatten()
                                        ax_cbar = fig.add_axes([p0[2] + 0.0075, p0[1], 0.005, p0[3] - p0[1]])
                                        ticks = np.linspace(0, 1, 5)
                                        tickLabels = np.linspace(c_min_error, c_max_error, 5)
                                        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                                        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap_error),
                                                                        orientation='vertical',
                                                                        ticks=ticks)
                                        cbar.set_ticklabels(tickLabels)

                                        for ax0 in ax[:-1, t]:
                                            ax0.set_xticklabels([])

                                        for ax0 in ax[:, t]:
                                            ax0.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                                            ax0.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                                            if (t > 0):
                                                ax0.set_yticklabels([])
                                            else:
                                                ax0.set_ylabel('y', fontsize=14)

                                    ax[0, t].set_title('Target={:.02f}'.format(t), fontsize=14)
                                    ax[-1, t].set_xlabel('x', fontsize=14)
                                file_dir = save_dir+"/Images"
                    # If director does not exist create it
                                if not os.path.exists(file_dir):
                                        os.makedirs(file_dir)
                                file_name = file_dir + "\{:3d}_{:3d}".format(epoch,index)

                                plt.savefig(file_name + ".png", bbox_inches='tight')
                                plt.close()


            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            epoch_len = len(str(args.epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' + f'train_loss: {train_loss:.6f} ' + f'valid_loss: {valid_loss:.6f}')  #bar at the end
            
            infile= open("Train_"+TIMESTAMP+".txt", 'a')
        
            print(train_loss,"  ",epoch," time",t1-t0, file=infile)
            infile= open("Valid_"+TIMESTAMP+".txt", 'a')
            print(valid_loss,"  ",epoch," time",t2-t0, file=infile)

            
            
            
            pla_lr_scheduler.step(valid_loss)  # lr_scheduler
            print ("lr for next epoch is",  optimizer.param_groups[0]['lr'] )
            
            model_dict = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
 
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
                


if __name__ == "__main__":
        train()
