'''This files trains and validates the network. Each forward pass from the network genrates 10 outputs'''



from network.Net import ED, Encoder, Decoder
from network.Config import convlstm_encoder_params, convlstm_decoder_params #parameters imported
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm #used for progress bar
import numpy as np
import argparse
import os
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
matplotlib.use('agg')
TIMESTAMP = "short"
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
parser.add_argument('-epochs', default=500, type=int, help='epochs')
args = parser.parse_args()

random_seed = 199
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)  #if more than 1 gpu card
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   #For reproducibility

save_dir = './Burg_seq10/' + TIMESTAMP
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
### Change #####
train_seq=10
######################################
NS=200 #NO of steps
VS = 10 # number of simulations required for validation
# Data can be genrated from the github Repo: https://github.com/cics-nd/ar-pde-cnn/tree/master/2D-Burgers-SWAG/solver
# This is the test dataset, One has to genrate alot more data (4000 sim) for the published results from the repo above.
# The code works for less data too, have tested it on lid driven and flow past cyl
TD_I = np.load("./data/dataset_Uinput.npy")  #Input and output data 
TD_T = np.load("./data/dataset_Uoutput.npy") #200,9,10,64,64 #200 simulations,of 90 time steps for image of size 64 x 64

TD_I = torch.from_numpy(TD_I)
TD_T = torch.from_numpy(TD_T)
TD_I=torch.unsqueeze(TD_I,3)
TD_T=torch.unsqueeze(TD_T,3)
print("shapes of input data and target data",TD_I.shape,TD_T.shape)

class Dataset():
    def __init__(self,X,Y):

        self.len = X.shape[0]
        self.x_data = (X[:, :])
        self.y_data = (Y[:])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# dataset = Dataset(TD_I,TD_T)
import matplotlib.pyplot as plt

cur_epoch = 0
encoder_params = convlstm_encoder_params
decoder_params = convlstm_decoder_params  #Obtaining parameters of the net (already imported)





def train():
        '''
        main function to run the training
        '''
        encoder = Encoder(encoder_params[0], encoder_params[1]).cuda() #Conv_nets(0)#CLSTM(1) goes to encoder file
        decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
        net = ED(encoder, decoder)  #Net is created
        early_stopping = EarlyStopping(patience=20, verbose=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

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
        # mini_val_loss = np.inf
      #  print((TD_I[0].shape))
        for epoch in range(cur_epoch, args.epochs + 1):
            for iter in range(0,NS-VS):
                dataset = Dataset(TD_I[iter],TD_T[iter])
                trainLoader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)
                
                T = tqdm(trainLoader, leave=False, total=len(trainLoader))
              
              
                
                    ###################
                    # train the model #
                    ###################
                loss=0.0
                count =0
                hidden_state_prev=None
                for i, (inputVar,targetVar) in enumerate(T):
                 
                    inputs = inputVar.float().to(device)  # B,S,C,H,W
                    label = targetVar.float().to(device)  # B,S,C,H,W
                    
                    net.train()
                    if count==0:
                        pred, state = net(inputs, hidden_state_prev)  # B,S,C,H,W
                     
                    else:
                        pred, state = net(pred_ip, hidden_state_prev)
                    hidden_state_prev=state
                    pred_ip=pred

                
                    
                    loss = loss+ lossfunction(pred, label)
                    T.set_postfix({
                        'trainloss': '{:.8f}'.format(loss_aver),
                        'epoch': '{:02d}'.format(epoch),
                        'Seq': '{:02d}'.format(iter)
                    })
                   # print(i)
                    count=count +1
                    if count==train_seq or i==8:
                   
                        loss_aver = loss.item() / (args.batch_size*count)
                        train_losses.append(loss_aver)
                        loss.backward(retain_graph=False)
                        
                        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.1)
                        optimizer.step()
                        optimizer.zero_grad()
                        count=0
                        loss=0.0
                        hidden_state_prev=None
                        

                
               
              
                # for name, param in net.named_parameters():
                #      if param.requires_grad:
                #         print (name, param.data)
                #         break
                
            for iter in range(NS-VS,NS):  
                    dataset = Dataset(TD_I[iter],TD_T[iter])
                    validLoader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)
                
                
                    ######################
                    # validate the model #
                    ######################
                    state_decoder=None
                    loss=0.0
                    count=0
                    with torch.no_grad():
                        net.eval()
                        T = tqdm(validLoader, leave=False, total=len(validLoader))
                        for i, (inputVar, targetVar) in enumerate(T):
                            inputs = inputVar.float().to(device)
                            label = targetVar.float().to(device)
                           
                            if count==0:
                                pred, state_dec = net(inputs, state_decoder)  # B,S,C,H,W
                        
                            else:
                                pred, state_dec = net(pred_ip, state_decoder)
                            state_decoder=state_dec
                            pred_ip=pred

                            loss = loss+lossfunction(pred, label)
                       
                            count=count +1
                            if count==train_seq or i==8:
                          
                                
                                loss_aver = loss.item() / (args.batch_size*count)
                                valid_losses.append(loss_aver)
                                
                                
                                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.1)
                          
                                count=0
                                loss=0.0
                                state_decoder=None

                            
                            
                  
                            cmap_error = "viridis"
                            cmap = "plasma"
                            input = inputs.cpu().numpy()
                            target = label.cpu().numpy()
                            prediction = pred.cpu().numpy()
                            error = np.abs(prediction - target)
                            print("epoch",epoch,"i",i,"iter",iter)
                            if epoch % 15 ==0 and i%1==0 and iter==3322:
                                fig, ax = plt.subplots(3, label.shape[1], figsize=(label.shape[1] * 5, 10))
                                fig.subplots_adjust(wspace=0.5)

                                for t in range((label.shape[1])):
                                    for j in range(1):

                                        c_max = np.max(np.array([target[0, t, 0], prediction[0, t, 0]]))
                                        c_min = np.min(np.array([target[0, t, 0], prediction[0, t, 0]]))
                                        ax[3 * j, t].imshow(target[0, t, 0], interpolation='nearest', cmap=cmap, origin='lower',
                                                            aspect='auto',
                                                            extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)
                                        ax[3 * j + 1, t].imshow(prediction[0, t, 0], interpolation='nearest', cmap=cmap,
                                                                origin='lower',
                                                                aspect='auto', extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)

                                        ax[3 * j + 2, t].imshow(error[0, t, 0], interpolation='nearest', cmap=cmap_error,
                                                                origin='lower',
                                                                aspect='auto', extent=[0, 1, 0, 1])
                                        c_max_error = np.max(error[0, t, 0])
                                        c_min_error = np.min(error[0, t, 0])

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
                                file_dir = save_dir+"\Images__burg_seq10"
                                if not os.path.exists(file_dir):
                                        os.makedirs(file_dir)
                                file_name = file_dir + "\{:3d}_{:3d}_{:3d}.png".format(iter,epoch,i)

                              
                                plt.savefig(file_name + ".png", bbox_inches='tight')
                 
                                plt.close()
                                T.set_postfix({
                                'trainloss': '{:.8f}'.format(loss_aver),
                                'epoch': '{:02d}'.format(epoch),
                                'Seq': '{:02d}'.format(iter)
                                        })
                        

            torch.cuda.empty_cache()
            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            epoch_len = len(str(args.epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' + f'train_loss: {train_loss:.6f} ' + f'valid_loss: {valid_loss:.6f}')  #bar at the end
            
            infile= open("avg_train_loss.txt", 'a')
           
            print(train_loss,"  ",epoch, file=infile)
            infile= open("avg_valid_loss.txt", 'a')
            print(valid_loss,"  ",epoch, file=infile)

                    

            # with open("avg_valid_loss.txt", 'wt') as f:
            #     for i in avg_valid_losses:
            #         print(i,"",epoch, file=f)
            
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

            pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print("no of parameters", pytorch_total_params)

if __name__ == "__main__":
        train()
