
''' the target and prediction should have image size of 1,5,2,64,64'''
'''The code is used for genrating sequence of 5 images'''


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
import matplotlib
#from tensorboardX import SummaryWriter
matplotlib.use('agg')
import torch
import numpy as np
import matplotlib.pyplot as plt


def Image_Series(target,prediction,load_dir,Name):
    #mpl.rcParams['font.family'] = ['times new roman']  # default is sans-serif
    #rc('text', usetex=True)
    cmap_error = "viridis"
    cmap = "plasma"
    # label=target=Z_act.cpu().numpy()
    # prediction=Z
    # error = np.abs(target - prediction)
    # #print(np.max(error))
    Q=0
    
    #test2=torch.unsqueeze((target[0,0,:,:,:,:]), 0)
    # label=target=target.detach().cpu().numpy()
    # prediction=prediction.detach().cpu().numpy()
    prediction=prediction.cpu()
    prediction=(prediction).numpy()
   # 
    target=target.cpu().numpy()
    #target=np.expand_dims(target, 1)
    #prediction=np.expand_dims(prediction, 1)
   # target=np.expand_dims(target, 0)
    #prediction=np.expand_dims(prediction, 0)
    error = np.abs(target - prediction)
    label=target=target
    if Q==0:
        fig, ax = plt.subplots(3, 5, figsize=(label.shape[1] * 5, 10))
        fig.subplots_adjust(wspace=0.5)

        for t in range((label.shape[1])):
            for j in range(1):
                c_max = np.max(np.array([target[0,t, 0]]))
                c_min = np.min(np.array([ target[0,t, 0]]))
                c_max1 = np.max(np.array([prediction[0,t, 0]]))
                c_min1 = np.min(np.array([ prediction[0,t, 0]]))
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
                ax_cbar1 = fig.add_axes([p1[2] + 0.0075, p1[1], 0.005, 0.45*(p0[3] - p1[1])])
                ax_cbar = fig.add_axes([p0[2] + 0.0075, p0[1], 0.005, 0.45*(p0[3] - p1[1])]) #top right, top left, width, height
               # ax_cbar2 = fig.add_axes([p1[2] + 0.0075, p1[1], 0.005, 0.5*(p0[3] - p1[1])])
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min, c_max, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]

                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical',
                                                ticks=ticks)
                cbar.set_ticklabels(tickLabels)
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min1, c_max1, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                cbar1 = mpl.colorbar.ColorbarBase(ax_cbar1, cmap=plt.get_cmap(cmap), orientation='vertical',
                                                ticks=ticks)

                cbar1.set_ticklabels(tickLabels)

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
#         file_dir = load_dir
# # If director does not exist create it
#         if not os.path.exists(file_dir):
#                 os.makedirs(file_dir)
        file_name = load_dir + "/"+Name+ "_5seq".format(Q)

        #file_dir = '.'
        # If director does not exist create it
    # if not os.path.exists(file_dir):
    #     os.makedirs(file_dir)
        #file_name = file_dir + "/burger2D_pred_main{:d}_{:d}".format(epoch, i)
        plt.savefig(file_name + ".png", bbox_inches='tight')
#          plt.savefig(file_name + ".pdf", bbox_inches='tight')
        plt.close()