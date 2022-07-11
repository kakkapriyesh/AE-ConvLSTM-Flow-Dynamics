'''This function is used to transfrom and inverse trasform a 64 x 64 matrix for the purpose of physics constraint training'''


from sklearn.preprocessing import MinMaxScaler

import numpy as np

def _MinMaxScale(data, size=[64,64]):
    '''
    Input 2D Matrix with size = size
    Scale the data as y = (x - min) / ( max - min)
    Output scaled 2D Matrix
    '''
    # reshape to 1D
    data = data.reshape((-1,1))
    # create scaler
    scaler = MinMaxScaler()
    # fit and transform in one step
    normalized = scaler.fit_transform(data)
    # inverse transform
    inverse = scaler.inverse_transform(normalized)

    return normalized.reshape(size), inverse.reshape(size)


class MinMaxScale_data():
    def __init__(self, data, size):
        self.size = size
        self.scaler = MinMaxScaler().fit(data)
       # print(self.scaler)

    def transform(self, data):
        # reshape to 1D
        data = data.reshape((-1,1))
        # fit and transform in one step
        normalized = self.scaler.transform(data)
        return normalized.reshape(self.size)
    
    def inv_transform(self, normalized):
        # reshape to 1D
        normalized = normalized.reshape((-1,1))
        # inverse transform
        inverse = self.scaler.inverse_transform(normalized)
        return inverse.reshape(self.size)

######################### Scaling definations ###########################################

#scale_data = MinMaxScale_data(TDv_sf_I .reshape((-1,1)), [64,64])


def scale(dataset_T,scale_data):
	
    # scale_data = MinMaxScale_data(dataset_T.reshape((-1,1)), [64,64])
    dataset_T_reshape= np.reshape(dataset_T,(dataset_T.shape[0],dataset_T.shape[2]*dataset_T.shape[1],dataset_T.shape[3],dataset_T.shape[4]))
    dataset_T_scaled = np.zeros((dataset_T_reshape.shape))	# [26, 20, 64,64]
    inverse = np.zeros(dataset_T_reshape.shape)
    test = 0
    for data_idx, input in enumerate(dataset_T_reshape):
        for t_idx in range((dataset_T_reshape).shape[1]):
            dataset_T_scaled[data_idx,t_idx] = scale_data.transform(input[t_idx])
	# 		inverse = scale_data.inv_transform(dataset_T_scaled[data_idx,t_idx])
	# 		test += average(input[t_idx]-inverse)

	# print(dataset_T_scaled.shape, test)
    dataset_T_scaled=np.reshape((dataset_T_scaled),dataset_T.shape)
    return dataset_T_scaled

def inv_scale(dataset_T,scale_data):
	
    #scale_data = MinMaxScale_data(dataset_T.reshape((-1,1)), [64,64])
    dataset_T_reshape= np.reshape(dataset_T,(dataset_T.shape[0],dataset_T.shape[2]*dataset_T.shape[1],dataset_T.shape[3],dataset_T.shape[4]))
    dataset_T_scaled = np.zeros((dataset_T_reshape.shape))	# [26, 20, 64,64]
    inverse = np.zeros(dataset_T_reshape.shape)
    test = 0
    for data_idx, input in enumerate(dataset_T_reshape):
        for t_idx in range((dataset_T_reshape).shape[1]):
            dataset_T_scaled[data_idx,t_idx] = scale_data.inv_transform(input[t_idx])
	# 		inverse = scale_data.inv_transform(dataset_T_scaled[data_idx,t_idx])
	# 		test += average(input[t_idx]-inverse)

	# print(dataset_T_scaled.shape, test)
    dataset_T_scaled=np.reshape((dataset_T_scaled),dataset_T.shape)
    return dataset_T_scaled


if __name__ == '__main__':

    y = np.array([(i+1)*0.1 for i in range(6)], dtype=float).reshape((3,-1))
    x=1-y
    # normalized, inverse = _MinMaxScale(x, [3,2])

    in_data = MinMaxScale_data(x.reshape((-1, 1)), [3,2])
    normalized = in_data.transform(x)

    #normalized[2,1] = 1.1
    inverse = in_data.inv_transform(normalized)
    x=1-x

    print(x)
    print(normalized)
    print(inverse)


