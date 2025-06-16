
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class stage1_Dataset(Dataset):
    
    def __init__(self, npy_path, f_num, u_max, u_min, image_size, is_train):
        # Set class attributes
        self.npy_path = npy_path
        self.image_size = image_size
        self.u_max = u_max
        self.u_min = u_min
        self.f_num = f_num
        self.is_train = is_train
        self.dataset = self.load_data()

    def load_data(self):
        
        v_list = [1]
        
        # 根据a_list和npy_dir生成训练集和测试集的路径
        data = np.load(self.npy_path).astype(np.float32) # [1000, 20, 128, 128]

        f_set = data[:, 0:1, :, :] #t=[1,10] [900, 10, 128, 128]
        u_set = data[:, 1:2, :, :] #t=50 [900, 10, 128, 128]
        temp_set = np.ones((f_set.shape[0], 1, f_set.shape[2], f_set.shape[3])) #u [300, 1, 128, 128]
        a_set = np.concatenate([temp_set*v_list[0]], axis=0)

        if self.u_max == self.u_min == 0:

            u_min = np.min(u_set)
            u_max = np.max(u_set)
            self.u_max = u_max
            self.u_min = u_min

        # pred_u_min = np.min(pred_u_set)
        # pred_u_max = np.max(pred_u_set)    

        u_set = (u_set - self.u_min) / (self.u_max - self.u_min)
        # pred_u_set = (pred_u_set - pred_u_min) / (pred_u_max - pred_u_min)
        f_set = (f_set - self.u_min) / (self.u_max - self.u_min)


        # f_min = np.min(f_set)
        # f_max = np.max(f_set)

        # f_set = (f_set - f_min) / (f_max - f_min)
        # u_set = (u_set - f_min) / (f_max - f_min)

        if self.is_train:
            repeat_count = 1

            a_train = torch.tensor(a_set, dtype=torch.float32)  # u0
            f_train = torch.tensor(f_set, dtype=torch.float32)  # u100
            u_train = torch.tensor(u_set, dtype=torch.float32)  # a

            # 对x_train进行复制并将复制的结果进行堆叠
            a_train = a_train.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化
            f_train = f_train.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化
            u_train = u_train.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化

            return [a_train, f_train, u_train]
        else:
            
            repeat_count = 1

            a_test = torch.tensor(a_set, dtype=torch.float32)  # u0
            f_test = torch.tensor(f_set, dtype=torch.float32)  # u100
            u_test = torch.tensor(u_set, dtype=torch.float32)  # a

            # 对x_train进行复制并将复制的结果进行堆叠
            a_tests = a_test.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化
            f_tests = f_test.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化
            u_tests = u_test.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化

            return [a_tests, f_tests, u_tests]

    
    def __len__(self):
        # Return the number of samples (n)
        return len(self.dataset[0])
    
    def extract_a(self, a_num, f_num):

        return self.dataset[0]
        
    
    def __getitem__(self, index):
        # Get the sample using index  
        # 对于2维的直接输入a就行因为a本身就是二维的
        a, f, u = self.dataset[0][index], self.dataset[1][index], self.dataset[2][index]


        # Return the pair (x, y) along with the disturbed G
        # a:[1,241,241] u:[241,241]
        return a, f, u


class stage2_Dataset(Dataset):
    
    def __init__(self, npy_dir, image_size, is_train):
        # Set class attributes
        self.npy_dir = npy_dir
        self.image_size = image_size


        self.is_train = is_train
        self.dataset = self.load_data()

    # 第二阶段不再进行归一化，因为stage1的结果已经是完成了归一化的
    def load_data(self):

        # 根据a_list和npy_dir生成训练集和测试集的路径
        data = np.load(self.npy_dir)  #[2,200,64,64]

        u_set = data[0].reshape(-1,1,self.image_size, self.image_size) # [10000, 1, 128, 128]
        pred_u_plus_set = data[1].reshape(-1,1,self.image_size, self.image_size) # [10000, 1, 128, 128]
        # pred_u_gauss_noise_set = data[2]



        if self.is_train:
            repeat_count = 1

            u_train = torch.tensor(u_set)  # u0
            # pred_u_train = torch.tensor(pred_u_set)  # u100
            pred_u_plus_train = torch.tensor(pred_u_plus_set)  # a

            # 对x_train进行复制并将复制的结果进行堆叠
            u_train = u_train.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化
            # pred_u_train = pred_u_train.repeat(repeat_count, 1, 1)  # 假设第二个维度不需要变化
            pred_u_plus_train = pred_u_plus_train.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化

            return [u_train, pred_u_plus_train]
        
        else:
            
            repeat_count = 1

            u_test = torch.tensor(u_set)  # u0
            pred_u_plus_test = torch.tensor(pred_u_plus_set)  # a

            # 对x_test进行复制并将复制的结果进行堆叠
            u_test = u_test.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化
            pred_u_plus_test = pred_u_plus_test.repeat(repeat_count, 1, 1, 1)  # 假设第二个维度不需要变化


            return [u_test, pred_u_plus_test]

    
    def __len__(self):
        # Return the number of samples (n)
        return len(self.dataset[0])
    
    def __getitem__(self, index):
        # Get the sample using index  
        # 对于2维的直接输入a就行因为a本身就是二维的
        u, pred_u_plus = self.dataset[0][index], self.dataset[1][index]

        # Return the pair (x, y) along with the disturbed G
        # a:[1,241,241] u:[241,241]
        return u, pred_u_plus