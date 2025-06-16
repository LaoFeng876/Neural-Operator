
import os
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Diffusion_Green import ResShiftTraining, ResShiftSampler
from Diffusion_Green.Model import UNet
from Scheduler import GradualWarmupScheduler
from Diffusion_Green.utils import stage1_Dataset, stage2_Dataset


def create_gaussian_kernel(sigma):
    """创建高斯滤波器的卷积核"""
    size = int(2 * 3 * sigma + 1)
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-3*sigma)**2+(y-3*sigma)**2)/(2*sigma**2)), (size, size))
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.view(1, 1, size, size)
    return kernel

stage1_train_path = ""
stage1_test_path = ""

def load_latest_checkpoint(directory, net_name, device):
    
    files = os.listdir(directory)
    
    
    net_files = [f for f in files if net_name in f]
    
    if not net_files:
        raise FileNotFoundError(f"No checkpoint found for {net_name} in {directory}")
    
    
    latest_file = max(net_files, key=lambda f: int(f.split('_')[-2]))  
    print(latest_file)
    
    
    ckpt = torch.load(os.path.join(directory, latest_file), map_location=device)
    
    return ckpt


def eval_stage1(modelConfig: Dict):
    print("stage1 evaluation start")
    
    with torch.no_grad():
        image_size = modelConfig["img_size"]


        for root, dirs, files in os.walk('datasets'):
            for filename in files:
                if 'test' in filename:
                    testset_path = os.path.join(root, filename)

        for root, dirs, files in os.walk('datasets'):
            for filename in files:
                if 'train' in filename:
                    trainset_path = os.path.join(root, filename)

        trainset = stage1_Dataset(trainset_path, 1000, 0, 0, image_size,  is_train=True) 
        testset  = stage1_Dataset(testset_path, 500, trainset.u_max, trainset.u_min, image_size, is_train=False) 

        
        train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False, drop_last=True)
        test_loader  = DataLoader(dataset=testset, batch_size=1, shuffle=False, drop_last=True)


        u_set = []  
        u0_set = [] 
        

        with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
            for a, f, u in tqdmDataLoader:
                
                u_set.append(u.cpu().squeeze())
                u0_set.append(f.cpu().squeeze().numpy())


        
        u_set = torch.tensor(np.stack(u_set, axis=0)).cuda()  
        u0_set = torch.tensor(np.stack(u0_set, axis=0)).cuda() 
        u_u0_set = torch.stack([u_set, u0_set], dim=0)

        
        global stage1_train_path
        stage1_train_path = 'data/'+trainset_path.split("/")[-1].split(".npy")[0]+f"_u_pred_u_gauss"+'.npy'
        np.save(stage1_train_path.split(".npy")[0], u_u0_set.cpu().detach().numpy())

        print("stage1 validset done")


        u_set = []  
        u0_set = [] 

        with tqdm(test_loader, dynamic_ncols=True) as tqdmDataLoader:
            for a, f, u in tqdmDataLoader:
                
                u_set.append(u.cpu().squeeze())
                u0_set.append(f.cpu().squeeze().numpy())
                

        u_set = torch.tensor(np.stack(u_set, axis=0)).cuda()
        u0_set = torch.tensor(np.stack(u0_set, axis=0)).cuda()
        u_u0_set = torch.stack([u_set, u0_set], dim=0)

        global stage1_test_path
        stage1_test_path = 'data/'+testset_path.split("/")[-1].split(".npy")[0]+f'_u_u0'+".npy"
        np.save(stage1_test_path.split(".npy")[0], u_u0_set.cpu().detach().numpy())
        print("stage1 testset done")

def train_stage2(modelConfig: Dict):

    print("train_stage2 start")

    device = torch.device(modelConfig["device"])
    batch_size = modelConfig["batch_size"]
    image_size = modelConfig["img_size"]
    

    global stage1_train_path
    if not os.path.exists(stage1_train_path):
        for root, dirs, files in os.walk('data'):
            for filename in files:
                if 'train' in filename:
                    stage1_train_path = os.path.join(root, filename)

        
        if not os.path.exists(stage1_train_path):
            raise ValueError(f"No such file or directory: {stage1_train_path}")
    
    validset = stage2_Dataset(stage1_train_path, image_size, is_train=True)

    
    diff_model = UNet(T=modelConfig["T"], pca_d=1, ch=modelConfig["channel"], out_c=1024, ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
    net_model1 = UNet(T=1, pca_d=1, ch=modelConfig["channel"], out_c=1, ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
    net_model3 = UNet(T=1, pca_d=1, ch=modelConfig["channel"], out_c=1, ch_mult=modelConfig["channel_mult"],
                  num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
    
    
    
    optimizer = torch.optim.AdamW(
        list(net_model1.parameters())+list(diff_model.parameters())+list(net_model3.parameters()), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["stage2_epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=10, after_scheduler=cosineScheduler)
    
    trainer = ResShiftTraining(
        diff_model, net_model1, net_model3, power=0.1, etas_end=0.9999, kappa=0.05, min_noise_level=0.001, T=modelConfig["T"]).to(device)
    
    valid_loader = DataLoader(dataset=validset, batch_size=modelConfig["batch_size"], shuffle=True, drop_last=True)


    
    for e in range(modelConfig["stage2_epoch"]):
        epoch_loss = 0.0
        count = 0
        err_sum = 0
        with tqdm(valid_loader, dynamic_ncols=True) as tqdmDataLoader:
            for u, u0 in tqdmDataLoader:
                
                optimizer.zero_grad()

                u, u0 = u.cuda(),  u0.cuda()   

                loss_res, err = trainer(u0, u)
                loss = loss_res*1e-3
                loss.backward()
                epoch_loss+=loss.detach().cpu().numpy()
                count+=1
                err_sum+=err.detach().cpu().numpy()

              
                torch.nn.utils.clip_grad_norm_(
                    list(net_model1.parameters())+list(diff_model.parameters())+list(net_model3.parameters()), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": epoch_loss/count,
                    "err: ": err_sum/count,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()

        if e !=0 and (e+1)%modelConfig["stage2_save_epoch"] == 0:
            torch.save(diff_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'stage2_diff_model_ckpt_' + str(e) + "_.pt"))
            torch.save(net_model1.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'stage2_net_model1_ckpt_' + str(e) + "_.pt"))
            torch.save(net_model3.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'stage2_net_model3_ckpt_' + str(e) + "_.pt"))
            
    print("train_stage2 done")

def eval_stage2(modelConfig: Dict):
    
    print("eval_stage2 start")

    
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        image_size = modelConfig["img_size"]
        batch_size = modelConfig["batch_size"]

        
        diff_model = UNet(T=modelConfig["T"], pca_d=1, ch=modelConfig["channel"], out_c=1024, ch_mult=modelConfig["channel_mult"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        
        net_model1 = UNet(T=1, pca_d=1, ch=modelConfig["channel"], out_c=1, ch_mult=modelConfig["channel_mult"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        
        net_model3 = UNet(T=1, pca_d=1, ch=modelConfig["channel"], out_c=1, ch_mult=modelConfig["channel_mult"],
                      num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

        

        ckpt1 = load_latest_checkpoint("Checkpoints_Green", "diff_model", device)
        diff_model.load_state_dict(ckpt1)
        diff_model.eval()

        ckpt1 = load_latest_checkpoint("Checkpoints_Green", "net_model1", device)
        net_model1.load_state_dict(ckpt1)
        net_model1.eval()

        ckpt1 = load_latest_checkpoint("Checkpoints_Green", "net_model3", device)
        net_model3.load_state_dict(ckpt1)
        net_model3.eval()

        print("Diffusion model load weight done.")

        sample = "resShift"
        

        if sample == "resShift":
            sampler = ResShiftSampler(
                diff_model, net_model1, net_model3, power=0.1, etas_end=0.9999, kappa=0.05, min_noise_level=0.001, T=modelConfig["T"]).to(device)
    

        
        global stage1_test_path
        if not os.path.exists(stage1_test_path):

            for root, dirs, files in os.walk('data'):
                for filename in files:
                    if 'test' in filename:
                        stage1_test_path = os.path.join(root, filename)
            
            if not os.path.exists(stage1_test_path):
                raise ValueError(f"Path does not exist: {stage1_test_path}")
        
        testset = stage2_Dataset(stage1_test_path, image_size, is_train=False)
        test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False)


        total_avg_u_err = []
        
        idx = 0

        fx_num = 50
        with tqdm(test_loader, dynamic_ncols=True) as tqdmDataLoader:
            for u, u0 in tqdmDataLoader:
                
                
                u, u0 = u.cuda(), u0.cuda()  

                if sample == "resShift":
                    pred_u, err = sampler(u0, u) 

                
                u0_cpu = u0.cpu().squeeze()  
                u_cpu = u.cpu().squeeze()  
                pred_u_cpu = pred_u.cpu().squeeze()  

                indices = [1]

                fig, axes = plt.subplots(2, 2, figsize=(12, 6))

                for i, ind in enumerate(indices):
                    ax = axes[0, 0]  
                    im = ax.imshow(u_cpu.detach().numpy(), cmap='viridis')  
                    ax.set_title(f"u_cpu")
                    ax.axis('off')  
                    fig.colorbar(im, ax=ax)  

                    ax = axes[0, 1]  
                    im = ax.imshow(u0_cpu.detach().numpy(), cmap='viridis')  
                    ax.set_title(f"u0")
                    ax.axis('off')  
                    fig.colorbar(im, ax=ax)  

                    ax = axes[1, 0]  
                    im = ax.imshow(pred_u_cpu.detach().numpy(), cmap='viridis')  
                    ax.set_title(f"pred_u")
                    ax.axis('off')  
                    fig.colorbar(im, ax=ax)  

                    ax = axes[1, 1]  
                    im = ax.imshow(torch.abs(pred_u_cpu - u_cpu).detach().numpy(), cmap='viridis')  
                    ax.set_title(f"pred_u-u_{err:.4f}")
                    ax.axis('off')  
                    fig.colorbar(im, ax=ax)  

                
                plt.tight_layout()
                plt.savefig(f'tesetset_pic/{idx}.png')
                plt.close()

                idx += 1  
                
                total_avg_u_err.append(err.cpu().numpy())

                with open('testset_err.txt', 'a') as f:
                    f.write(f"{err.cpu().numpy()}"+ '\n')

                    
        print(np.mean(np.array(total_avg_u_err)))

    print("eval_stage2 done")

