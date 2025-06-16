import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Diffusion_Green.Train import eval_stage1, train_stage2, eval_stage2



def main(model_config = None):
    modelConfig = {
        "stage1_epoch": 200,
        "stage2_epoch": 400,
        "batch_size": 8,  # 这里修改为1但是实际上在里面是会一次训练10个pca通道的
        "T": 15,
        "channel": 32,
        "channel_mult": [1, 2, 4, 8],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "img_size": 64,
        "grad_clip":1,
        "stage2_save_epoch": 50,
        "norm_scale": 1,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "save_weight_dir": "./Checkpoints_Green/"
        }
    if model_config is not None:
        modelConfig = model_config

    # 根据需要可以选择执行不同的训练阶段
    eval_stage1(modelConfig)
    #train_stage2(modelConfig)
    eval_stage2(modelConfig)



if __name__ == '__main__':
    main()
