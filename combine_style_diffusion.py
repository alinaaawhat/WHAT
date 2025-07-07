import torch
from Diffusion_model.denoising_diffusion_pytorch import Trainer1D_Train as Trainer1D, Unet1D_cond_train as Unet1D_cond, GaussianDiffusion1Dcond_train as GaussianDiffusion1Dcond
import os
import sys
sys.path.append('./')
sys.path.append('./Style_conditioner/')
from data_load.get_domainhar import get_acthar,get_acthar_client
import argparse
import copy
import numpy as np
from datetime import datetime

# Style conditioner imports
from Style_conditioner.utils import _logger, set_requires_grad
from Style_conditioner.trainer.trainer import Trainer, Trainer_ft
from Style_conditioner.models.TC import TC
from Style_conditioner.models.model import base_Model
import torch.utils.data as data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def aggregate_models(client_models):
    """聚合客户端模型参数"""
    if not client_models:
        return None
    
    # 使用第一个模型作为基础
    global_model = copy.deepcopy(client_models[0])
    global_state_dict = global_model.state_dict()
    
    # 收集所有client的参数
    client_state_dicts = [model.state_dict() for model in client_models]
    
    # 对每个参数进行平均
    for key in global_state_dict.keys():
        param_sum = torch.zeros_like(global_state_dict[key])
        for client_state_dict in client_state_dicts:
            param_sum += client_state_dict[key]
        global_state_dict[key] = param_sum / len(client_models)
    
    # 更新全局模型
    global_model.load_state_dict(global_state_dict)
    return global_model

def train_style_conditioner_client(args, client_id, global_model=None):
    """为指定client训练Style Conditioner"""
    print(f"Training Style Conditioner for Client {client_id}")
    
    data_type = args.selected_dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed + client_id  # 为每个客户端使用不同的seed来模拟不同数据分布
    
    # Load data for specific client
    train_loader, valid_loader, target_loader, _ = get_acthar_client(args, data_type, target, batch_size=args.batch_size, remain_rate=remain_rate, seed=SEED, train_diff=0,client_id=client_id)
    train_dataset = train_loader.dataset
    source_loaders = data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)

    # Import config dynamically
    import importlib
    module_name = f'Style_conditioner.config_files.{data_type}_Configs'
    ConfigModule = importlib.import_module(module_name)
    configs = ConfigModule.Config()
    configs.batch_size = args.batch_size

    # Load Model
    model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)
    
    # 如果有全局模型，加载全局模型参数
    if global_model is not None:
        model.load_state_dict(global_model.state_dict())
        print(f"Client {client_id} loaded global model parameters")

    # Setup optimizers
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    
    # 简化训练 - 只进行几个epoch的本地训练
    print(f"Starting local training for client {client_id}")
    for epoch in range(args.style_local_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(source_loaders):
            data, target = data.to(device), target.to(device)
            
            model_optimizer.zero_grad()
            temporal_contr_optimizer.zero_grad()
            
            # 简化的训练步骤（你需要根据实际的训练逻辑调整）
            predictions, features = model(data)
            
            # 这里需要根据你的具体损失函数来调整
            loss = torch.nn.CrossEntropyLoss()(predictions, target)
            
            loss.backward()
            model_optimizer.step()
            temporal_contr_optimizer.step()
            
            if batch_idx >= args.style_local_steps:  # 限制训练步数
                break
        
        print(f"Client {client_id} Epoch {epoch+1} completed")
    
    return model

def train_diffusion_client(args, client_id, global_style_model, global_diffusion_model=None):
    """为指定client训练Diffusion Model"""
    print(f"Training Diffusion Model for Client {client_id}")
    
    data_type = args.selected_dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed + client_id  # 为每个客户端使用不同的seed
    
    testuser = {}
    testuser['seed'] = SEED
    testuser['name'] = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+f'_client{client_id}'

    # 保存全局style model供diffusion使用
    style_model_path = f'./Style_conditioner/conditioner_pth/global_style_model_client{client_id}.pt'
    torch.save({
        'model_state_dict': global_style_model.state_dict(),
    }, style_model_path)
    testuser['conditioner'] = style_model_path
    
    train_loader, valid_loader, target_loader, testuser['n_class'] = get_acthar(args, data_type, target, batch_size=64, remain_rate=remain_rate, seed=SEED)
    source_loaders = train_loader

    testuser['remain_data'] = remain_rate

    for minibatch in source_loaders:
        batch_size = minibatch[0].shape[0]
        shapex = [minibatch[0].shape[0], minibatch[0].shape[1], minibatch[0].shape[2]]
        break
    
    if shapex[1] % 64 != 0:
        shapex[1] = 64 - (shapex[1] % 64) + shapex[1]

    model_our = Unet1D_cond(    
        dim=64,
        num_classes=100,
        dim_mults=(1, 2, 4, 8),
        channels=shapex[2],
        context_using=True
    )

    diffusion = GaussianDiffusion1Dcond(
        model_our,
        seq_length=shapex[1],
        timesteps=100,  
        objective='pred_noise'
    )
    diffusion = diffusion.to(device)
    
    # 如果有全局diffusion模型，加载参数
    if global_diffusion_model is not None:
        diffusion.load_state_dict(global_diffusion_model.state_dict())
        print(f"Client {client_id} loaded global diffusion model parameters")

    train_loader = source_loaders
    trainer = Trainer1D(
        diffusion,
        dataloader=train_loader,
        train_batch_size=shapex[0],
        train_lr=2e-4,
        train_num_steps=args.diffusion_local_steps,  # 使用本地训练步数
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        results_folder=f'./Diffusion_model/dm_pth/client_{client_id}/'
    )

    trainer.train(testuser)
    return diffusion

def main():
    parser = argparse.ArgumentParser()

    ######################## Model parameters ########################
    home_dir = os.getcwd()

    parser.add_argument('--seed', default=1, type=int, help='seed value')
    parser.add_argument('--selected_dataset', default='dsads', type=str, help='Dataset of choice: pamap, uschad, dsads')
    parser.add_argument('--target', default=0, type=int, help='Choose task id')
    parser.add_argument('--remain_rate', default=0.2, help='Using training data ranging from 0.2 to 1.0')
    parser.add_argument('--results_folder', default='./Diffusion_model/dm_pth/', type=str, help='saving directory')
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--batch_size', default=64, type=int, help='Training batch')
    
    # Federated learning parameters
    parser.add_argument('--num_clients', default=3, type=int, help='number of clients')
    parser.add_argument('--style_rounds', default=3, type=int, help='Style Conditioner federated rounds')
    parser.add_argument('--diffusion_rounds', default=3, type=int, help='Diffusion Model federated rounds')
    parser.add_argument('--style_local_epochs', default=2, type=int, help='local epochs for style training')
    parser.add_argument('--style_local_steps', default=50, type=int, help='local steps per epoch for style training')
    parser.add_argument('--diffusion_local_steps', default=500, type=int, help='local training steps for diffusion')

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs('./Style_conditioner/conditioner_pth/', exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)
    for client_id in range(1, args.num_clients + 1):
        os.makedirs(f'./Diffusion_model/dm_pth/client_{client_id}/', exist_ok=True)

    print("开始联邦学习训练流程:")
    print(f"数据集: {args.selected_dataset}")
    print(f"目标: {args.target}")
    print(f"客户端数量: {args.num_clients}")
    print(f"Style Conditioner 联邦轮数: {args.style_rounds}")
    print(f"Diffusion Model 联邦轮数: {args.diffusion_rounds}")

    # ============ Phase 1: Federated Style Conditioner Training ============
    print("\n" + "="*60)
    print("Phase 1: 联邦训练 Style Conditioner")
    print("="*60)
    
    global_style_model = None
    
    for round_num in range(args.style_rounds):
        print(f"\nStyle Conditioner Round {round_num + 1}/{args.style_rounds}")
        
        # 各客户端训练Style Conditioner
        client_style_models = []
        for client_id in range(1, args.num_clients + 1):
            client_model = train_style_conditioner_client(args, client_id, global_style_model)
            client_style_models.append(client_model)
        
        # 聚合Style Conditioner模型
        global_style_model = aggregate_models(client_style_models)
        print(f"Style Conditioner Round {round_num + 1} aggregation completed")
    
    # 保存全局Style Conditioner模型
    style_save_path = './Style_conditioner/conditioner_pth/global_style_model.pt'
    torch.save({
        'model_state_dict': global_style_model.state_dict(),
        'round': args.style_rounds
    }, style_save_path)
    print(f"Global Style Conditioner saved to: {style_save_path}")

    # ============ Phase 2: Federated Diffusion Model Training ============
    print("\n" + "="*60)
    print("Phase 2: 联邦训练 Diffusion Model")
    print("="*60)
    
    global_diffusion_model = None
    
    for round_num in range(args.diffusion_rounds):
        print(f"\nDiffusion Model Round {round_num + 1}/{args.diffusion_rounds}")
        
        # 各客户端训练Diffusion Model
        client_diffusion_models = []
        for client_id in range(1, args.num_clients + 1):
            client_model = train_diffusion_client(args, client_id, global_style_model, global_diffusion_model)
            client_diffusion_models.append(client_model)
        
        # 聚合Diffusion模型
        global_diffusion_model = aggregate_models(client_diffusion_models)
        print(f"Diffusion Model Round {round_num + 1} aggregation completed")
    
    # 保存全局Diffusion模型
    data_type = args.selected_dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed
    
    diffusion_save_path = f'./Diffusion_model/dm_pth/global_diffusion_model_{data_type}_tar_{target}_rm_{remain_rate}_seed_{SEED}.pt'
    torch.save({
        'model': global_diffusion_model.state_dict(),
        'round': args.diffusion_rounds,
        'args': args
    }, diffusion_save_path)
    print(f"Global Diffusion Model saved to: {diffusion_save_path}")

    print("\n" + "="*60)
    print("联邦学习训练完成!")
    print(f"Global Style Conditioner: {style_save_path}")
    print(f"Global Diffusion Model: {diffusion_save_path}")
    print("="*60)

if __name__ == "__main__":
    main()
