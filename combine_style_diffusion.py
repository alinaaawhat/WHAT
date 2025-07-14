import torch
import os
import numpy as np
from datetime import datetime
import argparse
import sys
sys.path.append('.')
sys.path.append('./Style_conditioner/')
sys.path.append('./Diffusion_model/')

# Style conditioner imports
from Style_conditioner.utils import _logger, set_requires_grad
from Style_conditioner.trainer.trainer import Trainer, Trainer_ft
from Style_conditioner.models.TC import TC
from Style_conditioner.models.model import base_Model

# Diffusion model imports
from Diffusion_model.denoising_diffusion_pytorch import Trainer1D_Train as Trainer1D, Unet1D_cond_train as Unet1D_cond, GaussianDiffusion1Dcond_train as GaussianDiffusion1Dcond

# Data loading
from data_load.get_domainhar import get_acthar,get_acthar_client,get_acthar_client_fre
import torch.utils.data as data

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_style_conditioner_client(args, client_id, global_model_path=None):
    """为指定client训练Style Conditioner"""
    print("=" * 50)
    print(f"开始为Client {client_id}训练 Style Conditioner")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # Some key info
    data_type = args.dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed
    testuser = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+f'_client{client_id}'
    batch_size = args.batch_size
    
    # Load data for specific client
    train_loader, valid_loader, target_loader, _ = get_acthar_client_fre(args, data_type, target, batch_size=args.batch_size, remain_rate=remain_rate, seed=SEED, train_diff=0, client_id=client_id)
    # for batch in train_loader:
    #     data, labels = batch[0], batch[1] 
    #     print(f"数据形状: {data.shape}, 数据范围: [{data.min():.4f}, {data.max():.4f}]")
    #     if torch.isnan(data).any():
    #         print("❌ 数据包含NaN，请检查数据预处理")
    #     break
    train_dataset = train_loader.dataset
    source_loaders = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    # Import config dynamicallynow
    import importlib
    module_name = f'Style_conditioner.config_files.{data_type}_Configs'
    ConfigModule = importlib.import_module(module_name)
    configs = ConfigModule.Config()
    configs.batch_size = batch_size

    # Fix random seeds for reproducibility
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # Setup logging
    experiment_log_dir = os.path.join(args.logs_save_dir, data_type+str(remain_rate)+f"_seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Mode:    {args.training_mode}')
    logger.debug("=" * 45)

    # Load Model
    model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)
    
    # Load global model if provided (for federated learning rounds after the first)
    if global_model_path and os.path.exists(global_model_path):
        print(f"Loading global model from: {global_model_path}")
        global_checkpoint = torch.load(global_model_path, map_location=device)
        model.load_state_dict(global_checkpoint['model_state_dict'])
        temporal_contr_model.load_state_dict(global_checkpoint['temporal_contr_state_dict'])

    if args.training_mode == "fine_tune":
        load_from = experiment_log_dir
        chkpoint_path = './Style_conditioner/conditioner_pth/global_style_conditioner/uschad_tar_0_rm_0.2seed_1_global.pt'
        chkpoint = torch.load(chkpoint_path, map_location=device)
        logs_save_dir = './Style_conditioner/conditioner_pth/'
        experiment_log_dir = os.path.join(logs_save_dir, data_type+str(remain_rate)+f"_seed_{SEED}")
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()
        del_list = ['logits']
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_dict[i]
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # 在创建模型和优化器之间添加：
        print("🔧 修复数据类型...")

        # 修复train_loader的数据类型
        original_train_loader = train_loader
        train_dataset = train_loader.dataset

        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        #########################
        # model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
        # temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
        logger.debug(f"Training time is : {datetime.now()-start_time}")
        Trainer_ft(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader, device, logger, configs, experiment_log_dir, args.training_mode, testuser)
    else:
        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader, device, logger, configs, experiment_log_dir, args.training_mode, testuser)
        logger.debug(f"Training time is : {datetime.now()-start_time}")
    
    print(f"Client {client_id} Style Conditioner 训练完成!")
    return testuser, model, temporal_contr_model
# def train_style_conditioner_client(args, client_id, global_model_path=None):
#     """为指定client训练Style Conditioner"""
#     print("=" * 50)
#     print(f"开始为Client {client_id}训练 Style Conditioner")
#     print("=" * 50)
    
#     start_time = datetime.now()
    
#     # Some key info
#     data_type = args.dataset
#     target = args.target
#     remain_rate = args.remain_rate
#     SEED = args.seed
#     testuser = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+f'_client{client_id}'
#     batch_size = args.batch_size
    
#     # Load data for specific client
#     train_loader, valid_loader, target_loader, _ = get_acthar_client_fre(args, data_type, target, batch_size=args.batch_size, remain_rate=remain_rate, seed=SEED, train_diff=0, client_id=client_id)
    
#     # 🔧 数据检查和修复（移到这里，所有模式都执行）
#     print("🔍 检查数据...")
#     for batch in train_loader:
#         data, labels = batch[0], batch[1] 
#         print(f"数据形状: {data.shape}")
#         print(f"数据类型: {data.dtype}")
#         print(f"标签类型: {labels.dtype}")
        
#         # 检查数据范围（安全方式）
#         if data.dtype not in [torch.complex128, torch.complex64]:
#             print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")
#         else:
#             print("⚠️ 检测到复数数据类型")
        
#         if torch.isnan(data).any():
#             print("❌ 数据包含NaN，请检查数据预处理")
#         break
    
#     # 🔧 定义数据类型修复函数
#     def collate_fn_fixed(batch):
#         data_list, label_list = [], []
#         for data, label in batch:
#             # 修复数据类型
#             if data.dtype in [torch.complex128, torch.complex64]:
#                 data = data.real.float()  # 取实部
#             else:
#                 data = data.float()  # 确保是float32
#             label = label.long()  # 确保标签是long
#             data_list.append(data)
#             label_list.append(label)
#         return torch.stack(data_list), torch.stack(label_list)
    
#     # 🔧 重新创建修复后的数据加载器
#     train_dataset = train_loader.dataset
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         drop_last=True, 
#         shuffle=True,
#         collate_fn=collate_fn_fixed
#     )
    
#     valid_dataset = valid_loader.dataset  
#     valid_loader = torch.utils.data.DataLoader(
#         valid_dataset,
#         batch_size=batch_size,
#         drop_last=False,
#         shuffle=False,
#         collate_fn=collate_fn_fixed
#     )
    
#     target_dataset = target_loader.dataset
#     target_loader = torch.utils.data.DataLoader(
#         target_dataset,
#         batch_size=batch_size, 
#         drop_last=False,
#         shuffle=False,
#         collate_fn=collate_fn_fixed
#     )
    
#     source_loaders = train_loader

#     # Import config
#     import importlib
#     module_name = f'Style_conditioner.config_files.{data_type}_Configs'
#     ConfigModule = importlib.import_module(module_name)
#     configs = ConfigModule.Config()
#     configs.batch_size = batch_size

#     # Fix random seeds
#     torch.manual_seed(SEED)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(SEED)

#     # Setup logging
#     experiment_log_dir = os.path.join(args.logs_save_dir, data_type+str(remain_rate)+f"_seed_{SEED}")
#     os.makedirs(experiment_log_dir, exist_ok=True)
#     log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
#     logger = _logger(log_file_name)
#     logger.debug("=" * 45)
#     logger.debug(f'Dataset: {data_type}')
#     logger.debug(f'Mode:    {args.training_mode}')
#     logger.debug("=" * 45)

#     # Load Model
#     model = base_Model(configs).to(device)
#     temporal_contr_model = TC(configs, device).to(device)
    
#     # Load global model if provided
#     if global_model_path and os.path.exists(global_model_path):
#         print(f"Loading global model from: {global_model_path}")
#         global_checkpoint = torch.load(global_model_path, map_location=device)
#         model.load_state_dict(global_checkpoint['model_state_dict'])
#         temporal_contr_model.load_state_dict(global_checkpoint['temporal_contr_state_dict'])

#     # 🔧 统一的优化器设置（修复学习率和参数）
#     model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
#     temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)

#     if args.training_mode == "fine_tune":
#         load_from = experiment_log_dir
#         chkpoint_path = './Style_conditioner/conditioner_pth/global_style_conditioner/uschad_tar_0_rm_0.2seed_1_global.pt'
#         chkpoint = torch.load(chkpoint_path, map_location=device)
#         logs_save_dir = './Style_conditioner/conditioner_pth/'
#         experiment_log_dir = os.path.join(logs_save_dir, data_type+str(remain_rate)+f"_seed_{SEED}")
#         pretrained_dict = chkpoint["model_state_dict"]
#         model_dict = model.state_dict()
#         del_list = ['logits']
#         pretrained_dict_copy = pretrained_dict.copy()
#         for i in pretrained_dict_copy.keys():
#             for j in del_list:
#                 if j in i:
#                     del pretrained_dict[i]
#         model_dict.update(pretrained_dict)
#         model.load_state_dict(model_dict)
        
#         logger.debug(f"Training time is : {datetime.now()-start_time}")
#         Trainer_ft(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader, device, logger, configs, experiment_log_dir, args.training_mode, testuser)
#     else:
#         Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader, device, logger, configs, experiment_log_dir, args.training_mode, testuser)
#         logger.debug(f"Training time is : {datetime.now()-start_time}")
    
#     print(f"Client {client_id} Style Conditioner 训练完成!")
#     return testuser, model, temporal_contr_model
def aggregate_style_conditioners(client_models, client_temporal_models, args, round_num):
    """聚合各个client的Style Conditioner"""
    print("=" * 50)
    print("开始聚合各Client的Style Conditioners")
    print("=" * 50)
    
    # 使用第一个client的模型作为基础
    global_model = client_models[0]
    global_temporal_model = client_temporal_models[0]
    
    # 获取状态字典
    global_state_dict = global_model.state_dict()
    global_temporal_state_dict = global_temporal_model.state_dict()
    
    # 收集所有client的参数
    client_state_dicts = [model.state_dict() for model in client_models]
    client_temporal_state_dicts = [model.state_dict() for model in client_temporal_models]
    
    # 对每个参数进行平均 - base_Model
    for key in global_state_dict.keys():
        param_sum = torch.zeros_like(global_state_dict[key])
        for client_state_dict in client_state_dicts:
            param_sum += client_state_dict[key]
        global_state_dict[key] = param_sum / len(client_models)
    
    # 对每个参数进行平均 - TC (temporal contrastive model)
    for key in global_temporal_state_dict.keys():
        param_sum = torch.zeros_like(global_temporal_state_dict[key])
        for client_temporal_state_dict in client_temporal_state_dicts:
            param_sum += client_temporal_state_dict[key]
        global_temporal_state_dict[key] = param_sum / len(client_temporal_models)
    
    # 更新全局模型
    global_model.load_state_dict(global_state_dict)
    global_temporal_model.load_state_dict(global_temporal_state_dict)
    
    # 保存聚合后的Style Conditioner
    global_style_folder = os.path.join(args.logs_save_dir, 'global_style_conditioner')
    os.makedirs(global_style_folder, exist_ok=True)
    
    data_type = args.dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed
    global_style_name = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+f'_global'
    
    # 保存全局Style Conditioner
    global_style_path = os.path.join(global_style_folder, global_style_name + '.pt')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'temporal_contr_state_dict': global_temporal_model.state_dict(),
        'round': round_num
    }, global_style_path)
    
    print(f"全局Style Conditioner已保存到: {global_style_path}")
    print("Style Conditioner聚合完成!")
    return global_model, global_temporal_model, global_style_path

def train_diffusion_model_client(args, testuser, client_id, global_diffusion_model=None):
    """为指定client训练Diffusion Model"""
    print("=" * 50)
    print(f"开始为Client {client_id}训练 Diffusion Model")
    print("=" * 50)
    
    # Prepare some key info 
    data_type = args.dataset
    target = args.target
    remain_rate = args.remain_rate
    
    testuser_dict = {}
    testuser_dict['seed'] = args.seed
    testuser_dict['name'] = testuser
    global_style_folder = os.path.join(args.logs_save_dir, 'global_style_conditioner')
    os.makedirs(global_style_folder, exist_ok=True)

    SEED = args.seed
    global_style_name = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+f'_global'
    
    
    global_style_path = os.path.join(global_style_folder, global_style_name + '.pt')
    print(global_style_path)

    # 添加调试和检查
    print(f"🔍 Debug - Client {client_id} 期望的Style Conditioner路径: {global_style_path}")
    print(f"🔍 Debug - 文件是否存在: {os.path.exists(global_style_path)}")
    
    # if os.path.exists(global_style_path):
    #     # 检查文件内容
    #     import torch
    #     try:
    #         ckpt = torch.load(global_style_path, map_location='cpu')
    #         model_dict = ckpt['model_state_dict']
    #         conv1_shape = model_dict['conv_block1.0.weight'].shape
    #         logits_shape = model_dict['logits.weight'].shape
    #         print(f"🔍 Debug - 文件中的模型配置:")
    #         print(f"    conv1通道数: {conv1_shape[1]}")
    #         print(f"    输出类别数: {logits_shape[0]}")
            
    #         # 检查是否匹配当前数据集
    #         if conv1_shape[1] != 6 or logits_shape[0] != 12:
    #             print(f"⚠️  警告: Style Conditioner配置与USC-HAD不匹配!")
    #             print(f"    期望: 6通道, 12类别")
    #             print(f"    实际: {conv1_shape[1]}通道, {logits_shape[0]}类别")
    #     except Exception as e:
    #         print(f"❌ 读取Style Conditioner文件失败: {e}")
    # else:
    #     print(f"❌ Style Conditioner文件不存在!")
    
    testuser_dict['conditioner'] = global_style_path
    # conditioner = os.getcwd()+os.path.join('/Style_conditioner/conditioner_pth/')
    testuser_dict['conditioner'] = global_style_path
    # testuser_dict['conditioner'] = '/home/SHIH0020/robustlearn/Style_conditioner/conditioner_pth/global_style_conditioner/uschad_tar_0_rm_0.2seed_1_global.pt'
    train_loader, valid_loader, target_loader, testuser_dict['n_class'] = get_acthar_client_fre(args, data_type, target, batch_size=64, remain_rate=remain_rate, seed=testuser_dict['seed'], client_id=client_id)
    source_loaders = train_loader
    
    # Remaining data
    testuser_dict['remain_data'] = remain_rate
    print(f"Client {client_id} Remain:", testuser_dict['remain_data'])

    for minibatch in source_loaders:
        batch_size = minibatch[0].shape[0]
        print(f"Client {client_id} print shape X:", minibatch[0].shape)
        shapex = [minibatch[0].shape[0], minibatch[0].shape[1], minibatch[0].shape[2]]  # length,channel
        break
    
    if shapex[1] % 64 != 0:  # Pad the length for Unet
        shapex[1] = 64 - (shapex[1] % 64) + shapex[1]

    model_our = Unet1D_cond(    
        dim=64,
        num_classes=100,  # style condition embedding dim
        dim_mults=(1, 2, 4, 8),
        channels=shapex[2],
        context_using=True  # use style condition
    )

    diffusion = GaussianDiffusion1Dcond(
        model_our,
        seq_length=shapex[1],
        timesteps=100,  
        objective='pred_noise'
    )
    diffusion = diffusion.to(device)
    
    # Load global diffusion model if provided (for federated learning rounds after the first)
    if global_diffusion_model is not None:
        print(f"Loading global diffusion model for Client {client_id}")
        diffusion.load_state_dict(global_diffusion_model.state_dict())
    
    train_loader = source_loaders
    # 修改保存路径以区分不同client
    client_results_folder = os.path.join(args.results_folder, f'client_{client_id}')
    os.makedirs(client_results_folder, exist_ok=True)
    
    trainer = Trainer1D(
        diffusion,
        dataloader=train_loader,
        train_batch_size=shapex[0],
        train_lr=2e-4,
        train_num_steps=args.local_training_steps,  # 使用本地训练步数
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        results_folder=client_results_folder
    )

    trainer.train(testuser_dict)
    print(f"Client {client_id} Diffusion Model 训练完成!")
    return diffusion

def aggregate_diffusion_models(client_models, args, round_num):
    """聚合各个client的diffusion model"""
    print("=" * 50)
    print("开始聚合各Client的Diffusion Models")
    print("=" * 50)
    
    # 使用第一个client的模型作为基础
    global_model = client_models[0]
    global_state_dict = global_model.state_dict()
    
    # 收集所有client的参数
    client_state_dicts = [model.state_dict() for model in client_models]
    
    # 对每个参数进行平均
    for key in global_state_dict.keys():
        # 计算所有client该参数的平均值
        param_sum = torch.zeros_like(global_state_dict[key])
        for client_state_dict in client_state_dicts:
            param_sum += client_state_dict[key]
        global_state_dict[key] = param_sum / len(client_models)
    
    # 更新全局模型
    global_model.load_state_dict(global_state_dict)
    
    # 保存聚合后的模型
    global_results_folder = os.path.join(args.results_folder, 'global_diffusion_model')
    os.makedirs(global_results_folder, exist_ok=True)
    
    data_type = args.dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed
    global_model_name = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+f'_global_round_{round_num}'
    
    # 保存全局模型
    global_model_path = os.path.join(global_results_folder, global_model_name + '.pt')
    torch.save({
        'model': global_model.state_dict(),
        'step': args.local_training_steps,
        'aggregation_round': round_num
    }, global_model_path)
    
    print(f"全局Diffusion Model已保存到: {global_model_path}")
    print("Diffusion Model聚合完成!")
    return global_model

def main():
    parser = argparse.ArgumentParser()

    ######################## Model parameters ########################
    home_dir = os.getcwd()
    
    # Common parameters
    parser.add_argument('--seed', default=1, type=int, help='seed value')
    parser.add_argument('--dataset', default='emg', type=str, help='Dataset of choice: pamap, uschad, dsads')
    parser.add_argument('--selected_dataset', default='emg', type=str, help='Dataset of choice: pamap, uschad, dsads')

    parser.add_argument('--remain_rate', default=0.2, type=float, help='Using training data ranging from 0.2 to 1.0')
    parser.add_argument('--remain_data_rate', default=0.2, type=float, help='Using training data ranging from 0.2 to 1.0')

    parser.add_argument('--target', default=0, type=int, help='Choose task id')
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--batch_size', default=32, type=int, help='Training batch')
    
    # Style conditioner specific parameters
    parser.add_argument('--experiment_description', default='Exp1', type=str, help='Experiment Description')
    parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
    parser.add_argument('--training_mode', default='self_supervised', type=str, help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear,rl')
    parser.add_argument('--logs_save_dir', default='./Style_conditioner/conditioner_pth/', type=str, help='saving directory')
    parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
    
    # Diffusion model specific parameters
    parser.add_argument('--results_folder', default='./Diffusion_model/dm_pth/', type=str, help='saving directory for diffusion model')
    
    # Federated learning parameters
    parser.add_argument('--local_training_steps', default=1, type=int, help='local training steps for each client')
    parser.add_argument('--aggregation_num', default=1, type=int, help='number of aggregation rounds')
    parser.add_argument('--num_clients', default=15, type=int, help='number of clients')
    parser.add_argument('--local_training_steps_style', default=1, type=int, help='local training steps for each client')
    parser.add_argument('--aggregation_num_style', default=1, type=int, help='number of aggregation rounds')

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs(args.logs_save_dir, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    print(f"开始联邦学习训练流程:")
    print(f"数据集: {args.dataset}")
    print(f"目标: {args.target}")
    print(f"剩余数据比例: {args.remain_rate}")
    print(f"种子: {args.seed}")
    print(f"批次大小: {args.batch_size}")
    print(f"客户端数量: {args.num_clients}")
    print(f"本地训练步数: {args.local_training_steps}")
    print(f"聚合轮数: {args.aggregation_num}")

    # 用于记录各client的testuser信息
    client_testusers = []
    # 全局模型变量
    global_style_model_path = None
    global_diffusion_model = None
    
    # Phase 1: Style Conditioner 联邦学习
    print(f"\n{'='*80}")
    print(f"阶段1: Style Conditioner 联邦学习 ({args.aggregation_num_style} 轮)")
    print(f"{'='*80}")
    
    for round_num in range(args.aggregation_num_style):
        print(f"\n{'='*60}")
        print(f"Style Conditioner 第 {round_num + 1}/{args.aggregation_num_style} 轮")
        print(f"{'='*60}")
        
        # Step 1: 各个client训练Style Conditioner
        print(f"\n第{round_num + 1}轮: 各Client训练Style Conditioner")
        round_client_style_models = []
        round_client_temporal_models = []
        
        for client_id in range(1, args.num_clients + 1):
            testuser, style_model, temporal_model = train_style_conditioner_client(
                args, client_id, global_style_model_path
            )
            if round_num == 0:  # 只在第一轮记录testuser
                client_testusers.append(testuser)
            round_client_style_models.append(style_model)
            round_client_temporal_models.append(temporal_model)
        
        # Step 2: 聚合各client的Style Conditioner
        print(f"\n第{round_num + 1}轮: 聚合各Client的Style Conditioners")
        global_style_model, global_temporal_model, global_style_model_path = aggregate_style_conditioners(
            round_client_style_models, round_client_temporal_models, args, round_num
        )
        
        print(f"Style Conditioner 第{round_num + 1}轮完成!")
    
    print(f"\n{'='*80}")
    print(f"阶段1完成: Style Conditioner 联邦学习已完成!")
    print(f"{'='*80}")
    
    # Phase 2: Diffusion Model 联邦学习
    print(f"\n{'='*80}")
    print(f"阶段2: Diffusion Model 联邦学习 ({args.aggregation_num} 轮)")
    print(f"{'='*80}")
    if not client_testusers:  
        data_type = args.dataset
        target = args.target
        remain_rate = args.remain_rate
        SEED = args.seed
        for client_id in range(1, args.num_clients + 1):
            testuser = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+f'_client{client_id}'
            client_testusers.append(testuser)

    for round_num in range(args.aggregation_num):
        print(f"\n{'='*60}")
        print(f"Diffusion Model 第 {round_num + 1}/{args.aggregation_num} 轮")
        print(f"{'='*60}")
        
        # Step 1: 各个client训练Diffusion Model
        print(f"\n第{round_num + 1}轮: 各Client训练Diffusion Model")
        round_client_diffusion_models = []
        
        for client_id in range(1, args.num_clients + 1):
            testuser = client_testusers[client_id - 1]  # 使用已记录的testuser
            diffusion_model = train_diffusion_model_client(
                args, testuser, client_id, global_diffusion_model
            )
            round_client_diffusion_models.append(diffusion_model)
        
        # Step 2: 聚合各client的Diffusion Model
        print(f"\n第{round_num + 1}轮: 聚合各Client的Diffusion Models")
        global_diffusion_model = aggregate_diffusion_models(
            round_client_diffusion_models, args, round_num
        )
        
        print(f"Diffusion Model 第{round_num + 1}轮完成!")
    
    print("=" * 80)
    print("联邦学习训练完成!")
    print(f"阶段1: Style Conditioner 联邦学习({args.aggregation_num}轮) - 已完成")
    print(f"阶段2: Diffusion Model 联邦学习({args.aggregation_num}轮) - 已完成")
    print(f"最终全局Style Conditioner模型已保存到: {global_style_model_path}")
    print(f"最终全局Diffusion Model已保存")
    print("=" * 80)

if __name__ == "__main__":
    main()
