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
from data_load.get_domainhar import get_acthar
import torch.utils.data as data

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_style_conditioner_client(args, client_id):
    """为指定client训练Style Conditioner"""
    print("=" * 50)
    print(f"开始为Client {client_id}训练 Style Conditioner")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # Some key info
    data_type = args.selected_dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed
    testuser = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+f'_client{client_id}'
    batch_size = args.batch_size
    
    # Load data for specific client
    train_loader, valid_loader, target_loader, _ = get_acthar(args, data_type, target, batch_size=args.batch_size, remain_rate=remain_rate, seed=SEED, train_diff=0, client_id=client_id)
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset
    source_loaders = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    # Import config dynamically
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

    if args.training_mode == "fine_tune":
        load_from = experiment_log_dir
        chkpoint = torch.load(os.path.join(load_from, testuser+"-ckp_last-dl.pt"), map_location=device)
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
        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        logger.debug(f"Training time is : {datetime.now()-start_time}")
        Trainer_ft(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader, device, logger, configs, experiment_log_dir, args.training_mode, testuser)
    else:
        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader, device, logger, configs, experiment_log_dir, args.training_mode, testuser)
        logger.debug(f"Training time is : {datetime.now()-start_time}")
    
    print(f"Client {client_id} Style Conditioner 训练完成!")
    return testuser

def train_diffusion_model_client(args, testuser, client_id):
    """为指定client训练Diffusion Model"""
    print("=" * 50)
    print(f"开始为Client {client_id}训练 Diffusion Model")
    print("=" * 50)
    
    # Prepare some key info 
    data_type = args.selected_dataset
    target = args.target
    remain_rate = args.remain_rate
    
    testuser_dict = {}
    testuser_dict['seed'] = args.seed
    testuser_dict['name'] = testuser

    conditioner = os.getcwd()+os.path.join('/Style_conditioner/conditioner_pth/')
    testuser_dict['conditioner'] = conditioner + testuser + '-' + f'ckp_last-dl.pt'
    
    train_loader, valid_loader, target_loader, testuser_dict['n_class'] = get_acthar(args, data_type, target, batch_size=64, remain_rate=remain_rate, seed=testuser_dict['seed'], client_id=client_id)
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

def aggregate_diffusion_models(client_models, args):
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
    global_results_folder = os.path.join(args.results_folder, 'global_model')
    os.makedirs(global_results_folder, exist_ok=True)
    
    data_type = args.selected_dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed
    global_model_name = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+'_global'
    
    # 保存全局模型
    global_model_path = os.path.join(global_results_folder, global_model_name + '.pt')
    torch.save({
        'model': global_model.state_dict(),
        'step': args.local_training_steps,
        'aggregation_round': args.aggregation_num
    }, global_model_path)
    
    print(f"全局模型已保存到: {global_model_path}")
    print("模型聚合完成!")
    return global_model

def main():
    parser = argparse.ArgumentParser()

    ######################## Model parameters ########################
    home_dir = os.getcwd()
    
    # Common parameters
    parser.add_argument('--seed', default=1, type=int, help='seed value')
    parser.add_argument('--selected_dataset', default='uschad', type=str, help='Dataset of choice: pamap, uschad, dsads')
    parser.add_argument('--remain_rate', default=0.2, type=float, help='Using training data ranging from 0.2 to 1.0')
    parser.add_argument('--target', default=4, type=int, help='Choose task id')
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--batch_size', default=64, type=int, help='Training batch')
    
    # Style conditioner specific parameters
    parser.add_argument('--experiment_description', default='Exp1', type=str, help='Experiment Description')
    parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
    parser.add_argument('--training_mode', default='self_supervised', type=str, help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear,rl')
    parser.add_argument('--logs_save_dir', default='./Style_conditioner/conditioner_pth/', type=str, help='saving directory')
    parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
    
    # Diffusion model specific parameters
    parser.add_argument('--results_folder', default='./Diffusion_model/dm_pth/', type=str, help='saving directory for diffusion model')
    
    # Federated learning parameters
    parser.add_argument('--local_training_steps', default=20000, type=int, help='local training steps for each client')
    parser.add_argument('--aggregation_num', default=1, type=int, help='number of aggregation rounds')
    parser.add_argument('--num_clients', default=3, type=int, help='number of clients')

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs(args.logs_save_dir, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    print(f"开始联邦学习训练流程:")
    print(f"数据集: {args.selected_dataset}")
    print(f"目标: {args.target}")
    print(f"剩余数据比例: {args.remain_rate}")
    print(f"种子: {args.seed}")
    print(f"批次大小: {args.batch_size}")
    print(f"客户端数量: {args.num_clients}")
    print(f"本地训练步数: {args.local_training_steps}")
    print(f"聚合轮数: {args.aggregation_num}")

    client_testusers = []
    client_diffusion_models = []
    
    # 进行指定轮数的聚合训练
    for round_num in range(args.aggregation_num):
        print(f"\n{'='*60}")
        print(f"开始第 {round_num + 1}/{args.aggregation_num} 轮联邦学习")
        print(f"{'='*60}")
        
        # Step 1: 各个client训练Style Conditioner
        print(f"\n第{round_num + 1}轮: 各Client训练Style Conditioner")
        for client_id in range(1, args.num_clients + 1):
            testuser = train_style_conditioner_client(args, client_id)
            if round_num == 0:  # 只在第一轮记录testuser
                client_testusers.append(testuser)
        
        # Step 2: 各个client训练Diffusion Model
        print(f"\n第{round_num + 1}轮: 各Client训练Diffusion Model")
        round_client_models = []
        for client_id in range(1, args.num_clients + 1):
            if round_num == 0:
                testuser = client_testusers[client_id - 1]
            else:
                # 后续轮次使用第一轮的testuser信息
                testuser = client_testusers[client_id - 1]
            diffusion_model = train_diffusion_model_client(args, testuser, client_id)
            round_client_models.append(diffusion_model)
        
        # Step 3: 聚合各client的Diffusion Model
        print(f"\n第{round_num + 1}轮: 聚合各Client的Diffusion Models")
        global_model = aggregate_diffusion_models(round_client_models, args)
        
        # 如果不是最后一轮，将全局模型分发给各client作为下一轮的初始模型
        if round_num < args.aggregation_num - 1:
            print(f"第{round_num + 1}轮完成，准备下一轮训练...")
            # 这里可以添加将全局模型参数分发给各client的逻辑
        else:
            print(f"所有{args.aggregation_num}轮联邦学习完成!")
            client_diffusion_models = round_client_models
    
    print("=" * 60)
    print("联邦学习训练完成!")
    print(f"最终全局模型已保存")
    print(f"各客户端模型数量: {len(client_diffusion_models)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
###########################################################
# import torch
# import os
# import numpy as np
# from datetime import datetime
# import argparse
# import sys
# sys.path.append('.')
# sys.path.append('./Style_conditioner/')
# sys.path.append('./Diffusion_model/')

# # Style conditioner imports
# from Style_conditioner.utils import _logger, set_requires_grad
# from Style_conditioner.trainer.trainer import Trainer, Trainer_ft
# from Style_conditioner.models.TC import TC
# from Style_conditioner.models.model import base_Model

# # Diffusion model imports
# from denoising_diffusion_pytorch import Trainer1D_Train as Trainer1D, Unet1D_cond_train as Unet1D_cond, GaussianDiffusion1Dcond_train as GaussianDiffusion1Dcond

# # Data loading
# from data_load.get_domainhar import get_acthar
# import torch.utils.data as data

# # Set device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def train_style_conditioner(args):
#     """训练Style Conditioner部分"""
#     print("=" * 50)
#     print("开始训练 Style Conditioner")
#     print("=" * 50)
    
#     start_time = datetime.now()
    
#     # Some key info
#     data_type = args.selected_dataset
#     target = args.target
#     remain_rate = args.remain_rate
#     SEED = args.seed
#     testuser = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)
#     batch_size = args.batch_size
    
#     # Load data
#     train_loader, valid_loader, target_loader, _ = get_acthar(args, data_type, target, batch_size=args.batch_size, remain_rate=remain_rate, seed=SEED, train_diff=0)
#     train_dataset = train_loader.dataset
#     valid_dataset = valid_loader.dataset
#     source_loaders = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

#     # Import config dynamically
#     import importlib
#     module_name = f'Style_conditioner.config_files.{data_type}_Configs'
#     ConfigModule = importlib.import_module(module_name)
#     configs = ConfigModule.Config()
#     configs.batch_size = batch_size

#     # Fix random seeds for reproducibility
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

#     if args.training_mode == "fine_tune":
#         load_from = experiment_log_dir
#         chkpoint = torch.load(os.path.join(load_from, testuser+"-ckp_last-dl.pt"), map_location=device)
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
#         model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
#         temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
#         logger.debug(f"Training time is : {datetime.now()-start_time}")
#         Trainer_ft(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader, device, logger, configs, experiment_log_dir, args.training_mode, testuser)
#     else:
#         model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
#         temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
#         Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, source_loaders, valid_loader, target_loader, device, logger, configs, experiment_log_dir, args.training_mode, testuser)
#         logger.debug(f"Training time is : {datetime.now()-start_time}")
    
#     print("Style Conditioner 训练完成!")
#     return testuser

# def train_diffusion_model(args, testuser):
#     """训练Diffusion Model部分"""
#     print("=" * 50)
#     print("开始训练 Diffusion Model")
#     print("=" * 50)
    
#     # Prepare some key info 
#     data_type = args.selected_dataset
#     target = args.target
#     remain_rate = args.remain_rate
    
#     testuser_dict = {}
#     testuser_dict['seed'] = args.seed
#     testuser_dict['name'] = testuser

#     conditioner = os.getcwd()+os.path.join('/Style_conditioner/conditioner_pth/')
#     testuser_dict['conditioner'] = conditioner + testuser + '-' + f'ckp_last-dl.pt'
    
#     train_loader, valid_loader, target_loader, testuser_dict['n_class'] = get_acthar(args, data_type, target, batch_size=64, remain_rate=remain_rate, seed=testuser_dict['seed'])
#     source_loaders = train_loader

#     # Remaining data
#     testuser_dict['remain_data'] = remain_rate
#     print("Remain:", testuser_dict['remain_data'])

#     for minibatch in source_loaders:
#         batch_size = minibatch[0].shape[0]
#         print("print shape X:", minibatch[0].shape)
#         shapex = [minibatch[0].shape[0], minibatch[0].shape[1], minibatch[0].shape[2]]  # length,channel
#         break
    
#     if shapex[1] % 64 != 0:  # Pad the length for Unet
#         shapex[1] = 64 - (shapex[1] % 64) + shapex[1]

#     model_our = Unet1D_cond(    
#         dim=64,
#         num_classes=100,  # style condition embedding dim
#         dim_mults=(1, 2, 4, 8),
#         channels=shapex[2],
#         context_using=True  # use style condition
#     )

#     diffusion = GaussianDiffusion1Dcond(
#         model_our,
#         seq_length=shapex[1],
#         timesteps=100,  
#         objective='pred_noise'
#     )
#     diffusion = diffusion.to(device)
    
#     train_loader = source_loaders
#     trainer = Trainer1D(
#         diffusion,
#         dataloader=train_loader,
#         train_batch_size=shapex[0],
#         train_lr=2e-4,
#         train_num_steps=60001,       # total training steps
#         gradient_accumulate_every=2,    # gradient accumulation steps
#         ema_decay=0.995,                # exponential moving average decay
#         amp=False,                       # turn on mixed precision 32->16
#         results_folder=args.results_folder 
#     )

#     trainer.train(testuser_dict)
#     print("Diffusion Model 训练完成!")

# def main():
#     parser = argparse.ArgumentParser()

#     ######################## Model parameters ########################
#     home_dir = os.getcwd()
    
#     # Common parameters
#     parser.add_argument('--seed', default=1, type=int, help='seed value')
#     parser.add_argument('--selected_dataset', default='uschad', type=str, help='Dataset of choice: pamap, uschad, dsads')
#     parser.add_argument('--remain_rate', default=0.2, type=float, help='Using training data ranging from 0.2 to 1.0')
#     parser.add_argument('--target', default=4, type=int, help='Choose task id')
#     parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
#     parser.add_argument('--batch_size', default=64, type=int, help='Training batch')
    
#     # Style conditioner specific parameters
#     parser.add_argument('--experiment_description', default='Exp1', type=str, help='Experiment Description')
#     parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
#     parser.add_argument('--training_mode', default='self_supervised', type=str, help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear,rl')
#     parser.add_argument('--logs_save_dir', default='./Style_conditioner/conditioner_pth/', type=str, help='saving directory')
#     parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
    
#     # Diffusion model specific parameters
#     parser.add_argument('--results_folder', default='./Diffusion_model/dm_pth/', type=str, help='saving directory for diffusion model')

#     args = parser.parse_args()

#     # Create necessary directories
#     os.makedirs(args.logs_save_dir, exist_ok=True)
#     os.makedirs(args.results_folder, exist_ok=True)

#     print(f"开始训练流程:")
#     print(f"数据集: {args.selected_dataset}")
#     print(f"目标: {args.target}")
#     print(f"剩余数据比例: {args.remain_rate}")
#     print(f"种子: {args.seed}")
#     print(f"批次大小: {args.batch_size}")

#     # Step 1: Train Style Conditioner
#     testuser = train_style_conditioner(args)
    
#     # Step 2: Train Diffusion Model
#     train_diffusion_model(args, testuser)
    
#     print("=" * 50)
#     print("所有训练完成!")
#     print("=" * 50)

# if __name__ == "__main__":
#     main()