import torch
import sys
sys.path.append('.')
sys.path.append('./Diffusion_model/')
from Diffusion_model.denoising_diffusion_pytorch import Unet1D_cond,GaussianDiffusion1Dcond
from data_load.get_domainhar import get_acthar
import torch.nn as nn
import torch.utils.data as data
from train_strategy import train_diversity
from copy import deepcopy
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Featurenet.utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
import os
import argparse
import numpy as np
from Featurenet.config_files.distrb_condition import cond_set
from Featurenet.alg import alg

def set_random_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def federated_average(client_models):
    """
    联邦平均算法：聚合客户端模型参数
    """
    if not client_models:
        print("警告：没有客户端模型可供聚合")
        return None
    
    print(f"开始聚合 {len(client_models)} 个客户端模型...")
    
    # 获取第一个模型作为基础
    global_model = deepcopy(client_models[0])
    global_state_dict = global_model.state_dict()
    
    # 只聚合指定的模块
    target_modules = ['featurizer', 'projection', 'classifier']
    
    # 统计需要聚合的参数
    aggregated_params = []
    for key in global_state_dict.keys():
        if any(module in key for module in target_modules):
            aggregated_params.append(key)
    
    print(f"将聚合以下模块的参数: {aggregated_params}")
    
    # 初始化聚合参数为零
    for key in aggregated_params:
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])
    
    # 对每个参数进行平均
    for i, model in enumerate(client_models):
        model_state_dict = model.state_dict()
        print(f"聚合客户端 {i+1} 的参数...")
        
        for key in aggregated_params:
            if key in model_state_dict:
                global_state_dict[key] += model_state_dict[key]
            else:
                print(f"警告：客户端 {i+1} 缺少参数 {key}")
    
    # 计算平均值
    for key in aggregated_params:
        global_state_dict[key] /= len(client_models)
    
    global_model.load_state_dict(global_state_dict)
    print("模型聚合完成")
    return global_model

def distribute_model(global_model, client_models):
    """
    将全局模型参数分发给客户端
    """
    if global_model is None:
        print("警告：全局模型为空，跳过分发")
        return client_models
    
    print(f"开始向 {len(client_models)} 个客户端分发全局模型...")
    
    global_state_dict = global_model.state_dict()
    target_modules = ['featurizer', 'projection', 'classifier']
    
    # 统计要分发的参数
    distributed_params = []
    for key in global_state_dict.keys():
        if any(module in key for module in target_modules):
            distributed_params.append(key)
    
    print(f"将分发以下模块的参数: {distributed_params}")
    
    for i, client_model in enumerate(client_models):
        client_state_dict = client_model.state_dict()
        
        # 只更新目标模块的参数
        updated_params = 0
        for key in distributed_params:
            if key in client_state_dict:
                client_state_dict[key] = global_state_dict[key].clone()
                updated_params += 1
            else:
                print(f"警告：客户端 {i+1} 缺少参数 {key}")
        
        client_model.load_state_dict(client_state_dict)
        print(f"客户端 {i+1} 更新了 {updated_params} 个参数")
    
    print("模型分发完成")
    return client_models

def evaluate_global_model(global_model, test_loaders, device):
    """评估全局模型在所有客户端测试数据上的性能"""
    if global_model is None:
        return {}
    
    global_model.eval()
    results = {}
    
    with torch.no_grad():
        for client_id, test_loader in test_loaders.items():
            correct = 0
            total = 0
            
            for data, target, domain in test_loader:
                data, target = data.to(device), target.to(device).long()
                
                # 数据预处理
                if len(data.shape) == 3:
                    data = data.unsqueeze(3)
                data = data.transpose(1, 2).squeeze(3).unsqueeze(2)
                
                try:
                    predictions, _ = global_model.predict(data)
                    _, predicted = torch.max(predictions.data, 1)
                    
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                except:
                    continue
            
            accuracy = correct / max(total, 1)
            results[f'client_{client_id}'] = accuracy
    
    return results
def save_federated_model(global_model, round_num, save_dir, args):
    """保存联邦学习的全局模型"""
    os.makedirs(save_dir, exist_ok=True)
    model_name = f"federated_global_model_round{round_num}_{args.dataset}_target{args.target}_seed{args.seed}.pt"
    model_path = os.path.join(save_dir, model_name)
    
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'round': round_num,
        'args': args
    }, model_path)
    
    print(f"Global model saved at round {round_num}: {model_path}")

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()

parser.add_argument('--seed', default=1, type=int, help='seed value')
parser.add_argument('--dataset', default='dsads', type=str, help='Dataset of choice: pamap, uschad, dsads')
parser.add_argument('--target', default=0,type=int, help='Choose task id')
parser.add_argument('--remain_data_rate', default=0.2,type=float, help='Using training data ranging from 0.2 to 1.0')
parser.add_argument('--remain_rate', default=0.2,type=float, help='Using training data ranging from 0.2 to 1.0')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
parser.add_argument('--batch_size', default=64,type=int, help='Training batch')
parser.add_argument('--Ocomb', default=5,type=int, help='the maxmium of combination')
parser.add_argument('--Ktimes', default=2,type=int, help='the ratio between new and ori')
parser.add_argument('--lr_decay_cls_f', default=1e-1,type=float, help='step-3-fea')
parser.add_argument('--lr_decay_cls', default=1e-1,type=float, help='step-3-net')
parser.add_argument('--lr_decay_ori', default=1e-1,type=float, help='step-2-net')
parser.add_argument('--lr_decay_ori_f', default=1e-1,type=float, help='step-2-fea')
parser.add_argument('--lr_decay1', default=1e-2,type=float, help='step-1-fea')
parser.add_argument('--lr_decay2', default=1,type=float, help='step-1-net')
parser.add_argument('--lr', type=float, default=7e-3, help="learning rate")
parser.add_argument('--step1', default=0,type=int, help='epoch1')
parser.add_argument('--step2', default=0,type=int, help='epoch2')
parser.add_argument('--step3', default=1,type=int, help='epoch3')


# 联邦学习特定参数
parser.add_argument('--num_clients', default=3, type=int, help='total number of clients')
parser.add_argument('--num_rounds', default=10, type=int, help='number of federated rounds')
parser.add_argument('--local_epochs', default=5, type=int, help='local training epochs per round')
parser.add_argument('--federated_save_dir', default='./federated_models/', type=str, help='directory to save federated models')

args = parser.parse_args()

# Some key info
target = args.target
data_type = args.dataset
remain_rate = args.remain_rate
batch_size = args.batch_size

set_random_seed(args.seed)

print("=" * 60)
print("开始联邦学习训练")
print(f"数据集: {data_type}")
print(f"目标域: {target}")
print(f"客户端数量: {args.num_clients}")
print(f"联邦轮数: {args.num_rounds}")
print(f"本地训练轮数: {args.local_epochs}")
print("=" * 60)

# 初始化客户端模型列表
client_models = []
client_testusers = []
client_diffusions = []

# 为每个客户端初始化数据和模型
for client_id in range(1, args.num_clients + 1):
    print(f"\n初始化客户端 {client_id}...")
    
    # 设置客户端特定信息
    testuser = {}
    testuser['target'] = target
    testuser['maxcond'] = args.Ocomb
    testuser['cond_weight'] = cond_set(data_type, remain_rate, target, testuser['maxcond'])
    testuser['dataset'] = data_type
    testuser['repeat'] = args.Ktimes
    testuser['seed'] = args.seed
    testuser['client_id'] = client_id
    testuser['num_clients'] = args.num_clients
    testuser['enable_cross_client_generation'] = False  # 简化版本先关闭跨客户端生成
    testuser['remain_data'] = args.remain_rate
    
    # 客户端特定的名称
    testuser['name'] = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])+f'_client{client_id}'
    
    # 加载客户端数据
    train_loader, valid_loader, target_loader, testuser['n_class'] = get_acthar(
        args, data_type, target, batch_size=64, remain_rate=testuser['remain_data'], 
        seed=testuser['seed'], client_id=client_id
    )
    
    # 设置路径
    project_root = os.path.dirname(os.getcwd())
    conditioner_base = os.path.join(project_root, 'Style_conditioner', 'conditioner_pth/')
    testuser['conditioner'] = conditioner_base + data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])+f'_client{client_id}'+'-'+f'ckp_last-dl.pt'
    testuser['newdata'] = project_root+'/Featurenet/new_data/' +testuser['name']+'-rep'+str(testuser['repeat'])+'-batch'+str(batch_size)+'-cond'+str(testuser['maxcond']) +'-weight'+(str(testuser['cond_weight']))+'.pt'
    
    testuser['diff'] = project_root+'/Diffusion_model/dm_pth/'+data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])+f'_client{client_id}'+'.pt'
    
    # 初始化扩散模型
    train_dataset = train_loader.dataset
    source_loaders = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    
    for minibatch in source_loaders:
        batch_size_actual = minibatch[0].shape[0]
        shape = [minibatch[0].shape[1], minibatch[0].shape[2]]
        break
    
    testuser['length'] = shape[0]
    if shape[0] % 64 != 0:
        shape[0] = 64 - (shape[0] % 64) + shape[0]
    
    model_our = Unet1D_cond(
        dim=64,
        num_classes=100,
        dim_mults=(1, 2, 4, 8),
        channels=shape[1],
        context_using=True
    )
    
    diffusion = GaussianDiffusion1Dcond(
        model_our,
        seq_length=shape[0],
        timesteps=100,
        objective='pred_noise'
    )
    diffusion = diffusion.to(device)
    
    # 加载预训练的扩散模型
    if os.path.exists(testuser['diff']):
        data_dict = torch.load(testuser['diff'])
        diffusion.load_state_dict(data_dict['model'])
    
    # 初始化算法模型
    dataset = testuser['name'].split('_tar')[0]
    args_data = get_args(dataset, args)
    algorithm_class = alg.get_algorithm_class()
    algorithm = algorithm_class(args_data).cuda()
    
    # 存储客户端信息
    client_models.append(algorithm)
    client_testusers.append(testuser)
    client_diffusions.append(diffusion)
    
    print(f"客户端 {client_id} 初始化完成")

# 联邦学习主循环
global_model = None

for round_num in range(args.num_rounds):
    print(f"\n{'='*50}")
    print(f"联邦学习轮次 {round_num + 1}/{args.num_rounds}")
    print(f"{'='*50}")
    
    # 1. 分发全局模型到各客户端
    if global_model is not None:
        print("分发全局模型到各客户端...")
        client_models = distribute_model(global_model, client_models)
    
    # 2. 各客户端本地训练
    trained_client_models = []
    
    for client_id in range(args.num_clients):
        print(f"\n客户端 {client_id + 1} 开始本地训练...")
        
        # 获取客户端数据
        testuser = client_testusers[client_id]
        diffusion = client_diffusions[client_id]
        algorithm = client_models[client_id]
        
        # 重新加载数据
        train_loader, valid_loader, target_loader, _ = get_acthar(
            args, data_type, target, batch_size=64, remain_rate=testuser['remain_data'], 
            seed=testuser['seed'], client_id=client_id + 1
        )
        
        train_dataset = train_loader.dataset
        source_loaders = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        
        # 本地训练（修改train_diversity以支持有限轮数训练）
        print(f"客户端 {client_id + 1} 训练 {args.local_epochs} 个epoch...")
        
        try:
            # 调用修改后的训练函数
            from train_strategy import train_diversity
            trained_algorithm = train_diversity(
                diffusion, args_data, source_loaders, valid_loader, target_loader, 
                testuser, max_epochs=args.local_epochs
            )
            trained_client_models.append(trained_algorithm)
            
        except Exception as e:
            print(f"客户端 {client_id + 1} 训练失败: {str(e)}")
            # 如果训练失败，使用原模型
            trained_client_models.append(algorithm)
        
        print(f"客户端 {client_id + 1} 本地训练完成")
    
    # 3. 服务器聚合
    print(f"\n服务器聚合模型...")
    global_model = federated_average(trained_client_models)
    
    # 4. 保存全局模型
    save_federated_model(global_model, round_num + 1, args.federated_save_dir, args)
    
    # 5. 更新客户端模型引用
    client_models = [deepcopy(global_model) for _ in range(args.num_clients)]
    
    # 6. 评估全局模型性能（可选）
    if round_num % 2 == 0:  # 每2轮评估一次
        print(f"评估第 {round_num + 1} 轮全局模型性能...")
        test_loaders = {}
        
        # 收集所有客户端的测试数据
        for client_id in range(1, args.num_clients + 1):
            try:
                _, _, target_loader, _ = get_acthar(
                    args, data_type, target, batch_size=64, 
                    remain_rate=args.remain_rate, seed=args.seed, client_id=client_id
                )
                test_loaders[client_id] = target_loader
            except Exception as e:
                print(f"无法加载客户端 {client_id} 的测试数据: {e}")
        
        # 评估性能
        performance = evaluate_global_model(global_model, test_loaders, device)
        for client_id, acc in performance.items():
            print(f"  {client_id} 测试准确率: {acc:.4f}")
        
        # 计算平均性能
        if performance:
            avg_acc = sum(performance.values()) / len(performance)
            print(f"  平均测试准确率: {avg_acc:.4f}")
    
    print(f"联邦学习轮次 {round_num + 1} 完成")

print("\n" + "="*60)
print("联邦学习训练完成！")
print(f"最终全局模型保存在: {args.federated_save_dir}")
print("="*60)













###############################################################################
# import torch
# import sys
# sys.path.append('.')
# sys.path.append('./Diffusion_model/')
# from Diffusion_model.denoising_diffusion_pytorch import Unet1D_cond,GaussianDiffusion1Dcond
# from data_load.get_domainhar import get_acthar
# import torch.nn as nn
# import torch.utils.data as data
# from train_strategy import train_diversity
# from copy import deepcopy
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# from Featurenet.utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
# import os
# import argparse
# import numpy as np
# from Featurenet.config_files.distrb_condition import cond_set
# from Featurenet.alg import alg

# def set_random_seed(SEED):
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def federated_average(client_models):
#     """
#     联邦平均算法：聚合客户端模型参数
#     """
#     if not client_models:
#         return None
    
#     # 获取第一个模型作为基础
#     global_model = deepcopy(client_models[0])
#     global_state_dict = global_model.state_dict()
    
#     # 只聚合指定的模块
#     target_modules = ['featurizer', 'projection', 'classifier']
    
#     # 初始化聚合参数
#     for key in global_state_dict.keys():
#         global_state_dict[key] = torch.zeros_like(global_state_dict[key])
    
#     # 对每个参数进行平均
#     for model in client_models:
#         model_state_dict = model.state_dict()
#         for key in global_state_dict.keys():
#             # 只聚合目标模块的参数
#             if any(module in key for module in target_modules):
#                 global_state_dict[key] += model_state_dict[key]
    
#     # 计算平均值
#     for key in global_state_dict.keys():
#         if any(module in key for module in target_modules):
#             global_state_dict[key] /= len(client_models)
    
#     global_model.load_state_dict(global_state_dict)
#     return global_model

# def distribute_model(global_model, client_models):
#     """
#     将全局模型参数分发给客户端
#     """
#     if global_model is None:
#         return client_models
    
#     global_state_dict = global_model.state_dict()
#     target_modules = ['featurizer', 'projection', 'classifier']
    
#     for i, client_model in enumerate(client_models):
#         client_state_dict = client_model.state_dict()
        
#         # 只更新目标模块的参数
#         for key in client_state_dict.keys():
#             if any(module in key for module in target_modules):
#                 client_state_dict[key] = global_state_dict[key].clone()
        
#         client_model.load_state_dict(client_state_dict)
    
#     return client_models

# def save_federated_model(global_model, round_num, save_dir, args):
#     """保存联邦学习的全局模型"""
#     os.makedirs(save_dir, exist_ok=True)
#     model_name = f"federated_global_model_round{round_num}_{args.dataset}_target{args.target}_seed{args.seed}.pt"
#     model_path = os.path.join(save_dir, model_name)
    
#     torch.save({
#         'model_state_dict': global_model.state_dict(),
#         'round': round_num,
#         'args': args
#     }, model_path)
    
#     print(f"Global model saved at round {round_num}: {model_path}")

# parser = argparse.ArgumentParser()

# ######################## Model parameters ########################
# home_dir = os.getcwd()

# parser.add_argument('--seed', default=1, type=int, help='seed value')
# parser.add_argument('--dataset', default='pamap', type=str, help='Dataset of choice: pamap, uschad, dsads')
# parser.add_argument('--target', default=1,type=int, help='Choose task id')
# parser.add_argument('--remain_rate', default=0.2,type=float, help='Using training data ranging from 0.2 to 1.0')
# parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
# parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
# parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
# parser.add_argument('--batch_size', default=64,type=int, help='Training batch')
# parser.add_argument('--Ocomb', default=5,type=int, help='the maxmium of combination')
# parser.add_argument('--Ktimes', default=2,type=int, help='the ratio between new and ori')
# parser.add_argument('--lr_decay_cls_f', default=1e-1,type=float, help='step-3-fea')
# parser.add_argument('--lr_decay_cls', default=1e-1,type=float, help='step-3-net')
# parser.add_argument('--lr_decay_ori', default=1e-1,type=float, help='step-2-net')
# parser.add_argument('--lr_decay_ori_f', default=1e-1,type=float, help='step-2-fea')
# parser.add_argument('--lr_decay1', default=1e-2,type=float, help='step-1-fea')
# parser.add_argument('--lr_decay2', default=1,type=float, help='step-1-net')
# parser.add_argument('--lr', type=float, default=7e-3, help="learning rate")
# parser.add_argument('--step1', default=0,type=int, help='epoch1')
# parser.add_argument('--step2', default=0,type=int, help='epoch2')
# parser.add_argument('--step3', default=1,type=int, help='epoch3')

# # 联邦学习特定参数
# parser.add_argument('--num_clients', default=3, type=int, help='total number of clients')
# parser.add_argument('--num_rounds', default=10, type=int, help='number of federated rounds')
# parser.add_argument('--local_epochs', default=5, type=int, help='local training epochs per round')
# parser.add_argument('--federated_save_dir', default='./federated_models/', type=str, help='directory to save federated models')

# args = parser.parse_args()

# # Some key info
# target = args.target
# data_type = args.dataset
# remain_rate = args.remain_rate
# batch_size = args.batch_size

# set_random_seed(args.seed)

# print("=" * 60)
# print("开始联邦学习训练")
# print(f"数据集: {data_type}")
# print(f"目标域: {target}")
# print(f"客户端数量: {args.num_clients}")
# print(f"联邦轮数: {args.num_rounds}")
# print(f"本地训练轮数: {args.local_epochs}")
# print("=" * 60)

# # 初始化客户端模型列表
# client_models = []
# client_testusers = []
# client_diffusions = []

# # 为每个客户端初始化数据和模型
# for client_id in range(1, args.num_clients + 1):
#     print(f"\n初始化客户端 {client_id}...")
    
#     # 设置客户端特定信息
#     testuser = {}
#     testuser['target'] = target
#     testuser['maxcond'] = args.Ocomb
#     testuser['cond_weight'] = cond_set(data_type, remain_rate, target, testuser['maxcond'])
#     testuser['dataset'] = data_type
#     testuser['repeat'] = args.Ktimes
#     testuser['seed'] = args.seed
#     testuser['client_id'] = client_id
#     testuser['num_clients'] = args.num_clients
#     testuser['enable_cross_client_generation'] = False  # 简化版本先关闭跨客户端生成
#     testuser['remain_data'] = args.remain_rate
    
#     # 客户端特定的名称
#     testuser['name'] = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])+f'_client{client_id}'
    
#     # 加载客户端数据
#     train_loader, valid_loader, target_loader, testuser['n_class'] = get_acthar(
#         args, data_type, target, batch_size=64, remain_rate=testuser['remain_data'], 
#         seed=testuser['seed'], client_id=client_id
#     )
    
#     # 设置路径
#     conditioner_base = os.getcwd()+os.path.join('/Style_conditioner/conditioner_pth/')
#     testuser['conditioner'] = conditioner_base + data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])+f'_client{client_id}'+'-'+f'ckp_last-dl.pt'
#     testuser['newdata'] = os.getcwd()+'/Featurenet/new_data/' +testuser['name']+'-rep'+str(testuser['repeat'])+'-batch'+str(batch_size)+'-cond'+str(testuser['maxcond']) +'-weight'+(str(testuser['cond_weight']))+'.pt'
#     testuser['diff'] = os.getcwd()+'/Diffusion_model/dm_pth/'+data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])+f'_client{client_id}'+'.pt'
    
#     # 初始化扩散模型
#     train_dataset = train_loader.dataset
#     source_loaders = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    
#     for minibatch in source_loaders:
#         batch_size_actual = minibatch[0].shape[0]
#         shape = [minibatch[0].shape[1], minibatch[0].shape[2]]
#         break
    
#     testuser['length'] = shape[0]
#     if shape[0] % 64 != 0:
#         shape[0] = 64 - (shape[0] % 64) + shape[0]
    
#     model_our = Unet1D_cond(
#         dim=64,
#         num_classes=100,
#         dim_mults=(1, 2, 4, 8),
#         channels=shape[1],
#         context_using=True
#     )
    
#     diffusion = GaussianDiffusion1Dcond(
#         model_our,
#         seq_length=shape[0],
#         timesteps=100,
#         objective='pred_noise'
#     )
#     diffusion = diffusion.to(device)
    
#     # 加载预训练的扩散模型
#     if os.path.exists(testuser['diff']):
#         data_dict = torch.load(testuser['diff'])
#         diffusion.load_state_dict(data_dict['model'])
    
#     # 初始化算法模型
#     dataset = testuser['name'].split('_tar')[0]
#     args_data = get_args(dataset, args)
#     algorithm_class = alg.get_algorithm_class()
#     algorithm = algorithm_class(args_data).cuda()
    
#     # 存储客户端信息
#     client_models.append(algorithm)
#     client_testusers.append(testuser)
#     client_diffusions.append(diffusion)
    
#     print(f"客户端 {client_id} 初始化完成")

# # 联邦学习主循环
# global_model = None

# for round_num in range(args.num_rounds):
#     print(f"\n{'='*50}")
#     print(f"联邦学习轮次 {round_num + 1}/{args.num_rounds}")
#     print(f"{'='*50}")
    
#     # 1. 分发全局模型到各客户端
#     if global_model is not None:
#         print("分发全局模型到各客户端...")
#         client_models = distribute_model(global_model, client_models)
    
#     # 2. 各客户端本地训练
#     trained_client_models = []
    
#     for client_id in range(args.num_clients):
#         print(f"\n客户端 {client_id + 1} 开始本地训练...")
        
#         # 获取客户端数据
#         testuser = client_testusers[client_id]
#         diffusion = client_diffusions[client_id]
#         algorithm = client_models[client_id]
        
#         # 重新加载数据
#         train_loader, valid_loader, target_loader, _ = get_acthar(
#             args, data_type, target, batch_size=64, remain_rate=testuser['remain_data'], 
#             seed=testuser['seed'], client_id=client_id + 1
#         )
        
#         train_dataset = train_loader.dataset
#         source_loaders = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        
#         # 本地训练（修改train_diversity以支持有限轮数训练）
#         print(f"客户端 {client_id + 1} 训练 {args.local_epochs} 个epoch...")
        
#         # 这里我们调用修改后的训练函数，限制训练轮数
#         # 注意：需要修改train_diversity函数以支持限制训练轮数
#         train_diversity(diffusion, args_data, source_loaders, valid_loader, target_loader, testuser, max_epochs=args.local_epochs)
        
#         trained_client_models.append(algorithm)
#         print(f"客户端 {client_id + 1} 本地训练完成")
    
#     # 3. 服务器聚合
#     print(f"\n服务器聚合模型...")
#     global_model = federated_average(trained_client_models)
    
#     # 4. 保存全局模型
#     save_federated_model(global_model, round_num + 1, args.federated_save_dir, args)
    
#     # 5. 更新客户端模型引用
#     client_models = [deepcopy(global_model) for _ in range(args.num_clients)]
    
#     print(f"联邦学习轮次 {round_num + 1} 完成")

# print("\n" + "="*60)
# print("联邦学习训练完成！")
# print(f"最终全局模型保存在: {args.federated_save_dir}")
# print("="*60)
################################################
# import torch

# import sys
# sys.path.append('.')
# sys.path.append('./Diffusion_model/')
# from denoising_diffusion_pytorch import Unet1D_cond,GaussianDiffusion1Dcond
# from data_load.get_domainhar import get_acthar
# from data_load.get_domainhar import get_acthar
# import torch.nn as nn
# import torch.utils.data as data
# from train_strategy import train_diversity
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# from Featurenet.utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
# import os
# import argparse
# import numpy as np
# from Featurenet.config_files.distrb_condition import cond_set

# def set_random_seed(SEED):
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# parser = argparse.ArgumentParser()

# ######################## Model parameters ########################
# home_dir = os.getcwd()

# parser.add_argument('--seed', default=1, type=int,
#                     help='seed value')
# parser.add_argument('--dataset', default='pamap', type=str,
#                     help='Dataset of choice: pamap, uschad, dsads')
# parser.add_argument('--target', default=1,type=int,
#                     help='Choose task id')
# parser.add_argument('--remain_rate', default=0.2,type=float,
#                     help='Using training data ranging from 0.2 to 1.0')
# parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
#                     help='saving directory')
# parser.add_argument('--device', default='cuda', type=str,
#                     help='cpu or cuda')
# parser.add_argument('--home_path', default=home_dir, type=str,
#                     help='Project home directory')
# parser.add_argument('--batch_size', default=64,type=int,
#                     help='Training batch')
# parser.add_argument('--Ocomb', default=5,type=int,
#                     help='the maxmium of combination')

# parser.add_argument('--Ktimes', default=2,type=int,
#                     help='the ratio between new and ori')
# parser.add_argument('--lr_decay_cls_f', default=1e-1,type=float,
#                     help='step-3-fea')
# parser.add_argument('--lr_decay_cls', default=1e-1,type=float,
#                     help='step-3-net')
# parser.add_argument('--lr_decay_ori', default=1e-1,type=float,
#                     help='step-2-net')
# parser.add_argument('--lr_decay_ori_f', default=1e-1,type=float, #1e-1
#                     help='step-2-fea')
# parser.add_argument('--lr_decay1', default=1e-2,type=float, #1e-2
#                     help='step-1-fea')
# parser.add_argument('--lr_decay2', default=1,type=float,
#                     help='step-1-net')
# parser.add_argument('--lr', type=float, default=7e-3, help="learning rate")
# parser.add_argument('--step1', default=0,type=int,
#                     help='epoch1')
# parser.add_argument('--step2', default=0,type=int,
#                     help='epoch2')
# parser.add_argument('--step3', default=1,type=int,
#                     help='epoch3')

# args = parser.parse_args()
   


# #Some key info
# target = args.target
# data_type  = args.dataset
# remain_rate =args.remain_rate
# testuser = {}
# batch_size = args.batch_size
# testuser['target'] = target
# testuser['maxcond'] =args.Ocomb
# testuser['cond_weight'] = cond_set(data_type, remain_rate, target, testuser['maxcond'])
# testuser['dataset'] = data_type
# testuser['repeat'] = args.Ktimes #repeat diverse samples for k times 
# testuser['seed']  = args.seed
# set_random_seed(testuser['seed'])
# testuser['remain_data'] = args.remain_rate
# print("Remain:",testuser['remain_data'])
# testuser['name'] = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])
# dataset =testuser['name'].split('_tar')[0] 
# args_data = get_args(dataset,args)


# # for attr, value in vars(args_data).items():
# #     setattr(args, attr, value)

# conditioner = os.getcwd()+os.path.join('/Style_conditioner/conditioner_pth/')

# # Load data
# train_loader, valid_loader, target_loader,testuser['n_class'] = get_acthar(args,data_type ,target , batch_size =64,remain_rate = testuser['remain_data'], seed = testuser['seed'])
# train_dataset = train_loader.dataset
# valid_dataset = valid_loader.dataset
# source_loaders= data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
# valid_loader= data.DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
# testuser['conditioner'] =  conditioner + data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(testuser['seed'])+'-'+f'ckp_last-dl.pt'

# # Load style conditioner, diffusion and newdata_path

# testuser['newdata']  = os.getcwd()+'/Featurenet/new_data/' +testuser['name']+'-rep'+str(testuser['repeat'])+'-batch'+str(batch_size)+'-cond'+str(testuser['maxcond']) +'-weight'+(str(testuser['cond_weight']))+'.pt'
# testuser['diff'] = os.getcwd()+'/Diffusion_model/dm_pth/'+testuser['name']+'.pt'

# for minibatch in source_loaders:
#     batch_size = minibatch[0].shape[0]
#     print("print shape X:",minibatch[0].shape)
#     shape=[minibatch[0].shape[1],minibatch[0].shape[2]]#length,channel
#     break

# testuser['length'] = shape[0]
# if shape[0] % 64 !=0:
#     shape[0] =  64 - (shape[0] % 64) +shape[0]


# model_our = Unet1D_cond(    
#     dim = 64,
#     num_classes = 100, 
#     dim_mults = (1, 2, 4, 8),
#     channels = shape[1],
#     context_using = True
# )

# diffusion = GaussianDiffusion1Dcond(
#     model_our ,
#     seq_length = shape[0],
#     timesteps = 100,  
#     objective = 'pred_noise'
# )

# diffusion= diffusion.to(device)
# diffusion= diffusion.to(device)
# data = torch.load(testuser['diff'])
# diffusion.load_state_dict(data['model'])
# criterion = nn.CrossEntropyLoss()


# train_diversity(diffusion,args,source_loaders, valid_loader, target_loader,testuser) #fine_tune using generative samples
# print(testuser['name'])
# print(testuser['newdata'])

