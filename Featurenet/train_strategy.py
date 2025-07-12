import time
from alg.opt import *
from alg import alg, modelopera
import torch.nn.functional as F
import sys
sys.path.append('.')
from Featurenet.utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
import random
import os
import torch
import sys
from data_load.get_domainhar import get_acthar, get_acthar_client
from torch import optim
import torch.nn as nn
from torch.nn.functional import cosine_similarity
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
import random
import logging
from logging import handlers
from copy import deepcopy
import sys
sys.path.append('./data_load/')
from data_load.data_util.sensor_loader import SensorDataset,DataDataset
from torch.utils.data import DataLoader
import torch.utils.data as data
import pickle

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str) 
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)

from datetime import datetime
from itertools import combinations
import math

def federated_average(models):
    """
    联邦平均算法：对多个模型参数进行平均
    """
    global_dict = {}
    
    for key in models[0].state_dict().keys():
        param_stack = torch.stack([model.state_dict()[key] for model in models], 0)
        
        # 检查参数类型，只对浮点型参数求平均，整数型参数直接使用第一个模型的值
        if param_stack.dtype.is_floating_point or param_stack.dtype.is_complex:
            global_dict[key] = param_stack.mean(0)
        else:
            # 对于整数类型参数（如batch norm的running count），使用第一个模型的值
            global_dict[key] = models[0].state_dict()[key]
    
    return global_dict



def train_client_local(client_id, model, args, client_data, valid_loader, testuser, logger, local_epochs=50):
    """
    单个客户端本地训练 
    """
    logger.debug(f"Starting local training for client {client_id}")
    
    # 为每个客户端创建独立的优化器
    algorithm = deepcopy(model)
    opto = get_optimizer(algorithm, args, nettype='step-2')
    optc = get_optimizer(algorithm, args, nettype='step-3')
    optf = get_optimizer(algorithm, args, nettype='step-1') 
    schedulera, schedulerd, scheduler, use_slr = get_slr(testuser['dataset'], testuser['target'], optf, opto, optc)
    
    # 使用传入的客户端数据
    client_train_loader = client_data
    
    # 根据错误日志，设置最大类别数（看到y范围是0-18，所以设为19）
    max_classes = 19
    
    # 客户端本地训练
    for epoch in range(local_epochs):
        algorithm.train()
        
        if epoch % 10 == 0:
            logger.debug(f"Client {client_id} - Local epoch {epoch}")
        #  Class spec - 使用x,y,d三个参数
        for step in range(args.step3):
            successful_batches = 0
            for batch_no, minibatch in enumerate(client_train_loader, start=1):
                try:
                    x = minibatch[0]  # 输入数据
                    y = minibatch[1]  # 标签  
                    d = minibatch[2]  # 域标签
                    
                    # 检查数据完整性
                    if x.numel() == 0 or y.numel() == 0 or d.numel() == 0:
                        continue
                    
                    # 转移到设备
                    train_x = x.to(device)
                    train_y = y.to(device)
                    train_d = d.to(device)
                    
                    # 修复维度处理 - 确保数据shape正确
                    if len(train_x.shape) == 3:
                        train_x = train_x.unsqueeze(3)
                    train_x = train_x.transpose(1, 2).squeeze(3).unsqueeze(2)
                    
                    # 确保长度匹配
                    if train_x.shape[1] > testuser['length']:
                        train_x = train_x[:,:testuser['length'],:,:]
                    elif train_x.shape[1] < testuser['length']:
                        # 如果长度不足，进行padding
                        padding_size = testuser['length'] - train_x.shape[1]
                        train_x = F.pad(train_x, (0, 0, 0, 0, 0, padding_size), 'constant', 0)
                    
                    # 最终确保维度正确：[batch_size, length, 1, channels] -> [batch_size, length, channels]
                    if train_x.dim() == 4 and train_x.shape[2] == 1:
                        train_x = train_x.squeeze(2)
                  
                    # 转换数据类型
                    train_y = train_y.long()
                    train_d = train_d.long()
                    
                    # 更新模型 - 传入x,y参数（根据原代码，这里只需要x,y）
                    loss_list, y_pred, index_worse, all_z = algorithm.update_cs(train_x, train_y, optc)
                    successful_batches += 1
                    
                except Exception as e:
                    logger.error(f"Client {client_id} Step3 batch {batch_no} error: {e}")
                    continue
            
            # logger.debug(f"Client {client_id} Step3 completed {successful_batches}/{len(client_train_loader)} batches successfully")
            if use_slr:
                scheduler.step()
            else:
                if 'loss_list' in locals() and successful_batches > 0:
                    scheduler.step(loss_list['total'])
    
    logger.debug(f"Client {client_id} local training completed")
    return algorithm

def train_diversity( args, train_loader, valid_loader, test_loader, testuser, num_clients=6):
    nowtime = datetime.now()
    timename = nowtime.strftime('%d_%m_%Y_%H_%M_%S')
    log_file_name = os.getcwd()+os.path.join('/Featurenet/logs/', testuser['name']+f"_federated_logs_{nowtime.strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)

    algorithm_class = alg.get_algorithm_class()
    algorithm = algorithm_class(args).cuda()
    opto = get_optimizer(algorithm, args, nettype='step-2')
    optc = get_optimizer(algorithm, args, nettype='step-3')
    optf = get_optimizer(algorithm, args, nettype='step-1') 
    schedulera, schedulerd, scheduler, use_slr = get_slr(testuser['dataset'], testuser['target'], optf, opto, optc)
    
    # 定义client_train_loaders字典
    client_train_loaders = {}
    
    # 加载全局数据文件用于获取客户端数据
    data_file_path = f"/home/SHIH0020/robustlearn/datasets/FL_{testuser['dataset']}/{testuser['dataset']}_crosssubject_rawaug_rate{testuser['remain_data']}_t{testuser['target']}_seed{testuser['seed']}_scalerminmax.pkl"
    try:
        with open(data_file_path, 'rb') as f:
            global_data = pickle.load(f)
        logger.debug(f"Loaded global data from {data_file_path}")
    except Exception as e:
        logger.warning(f"Could not load global data: {e}")
        global_data = {}
    
    # 为每个客户端加载数据
    for client_id in range(1, num_clients + 1):
        logger.debug(f"Processing client {client_id}")
        
        # 检查是否有客户端特定的数据
        if 'client_raw_trs' in global_data and client_id in global_data['client_raw_trs']:
            try:
                client_raw_x = global_data['client_raw_trs'][client_id][0]  # shape: (542, 125, 45)
                client_raw_y = global_data['client_raw_trs'][client_id][1]  # shape: (542,)
                data_type = args.dataset
                target = args.target
                batch_size = 64  # 定义batch_size
                
                client_train_loader, client_valid_loader, target_loader, n_class = get_acthar_client(
                    args, data_type, target, batch_size=batch_size, 
                    remain_rate=testuser['remain_data'], seed=testuser['seed'], client_id=client_id
                )
                
                train_dataset = client_train_loader.dataset
                valid_dataset = client_valid_loader.dataset
                source_loaders = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
                
                # 正确赋值client_train_loader
                client_train_loaders[client_id] = source_loaders
                
                logger.debug(f"Client {client_id} loaded {len(train_dataset)} training samples")
                
            except Exception as e:
                logger.warning(f"Error loading client {client_id} data: {e}, using fallback")
                # Fallback: use original train_loader if loading fails
                client_train_loaders[client_id] = train_loader
        else:
            # 如果没有客户端特定数据，使用原始train_loader
            logger.debug(f"No specific data for client {client_id}, using default train_loader")
            client_train_loaders[client_id] = train_loader
        
        logger.debug(f"Client {client_id} data processing completed")
    
    # 初始化全局模型
    algorithm_class = alg.get_algorithm_class()
    global_model = algorithm_class(args).cuda()
    
    # 联邦学习参数
    aggregation_rounds = 10
    local_epochs = 50
    
    logger.debug(f"Starting federated learning with {num_clients} clients, {aggregation_rounds} rounds, {local_epochs} local epochs")
    
    best_test_acc = 0
    
    # 联邦学习主循环
    for round_idx in range(aggregation_rounds):
        logger.debug(f"\n========= Federated Round {round_idx + 1}/{aggregation_rounds} =========")
        
        # 存储客户端模型
        client_models = []
        
        # 每个客户端本地训练
        for client_id in range(1, num_clients + 1):
            logger.debug(f"Training client {client_id}")
            
            try:
                # 客户端从全局模型开始训练，使用对应客户端的数据
                client_model = train_client_local(
                    client_id=client_id,
                    model=global_model,
                    args=args,
                    client_data=client_train_loaders[client_id],  
                    valid_loader=valid_loader,
                    testuser=testuser,
                    logger=logger,
                    local_epochs=local_epochs
                )
                
                client_models.append(client_model)
                logger.debug(f"Client {client_id} training completed successfully")
                
            except Exception as e:
                logger.error(f"Error training client {client_id}: {e}")
                logger.error("Skipping this client for this round")
                continue
        
        if not client_models:
            logger.error("No client models trained successfully. Stopping federated learning.")
            break
            
        # 联邦平均聚合
        logger.debug("Aggregating client models...")
        try:
            global_dict = federated_average(client_models)
            global_model.load_state_dict(global_dict)
            logger.debug("Model aggregation completed successfully")
        except Exception as e:
            logger.error(f"Error during model aggregation: {e}")
            continue
        
        # 在测试集上评估全局模型 - 使用x,y,d三个参数，d设为全0
        acc = 0
        num = 0
        global_model.eval()
        
        try:
            for batch_no, minibatch in enumerate(test_loader, start=1):
                try:
                    x = minibatch[0]  # 输入数据
                    y = minibatch[1]  # 标签
                    d = minibatch[2]  # 域标签
                    
                    # 将d的所有值设为0（按用户要求）
                    d = torch.zeros_like(d)
                    
                    # 转移到设备并转换数据类型
                    x, y, d = x.to(device), y.long().to(device), d.long().to(device)
                    
                    
                    # 维度处理 - 与训练时保持一致
                    if len(x.shape) == 3:
                        x = x.unsqueeze(3)

                    x = x.transpose(1, 2).squeeze(3).unsqueeze(2)
                    
                    # 确保长度匹配
                    if x.shape[1] > testuser['length']:
                        x = x[:,:testuser['length'],:,:]
                    elif x.shape[1] < testuser['length']:
                        padding_size = testuser['length'] - x.shape[1]
                        x = F.pad(x, (0, 0, 0, 0, 0, padding_size), 'constant', 0)
                    
                    # 最终确保维度正确
                    if x.dim() == 4 and x.shape[2] == 1:
                        x = x.squeeze(2)
                    
                    y_pred_list = [] 
                    y_pred, target_z = global_model.predict(x.float())
                    y_prob = F.softmax(y_pred, dim=1) 
                    pred1 = y_prob
                    y_pred_list.append(y_prob.cpu().detach().numpy())
                    y_pred_list = np.array(y_pred_list)
                    class_score = np.sum(y_pred_list, axis=0)
                    y_pred = np.argmax(class_score, axis=1)
                    y_true = y.cpu().detach().numpy()
                    acc += np.sum(y_pred == y_true)
                    num += len(y_pred)
                    
                except Exception as e:
                    logger.warning(f"Error in test batch {batch_no}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            test_acc = 0
        else:
            test_acc = acc/num if num > 0 else 0
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            
        logger.debug(f"Round {round_idx + 1} - Test Accuracy: {test_acc:.5f}, Best Accuracy: {best_test_acc:.5f}")
        
    logger.debug(f"\nFederated learning completed. Final best test accuracy: {best_test_acc:.5f}")
    logger.debug(f"Final results saved for: {testuser['name']}")
    
    return best_test_acc
