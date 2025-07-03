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
from Style_conditioner.get_conditioner import conditioner
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
from data_util.sensor_loader import SensorDataset,DataDataset
from torch.utils.data import DataLoader
import torch.utils.data as data

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
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

def combine_label(y,logger, maxlen, pre_defined_weights, weighted = True):
    index_dict = {}
    for i, label in enumerate(y):
        label = label.item()
        if label not in index_dict:
            index_dict[label] = [i]
        else:
            index_dict[label].append(i)
    combination_dict = {}
    weight_dict = {}

    for label, indices in index_dict.items():
        all_combinations = [] 
        all_weights = []
        all_weights_no_repeat = []

        if maxlen == None:
            sum_of_all_cond_num_weights = sum([pre_defined_weights[i] for i in range(len(indices))])
            for r in range(1, len(indices) + 1):
                all_combinations.extend(combinations(indices, r))
                if weighted: 
                    cond_num_weight = pre_defined_weights[r-1] / sum_of_all_cond_num_weights
                    all_weights.extend([cond_num_weight/math.comb(len(indices), r)] * int(math.comb(len(indices), r)))
        else:
            if maxlen > len(indices): maxlen = len(indices)
            sum_of_all_cond_num_weights = sum([pre_defined_weights[i] for i in range(maxlen)])
            for r in range(1,  maxlen + 1):
                all_combinations.extend(combinations(indices, r))
                if weighted: 
                    cond_num_weight = pre_defined_weights[r-1] / sum_of_all_cond_num_weights
                    all_weights.extend([cond_num_weight/math.comb(len(indices), r)] * int(math.comb(len(indices), r)))

        if weighted: 
            weight_dict[label] = all_weights
        combination_dict[label] = all_combinations
    return combination_dict, weight_dict

def sample_subbatch(train_x, train_y,train_d,candidate_indices):
    train_y_shape = train_y.shape
    num_indices = int(train_y_shape[0]/3*2) 
    selected_indices = []
    indices = random.sample(candidate_indices, num_indices)
    for index in indices:
        candidate_indices.remove(index)

    selected_indices = torch.tensor(indices)
    train_x_selected = train_x[selected_indices]
    train_y_selected = train_y[selected_indices]
    train_d_selected = train_d[selected_indices]
    return train_x_selected,train_y_selected,train_d_selected,candidate_indices

def generate_cross_client_styles(x, y, testuser):
    """
    Generate styles using cross-client style conditioners
    """
    cross_client_styles = []
    
    # Check if cross-client style conditioners are available
    if 'cross_client_conditioners' not in testuser or not testuser.get('enable_cross_client_generation', False):
        return None
    
    for conditioner_path in testuser['cross_client_conditioners']:
        if os.path.exists(conditioner_path):
            try:
                # Create a temporary testuser for the other client
                temp_testuser = testuser.copy()
                temp_testuser['conditioner'] = conditioner_path
                
                # Generate styles using the other client's style conditioner
                styles = conditioner(x, y, temp_testuser)
                cross_client_styles.append(styles)
            except Exception as e:
                print(f"Warning: Failed to load cross-client conditioner {conditioner_path}: {e}")
                continue
        else:
            print(f"Warning: Cross-client conditioner not found: {conditioner_path}")
    
    return cross_client_styles if cross_client_styles else None

def generate_cross_client_synthetic_data(model, x, y, testuser, logger):
    """
    Generate synthetic data using aggregated cross-client styles
    """
    if not testuser.get('enable_cross_client_generation', False):
        return None
    
    # Generate cross-client styles
    cross_client_styles = generate_cross_client_styles(x, y, testuser)
    
    if cross_client_styles is None or len(cross_client_styles) == 0:
        print("No cross-client styles available, skipping cross-client generation")
        return None
    
    # Aggregate styles from different clients (simple averaging)
    aggregated_styles = torch.stack(cross_client_styles).mean(dim=0)
    
    # Repeat for multiple samples
    aggregated_styles = aggregated_styles.repeat(testuser['repeat'], 1)
    
    # Generate combination dictionary for cross-client generation
    combination_dict, weights_dict = combine_label(y, logger, testuser['maxcond'], testuser['cond_weight'])
    
    batch_size = y.shape[0] * testuser['repeat']
    random_combinations = []
    keys_tensor = []

    # Calculate the number of samples within each key
    samples_per_key = batch_size // len(combination_dict)
    
    for key in combination_dict.keys():
        values = combination_dict[key]
        random.shuffle(values)
        sampled_values = random.sample(values, min(samples_per_key, len(values)))
        
        for value in sampled_values:
            if value not in random_combinations:
                random_combinations.append(value)
                keys_tensor.append(key)

    # Fill remaining samples if needed
    times_sample = 0
    while len(random_combinations) < batch_size:
        times_sample += 1
        if times_sample > 1500:
            random_value = random.randint(0, len(random_combinations) - 1)
            random_combinations.extend([random_combinations[random_value]])
            keys_tensor.append(keys_tensor[random_value])
        else:
            for key, values in combination_dict.items():
                sampled_values = random.sample(values, 1)
                if sampled_values[0] not in random_combinations:
                    random_combinations.extend(sampled_values)
                    keys_tensor.append(key)
                if len(random_combinations) == batch_size:
                    break

    # Generate synthetic data using diffusion model
    model.eval()
    with torch.no_grad():
        cross_client_synthetic = model.sample(aggregated_styles, random_combinations, rescaled_phi=0.7)
    
    # Handle padding if necessary
    if hasattr(testuser, 'length') and 'length' in testuser:
        try:
            cross_client_synthetic = cross_client_synthetic[:,:,:, -testuser['length']:]
        except:
            pass
    
    return {
        'x': cross_client_synthetic.unsqueeze(2),
        'y': torch.tensor(keys_tensor).to(device),
        'd': torch.zeros(len(keys_tensor)).to(device)  # Mark as synthetic data
    }

def train_diversity(model, args, train_loader, valid_loader, test_loader, testuser, max_epochs=120):
    """
    修改后的训练函数，支持限制最大训练轮数（用于联邦学习）
    """
    nowtime = datetime.now()
    timename = nowtime.strftime('%d_%m_%Y_%H_%M_%S')
    log_file_name = os.getcwd()+os.path.join('/Featurenet/logs/', testuser['name']+f"logs_{nowtime.strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)

    algorithm_class = alg.get_algorithm_class()
    algorithm = algorithm_class(args).cuda()
    opto = get_optimizer(algorithm, args, nettype='step-2')
    optc = get_optimizer(algorithm, args, nettype='step-3')
    optf = get_optimizer(algorithm, args, nettype='step-1') 
    schedulera, schedulerd, scheduler, use_slr = get_slr(testuser['dataset'], testuser['target'], optf, opto, optc)
    diff_sample = {}
    cross_client_sample = {}
    
    file_pathv = testuser['newdata']
    cross_client_file_path = testuser.get('cross_client_newdata', None)
    
    # Generate synthetic data (same as original)
    if os.path.exists(file_pathv):
        diff_sample = torch.load(file_pathv)  
    else:
        k_data = -1
        for batch_no, minibatch in enumerate(train_loader, start=1):
            k_data = k_data + 1
            x = minibatch[0]
            y = minibatch[1]
            x, y = x.to(device), y.long().to(device)
            length = x.shape[1]
            remainder = 0
            if x.size(1) % 64 != 0:
                remainder = 64 - (x.size(1) % 64)
                pad_x = F.pad(x, (0, 0, 0, remainder, 0, 0))
            if len(x.shape) == 3:
                x = x.unsqueeze(3)
            x = x.transpose(1, 2).squeeze(3).unsqueeze(2)
            
            if k_data in diff_sample.keys():
                pass
            else:
                # Generate original synthetic data
                styles = conditioner(x, y, testuser)
                styles = styles.repeat(testuser['repeat'], 1)
                try:
                    pad_x
                    if len(pad_x.shape) == 3:
                        pad_x = pad_x.unsqueeze(3)
                    x = pad_x.transpose(1, 2).squeeze(3).unsqueeze(2)
                    x_aug = x.repeat(testuser['repeat'], 1, 1, 1)
                except:
                    x_aug = x.repeat(testuser['repeat'], 1, 1, 1)
                    pass
                    
                x_ = x_aug.squeeze(2).float()
                combination_dict, weights_dict = combine_label(y, logger, testuser['maxcond'], testuser['cond_weight'])
                batch_size = y.shape[0] * testuser['repeat']
                random_combinations = []
                keys_tensor = []

                # Calculate the number of samples within each key
                samples_per_key = batch_size // len(combination_dict)
                remaining_samples = batch_size % len(combination_dict)

                for key in combination_dict.keys():
                    values = combination_dict[key]
                    weights = weights_dict[key]
                    random.shuffle(values)
                    sampled_values = random.sample(values, min(samples_per_key, len(values)))
                    for value in sampled_values:
                        if value not in random_combinations:
                            random_combinations.append(value)
                            keys_tensor.append(key)

                times_sample = 0
                while len(random_combinations) < batch_size:
                    times_sample = times_sample + 1
                    if times_sample > 1500:
                        random_value = random.randint(0, len(random_combinations) - 1)
                        random_combinations.extend([random_combinations[random_value]])
                        keys_tensor.append(keys_tensor[random_value])
                    else:
                        for key, values in combination_dict.items():
                            sampled_values = random.sample(values, 1)
                            if sampled_values[0] not in random_combinations:
                                random_combinations.extend(sampled_values)
                                keys_tensor.append(key)
                            if len(random_combinations) == batch_size:
                                break

                model.eval()
                interpolate_out = model.sample(styles, random_combinations, rescaled_phi=0.7)
                try:
                    pad_x
                    interpolate_out = interpolate_out[:,:,:, -testuser['length']:]
                except:
                    pass

                diff_sample[k_data] = {}
                
                diff_sample[k_data]['x'] = x.float()
                diff_sample[k_data]['y'] = torch.tensor(keys_tensor).to(device)
                shape = y.shape
                diff_sample[k_data]['d'] = torch.zeros(shape)
                domain_i = 1
                diff_sample[k_data]['x'] = torch.cat([interpolate_out.unsqueeze(2), diff_sample[k_data]['x']], dim=0)
                diff_sample[k_data]['y'] = torch.cat([diff_sample[k_data]['y'], y], dim=0)
                zeros_tensor = torch.zeros(shape)
                diff_sample[k_data]['d'] = diff_sample[k_data]['d'].repeat(testuser['repeat'])
                diff_sample[k_data]['d'] = torch.cat([diff_sample[k_data]['d'], zeros_tensor + domain_i], dim=0)
                train_y = diff_sample[k_data]['y']
                diff_sample[k_data]['d'] = torch.zeros(diff_sample[k_data]['d'].shape).to(device)
                chose_len = int(train_y.shape[0]/(testuser['repeat']+1)) 
                diff_sample[k_data]['d'][-chose_len:] = diff_sample[k_data]['d'][-chose_len:] + 1   
                        
                torch.save(diff_sample, file_pathv)

    # Generate cross-client synthetic data if enabled
    if testuser.get('enable_cross_client_generation', False) and cross_client_file_path:
        if os.path.exists(cross_client_file_path):
            cross_client_sample = torch.load(cross_client_file_path)
        else:
            k_data = -1
            for batch_no, minibatch in enumerate(train_loader, start=1):
                k_data = k_data + 1
                x = minibatch[0]
                y = minibatch[1]
                x, y = x.to(device), y.long().to(device)
                
                # Prepare data
                if x.size(1) % 64 != 0:
                    remainder = 64 - (x.size(1) % 64)
                    pad_x = F.pad(x, (0, 0, 0, remainder, 0, 0))
                if len(x.shape) == 3:
                    x = x.unsqueeze(3)
                x = x.transpose(1, 2).squeeze(3).unsqueeze(2)
                
                if k_data not in cross_client_sample.keys():
                    # Generate cross-client synthetic data
                    cross_client_data = generate_cross_client_synthetic_data(model, x, y, testuser, logger)
                    
                    if cross_client_data is not None:
                        cross_client_sample[k_data] = cross_client_data
                        print(f"Generated cross-client synthetic data for batch {k_data}")
                    else:
                        print(f"Failed to generate cross-client synthetic data for batch {k_data}")
            
            if cross_client_sample:
                torch.save(cross_client_sample, cross_client_file_path)
                print(f"Cross-client synthetic data saved to {cross_client_file_path}")

    # Combine original and cross-client synthetic data
    for k_data in diff_sample.keys():
        if k_data == 0:
            data_train = diff_sample[k_data]['x']
            label_train = diff_sample[k_data]['y']
            domain_train = diff_sample[k_data]['d']
        else:
            data_train = torch.cat([data_train, diff_sample[k_data]['x']], dim=0)
            label_train = torch.cat([label_train, diff_sample[k_data]['y']], dim=0)
            domain_train = torch.cat([domain_train, diff_sample[k_data]['d']], dim=0)
    
    # Add cross-client synthetic data if available
    if cross_client_sample:
        for k_data in cross_client_sample.keys():
            data_train = torch.cat([data_train, cross_client_sample[k_data]['x']], dim=0)
            label_train = torch.cat([label_train, cross_client_sample[k_data]['y']], dim=0)
            # Mark cross-client synthetic data with domain=2
            cross_client_domain = torch.full_like(cross_client_sample[k_data]['d'], 2).to(device)
            domain_train = torch.cat([domain_train, cross_client_domain], dim=0)
        
        print(f"Added {sum([len(cross_client_sample[k]['y']) for k in cross_client_sample.keys()])} cross-client synthetic samples")

    generate_dataset = DataDataset(x=data_train.cpu(), label=label_train.cpu(), alabel=domain_train.cpu(), dataset=testuser['dataset'])
    train_loader = DataLoader(dataset=generate_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    epochs = 1
    best_acc2 = 0
    best_test_acc = 0

    new_x_data = []
    new_y_data = []
    new_d_data = []
    
    # 修改：限制训练轮数，支持联邦学习
    print(f"开始训练，最大轮数: {max_epochs}")
    
    for epoch in range(max_epochs):
        algorithm.train()
        
        # 完整的训练循环，基于原始代码逻辑
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch_idx, (data, target, domain) in enumerate(train_loader):
            data, target, domain = data.to(device), target.to(device).long(), domain.to(device).long()
            
            # 数据预处理 - 确保维度正确
            if len(data.shape) == 3:
                data = data.unsqueeze(3)  # [batch, length, channels] -> [batch, length, channels, 1]
            
            # 转换数据格式以匹配模型期望的输入
            data = data.transpose(1, 2).squeeze(3).unsqueeze(2)  # [batch, channels, 1, length]
            
            try:
                # Step 3: 分类器训练 (主要训练步骤)
                if args.step3 > 0:
                    loss_dict_cs, preds, index_list, features = algorithm.update_cs(data, target, optc)
                    
                    # 计算准确率
                    with torch.no_grad():
                        _, predicted = torch.max(preds.data, 1)
                        correct = (predicted == target).sum().item()
                        accuracy = correct / target.size(0)
                        epoch_acc += accuracy
                        epoch_loss += loss_dict_cs.get('total', 0.0)
                
                # Step 2: 域判别器训练 (如果启用)
                if args.step2 > 0:
                    # 准备域标签
                    domain_labels = domain
                    loss_dict_os = algorithm.update_os(data, target, domain_labels, opto)
                
                # Step 1: 特征学习训练 (如果启用) 
                if args.step1 > 0:
                    # 创建组合标签用于对抗训练
                    combined_labels = domain * testuser['n_class'] + target
                    loss_dict_ft = algorithm.update_ft(data, target, domain, optf)
                
                # 学习率调度
                if use_slr:
                    if args.step1 > 0:
                        schedulera.step(loss_dict_ft.get('class', 0.0))
                    if args.step2 > 0:
                        schedulerd.step(loss_dict_os.get('total', 0.0))
                    if args.step3 > 0:
                        scheduler.step(loss_dict_cs.get('total', 0.0))
                else:
                    if args.step1 > 0:
                        schedulera.step()
                    if args.step2 > 0:
                        schedulerd.step()
                    if args.step3 > 0:
                        scheduler.step()
                
                num_batches += 1
                
                # 打印训练进度
                if batch_idx % 50 == 0:
                    client_id = testuser.get('client_id', 'unknown')
                    progress = 100. * batch_idx / len(train_loader)
                    print(f'Client {client_id} Epoch: {epoch+1}/{max_epochs} '
                          f'[{batch_idx * len(data)}/{len(train_loader.dataset)} ({progress:.0f}%)] '
                          f'Loss: {epoch_loss/(num_batches+1e-8):.6f} '
                          f'Acc: {epoch_acc/(num_batches+1e-8):.4f}')
                
            except Exception as e:
                print(f"训练步骤出错 - Batch {batch_idx}: {str(e)}")
                # 如果某个batch出错，跳过但继续训练
                continue
        
        # 每个epoch结束后的统计
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        client_id = testuser.get('client_id', 'unknown')
        print(f'Client {client_id} Epoch {epoch+1}/{max_epochs} 完成 - '
              f'平均损失: {avg_loss:.6f}, 平均准确率: {avg_acc:.4f}')
        
        # 可选：每几个epoch进行验证
        if epoch % 2 == 0 and valid_loader is not None:
            algorithm.eval()
            val_acc = evaluate_model(algorithm, valid_loader, device, testuser)
            print(f'Client {client_id} Epoch {epoch+1} 验证准确率: {val_acc:.4f}')
            algorithm.train()
    
    print(f"客户端 {testuser.get('client_id', 'unknown')} 训练完成")
    return algorithm

def evaluate_model(model, data_loader, device, testuser):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target, domain in data_loader:
            data, target = data.to(device), target.to(device).long()
            
            # 数据预处理
            if len(data.shape) == 3:
                data = data.unsqueeze(3)
            data = data.transpose(1, 2).squeeze(3).unsqueeze(2)
            
            try:
                # 预测
                predictions, features = model.predict(data)
                _, predicted = torch.max(predictions.data, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
            except Exception as e:
                print(f"评估时出错: {str(e)}")
                continue
    
    accuracy = correct / max(total, 1)
    return accuracy
