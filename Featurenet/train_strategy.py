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


def load_client_styles(testuser, logger):
    """
    加载所有客户端的styles文件
    """
    
    styles_path = "/home/SHIH0020/robustlearn/client_styles/dsads_tar0_rate0.2_seed1_client_styles.pkl"
    
    if os.path.exists(styles_path):
        logger.debug(f"Loading client styles from: {styles_path}")
        with open(styles_path, 'rb') as f:
            client_styles = pickle.load(f)
        
        logger.debug("Loaded client styles information:")
        for key, value in client_styles.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"  {key}: shape={value.shape}")
        
        return client_styles
    else:
        logger.debug(f"Client styles file not found: {styles_path}")
        return None

# 从多个客户端styles中采样的函数
def sample_mixed_styles(client_styles, current_y, testuser, logger):
    """
    从多个客户端的styles中采样,混合使用
    """
    if client_styles is None:
        return None
    
    batch_size = current_y.shape[0]
    mixed_styles = []
    
    # 收集所有客户端的styles
    all_client_styles = {}
    all_client_labels = {}
    
    for key, value in client_styles.items():
        if 'style' in key:
            client_id = key.split('_')[0]
            all_client_styles[client_id] = value.to(device)
            label_key = f"{client_id}_labels"
            if label_key in client_styles:
                all_client_labels[client_id] = client_styles[label_key]
    
    logger.debug(f"Available clients for style mixing: {list(all_client_styles.keys())}")
    
    # 为当前批次的每个样本选择styles
    for i in range(batch_size):
        current_label = current_y[i].item()
        
        # 从所有客户端中随机选择一个
        available_clients = list(all_client_styles.keys())
        selected_client = random.choice(available_clients)
        
        client_style = all_client_styles[selected_client]
        client_label = all_client_labels[selected_client]
        
        # 尝试找到相同标签的样本
        same_label_indices = (client_label == current_label).nonzero(as_tuple=True)[0]
        
        if len(same_label_indices) > 0:
            selected_idx = random.choice(same_label_indices)
            selected_style = client_style[selected_idx].unsqueeze(0)
        else:
            selected_idx = random.randint(0, client_style.shape[0] - 1)
            selected_style = client_style[selected_idx].unsqueeze(0)
        
        mixed_styles.append(selected_style)
    
    mixed_styles = torch.cat(mixed_styles, dim=0)
    logger.debug(f"Generated mixed styles shape: {mixed_styles.shape}")
    
    return mixed_styles

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

def train_diversity(model, args, train_loader,valid_loader, test_loader,testuser) :
    nowtime = datetime.now()
    timename = nowtime.strftime('%d_%m_%Y_%H_%M_%S')
    log_file_name = os.getcwd()+os.path.join('/Featurenet/logs/', testuser['name']+f"logs_{nowtime.strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)

    # 新增：加载客户端styles文件
    client_styles = load_client_styles(testuser, logger)

    algorithm_class = alg.get_algorithm_class()
    algorithm = algorithm_class(args).cuda()
    opto = get_optimizer(algorithm, args, nettype='step-2')
    optc = get_optimizer(algorithm, args, nettype='step-3')
    optf = get_optimizer(algorithm, args, nettype='step-1') 
    schedulera, schedulerd, scheduler, use_slr = get_slr(testuser['dataset'], testuser['target'], optf, opto, optc)
    diff_sample = {}
    file_pathv = testuser['newdata'] 
    use_mixed = True
    if os.path.exists(file_pathv):
        diff_sample = torch.load(file_pathv)  
    else:
        k_data = -1
        for batch_no,  minibatch in enumerate(train_loader, start=1):
            k_data = k_data + 1
            x=  minibatch[0]
            y = minibatch[1]
            x, y= x.to(device), y.long().to(device)
            length = x.shape[1]
            remainder = 0
            if x.size(1) % 64 !=0:
                remainder = 64 - (x.size(1) % 64)
                pad_x = F.pad(x, (0, 0, 0, remainder, 0, 0))
            if len(x.shape) == 3:
                x = x.unsqueeze(3)
            x = x.transpose(1, 2).squeeze(3).unsqueeze(2)
            
            if k_data in diff_sample.keys():
                pass
            else:
                # 当前客户端styles和混合styles
                if client_styles is not None:
                    # 1. 使用当前客户端的styles
                    current_styles = conditioner(x, y, testuser)
                    current_styles = current_styles.repeat(testuser['repeat'], 1)
                    
                    # 2. 使用其他客户端的混合styles
                    mixed_styles = sample_mixed_styles(client_styles, y, testuser, logger)
                    if mixed_styles is not None:
                        mixed_styles = mixed_styles.repeat(testuser['repeat'], 1)
                        # 合并两种styles
                        # styles = current_styles
                        styles = torch.cat([current_styles, mixed_styles], dim=0)
                        use_mixed = True
                        logger.debug(f"Using both current and mixed client styles for batch {k_data}")
                    else:
                        styles = current_styles
                        use_mixed = False
                        logger.debug(f"Fallback to current conditioner only for batch {k_data}")
                else:
                    styles = conditioner(x, y, testuser)
                    styles = styles.repeat(testuser['repeat'], 1)
                    use_mixed = False
                    logger.debug(f"Using current conditioner for batch {k_data}")

                try:
                    pad_x
                    if len(pad_x.shape) == 3:
                        pad_x = pad_x.unsqueeze(3)
                    x = pad_x.transpose(1, 2).squeeze(3).unsqueeze(2)
                    repeat_times = testuser['repeat'] * 2 if use_mixed else testuser['repeat']
                    x_aug = x.repeat(repeat_times, 1, 1, 1)
                except:
                    repeat_times = testuser['repeat'] * 2 if use_mixed else testuser['repeat']
                    x_aug = x.repeat(repeat_times, 1, 1, 1)
                    pass
                    
                x_ = x_aug.squeeze(2).float()
                combination_dict, weights_dict = combine_label(y, logger, testuser['maxcond'], testuser['cond_weight'])
                if use_mixed:
                    batch_size = y.shape[0] * testuser['repeat'] * 2
                else:
                    batch_size = y.shape[0] * testuser['repeat']
                    
                random_combinations = []
                keys_tensor = []

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
                interpolate_out = model.sample(styles, random_combinations, rescaled_phi = 0.7)
                
                try:
                    pad_x
                    interpolate_out = interpolate_out[:,:,:, -testuser['length']:] #([256, 45, 128]
                except:
                    pass

                diff_sample[k_data] = {}
                diff_sample[k_data]['x'] = x.float()
                diff_sample[k_data]['y'] = torch.tensor(keys_tensor).to(device)
                shape = y.shape

                # 不同的domain标记
                if use_mixed:
                    # 有两种生成数据：当前客户端(d=1)和混合客户端(d=2)
                    current_samples = interpolate_out[:len(interpolate_out)//2]
                    mixed_samples = interpolate_out[len(interpolate_out)//2:]
                    
                    generated_x = torch.cat([current_samples, mixed_samples], dim=0)
                    diff_sample[k_data]['x'] = torch.cat([generated_x.unsqueeze(2), diff_sample[k_data]['x']], dim=0)
                    
                    domain_current = torch.ones(current_samples.shape[0]) * 1
                    domain_mixed = torch.ones(mixed_samples.shape[0]) 
                    domain_original = torch.zeros(shape[0])
                    
                    diff_sample[k_data]['d'] = torch.cat([domain_current, domain_mixed, domain_original], dim=0).to(device)
                    diff_sample[k_data]['y'] = torch.cat([diff_sample[k_data]['y'], y], dim=0)
                    
                    logger.debug(f"Generated {current_samples.shape[0]} current client samples (d=1) and {mixed_samples.shape[0]} mixed client samples (d=2)")
                    
                else:
                    # 只有当前客户端的数据
                    diff_sample[k_data]['x'] = torch.cat([interpolate_out.unsqueeze(2), diff_sample[k_data]['x']], dim=0)
                    diff_sample[k_data]['y'] = torch.cat([diff_sample[k_data]['y'], y], dim=0)
                    
                    diff_sample[k_data]['d'] = torch.zeros(diff_sample[k_data]['y'].shape[0]).to(device)
                    diff_sample[k_data]['d'][:len(interpolate_out)] = 1
                    diff_sample[k_data]['d'][len(interpolate_out):] = 0
                        
                torch.save(diff_sample, file_pathv)
                
    # 记录生成数据的统计信息
    logger.debug("Generated data summary:")
    total_generated_current = 0
    total_generated_mixed = 0
    total_original = 0
    for k_data in diff_sample.keys():
        batch_d = diff_sample[k_data]['d']
        current_count = (batch_d == 1).sum().item()
        mixed_count = (batch_d == 2).sum().item()
        original_count = (batch_d == 0).sum().item()
        total_generated_current += current_count
        total_generated_mixed += mixed_count
        total_original += original_count
        
    logger.debug(f"Total current client samples: {total_generated_current}")
    logger.debug(f"Total mixed client samples: {total_generated_mixed}")
    logger.debug(f"Total original samples: {total_original}")
        
    for k_data in diff_sample.keys():
        if k_data == 0:
            data_train =  diff_sample[k_data]['x']#([2710, 45, 1, 128])
            label_train = diff_sample[k_data]['y']
            domain_train = diff_sample[k_data]['d']
        else:
            data_train = torch.cat([data_train, diff_sample[k_data]['x']],dim = 0)
            label_train = torch.cat([label_train, diff_sample[k_data]['y']],dim = 0)
            domain_train = torch.cat([ domain_train, diff_sample[k_data]['d']],dim = 0)
            
    generate_dataset = DataDataset(x= data_train.cpu(), label = label_train.cpu(), alabel = domain_train.cpu(), dataset = testuser['dataset'])
    train_loader = DataLoader(dataset=generate_dataset , batch_size=args.batch_size, shuffle=True, pin_memory=True) #[2710, 45, 1, 128]
    

    epochs = 1
    
    best_acc2= 0
    best_test_acc = 0

    new_x_data = []
    new_y_data = []
    new_d_data = []
    for epoch in range(120):

        algorithm.train()
        print(f'\n========epoch {epoch}========')
        print('====1.Fine grained====')
        loss_list = ['class']    
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)
        for step in range(args.step1):
            for batch_no,  minibatch in enumerate(train_loader, start=1): #new_domain
                x=  minibatch[0]
                x = x[:,:,:,-testuser['length']:]
                y = minibatch[1]
                d = minibatch[2]
                train_x, train_y, train_d= x.to(device), y.long().to(device),d.long().to(device) #batch, len,channel,1#train_x:([64, 45, 1, 125]) train_y:([64]) train_z:[64]
                loss_result_dict = algorithm.update_ft(train_x, train_y, train_d, optf)

            print_row([step]+[loss_result_dict[item]
                            for item in loss_list], colwidth=15)
            schedulera.step(loss_result_dict['class'])
   
            logger.debug("%s", [step] + [loss_result_dict[item] for item in loss_list])


        logger.debug("====2.Ori-spec====")
        for step in range(args.step2):

            for batch_no,  minibatch in enumerate(train_loader, start=1): #new_domain
                x=  minibatch[0]
                x = x[:,:,:,-testuser['length']:]
                y = minibatch[1]
                d = minibatch[2]
                train_x, train_y, train_d= x.to(device), y.long().to(device),d.long().to(device) #batch, len,channel,1
                loss_result_dict = algorithm.update_os(train_x, train_y, train_d, opto) #classifier 
            print("Step 2 :", loss_result_dict)
           # schedulerd.step()
            schedulerd.step(loss_result_dict['total'])
        
        logger.debug("====3.Class spec====")
        for step in range(args.step3):
            k_data = -1
            num = 0
            acc = 0
            syn_accnum  = 0
            drop_log = args.drop_rate
            drop_epoch = args.drop
            for batch_no,  minibatch in enumerate(train_loader, start=1): #new_domain
                k_data = k_data + 1
                train_x =  minibatch[0].to(device) #batch,channel, 1, length [64, 45, 1, 128])
                train_y = minibatch[1].to(device)
                train_d = minibatch[2].to(device)
                train_x = train_x[:,:,:,-testuser['length']:]
                remainder = 0
                loss_list, y_pred, index_worse,all_z = algorithm.update_cs(train_x, train_y, optc)
                y_pred = y_pred.squeeze()
                y_pred = y_pred.cpu().detach().numpy()
                y_true = train_y.cpu().detach().numpy()
                #aculate acc
                y_pred = np.argmax(y_pred, axis=1)
                acc  += np.sum(y_pred == y_true)
                num += len(y_pred)  
                chose_batch = int( len(y_pred)/(testuser['repeat']+1))
                start_ori = len(y_pred) - chose_batch
                syn_accnum = syn_accnum +  np.sum(y_pred[:start_ori] == y_true[:start_ori])
                

 
            print("Step 3:", loss_list)
            if use_slr:
                scheduler.step()
            else:
                scheduler.step(loss_list['total'])
            train_acc = acc/num
            ori_acc = (acc - syn_accnum)/(num / (testuser['repeat']+1))
            diff_acc = syn_accnum/(num  - num / (testuser['repeat']+1))


        acc = 0
        num = 0
        avg_loss_valid = 0
        algorithm_copy = deepcopy(algorithm)
        algorithm_copy.eval()
        for batch_no,  minibatch in enumerate(valid_loader, start=1): #new_domain
            x=  minibatch[0] #batch, length, channel
            y = minibatch[1]

            x, y=x.to(device), y.long().to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(3)
            x = x.transpose(1, 2).squeeze(3).unsqueeze(2) # batch, channel,1,length
            y_pred_list = []
          
            y_pred,target_z = algorithm_copy.predict(x.float())
            y_prob = F.softmax(y_pred, dim=1)
            y_pred_list.append(y_prob.cpu().detach().numpy() )

            y_pred_list = np.array(y_pred_list) #K * (batch, class)
            class_score = np.sum(y_pred_list,axis = 0) #batch,class
            y_pred = np.argmax(class_score, axis=1)  #batch
            y_true = y.cpu().detach().numpy()

            #caculate acc
            acc += np.sum(y_pred == y_true)
            num += len(y_pred)
        # print(batch_no, "valid acc:", acc/num)
        # logger.debug("%s Valid acc: %s", batch_no, acc/num)
        valid_acc = acc/num
       # writer.add_scalar('Valid Accuracy', acc/num, epoch + 1)
        if acc/num > best_acc2:
            best_acc2 = acc/num 
            counter = 0  # 
            best_model = algorithm.state_dict()  
            logger.debug("Update")
            logger.debug("Best Valid acc: %s", best_acc2)
            flag = 1
            stop = 0
        else:
            flag = 0
            stop = stop + 1

        acc = 0
        num = 0
        avg_loss_valid = 0
        algorithm_copy = deepcopy(algorithm)
        algorithm_copy.eval()
        for batch_no,  minibatch in enumerate(test_loader, start=1):
            x=  minibatch[0]
            y = minibatch[1]
            x, y=x.to(device), y.long().to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(3)

            x = x.transpose(1, 2).squeeze(3).unsqueeze(2)
            y_pred_list = [] 
            y_pred, target_z= algorithm_copy.predict(x.float())
            y_prob = F.softmax(y_pred, dim=1) 
            pred1 = y_prob
            y_pred_list.append(y_prob.cpu().detach().numpy() )
            y_pred_list = np.array(y_pred_list) #K * (batch, class)
            class_score = np.sum(y_pred_list,axis = 0) #batch,class
            y_pred = np.argmax(class_score, axis=1)  #batch
            y_true = y.cpu().detach().numpy()
            acc += np.sum(y_pred == y_true)
            num += len(y_pred)
        test_acc = acc/num
        if flag:
            best_acc = acc/num
        logger.debug("%s ENV", testuser['name'])

        logger.debug(f"Train Accuracy: {train_acc:.5f}, Valid Accuracy: {valid_acc:.5f}, Test Accuracy: {test_acc:.5f}, Best Accuracy: {best_acc:.5f}")

        if stop > 80:
            print("early stop!")
            break
