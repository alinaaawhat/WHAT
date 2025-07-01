# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import utils
from main import args_parse
from aug_preprocess import raw_to_aug


def load_raw_aug_data(raw_data, aug_data, scaler_method, dataset, target, n_domain):
    raw_trs, aug_trs, raw_vas, aug_vas, raw_trt, aug_trt, raw_vat, aug_vat, raw_tet, aug_tet = [
    ], [], [], [], [], [], [], [], [], []
    raw_tr_sx, raw_tr_sy, raw_va_sx, raw_va_sy, raw_tr_tx, raw_tr_ty, raw_va_tx, raw_va_ty, raw_te_tx, raw_te_ty = [
    ], [], [], [], [], [], [], [], [], []
    aug_tr_sx, aug_tr_sy, aug_va_sx, aug_va_sy, aug_tr_tx, aug_tr_ty, aug_va_tx, aug_va_ty, aug_te_tx, aug_te_ty = [
    ], [], [], [], [], [], [], [], [], []
    
    # ===== change: 分别收集每个client的数据 =====
    client_raw_train_data = {}  # {client_id: [x, y]}
    client_aug_train_data = {}  # {client_id: [x, y, auglabel]}  
    client_raw_val_data = {}    # {client_id: [x, y]}
    client_aug_val_data = {}    # {client_id: [x, y, auglabel]}
    
    for i in range(n_domain):
        if i == target:
            raw_tr_tx, raw_tr_ty = pick_data(
                raw_data, 'raw', 'train', i)
            aug_tr_tx, aug_tr_ty, auglabel_t_tr = pick_data(
                aug_data, 'aug', 'train', i)

            raw_va_tx, raw_va_ty = pick_data(
                raw_data, 'raw', 'val', i)
            aug_va_tx, aug_va_ty, auglabel_t_va = pick_data(
                aug_data, 'aug', 'val', i)

            raw_te_tx, raw_te_ty = pick_data(
                raw_data, 'raw', 'test', i)
            aug_te_tx, aug_te_ty, auglabel_t_te = pick_data(
                aug_data, 'aug', 'test', i)
        else:
            # ===== 新增：为每个source domain单独收集数据 =====
            raw_tr_sx_i, raw_tr_sy_i = pick_data(raw_data, 'raw', 'train', i)
            raw_va_sx_i, raw_va_sy_i = pick_data(raw_data, 'raw', 'val', i)
            aug_tr_sx_i, aug_tr_sy_i, auglabel_s_tr_i = pick_data(
                aug_data, 'aug', 'train', i)
            aug_va_sx_i, aug_va_sy_i, auglabel_s_va_i = pick_data(
                aug_data, 'aug', 'val', i)
            
            # 存储每个client的数据
            client_raw_train_data[i] = [raw_tr_sx_i, raw_tr_sy_i]
            client_aug_train_data[i] = [aug_tr_sx_i, aug_tr_sy_i, auglabel_s_tr_i]
            client_raw_val_data[i] = [raw_va_sx_i, raw_va_sy_i]
            client_aug_val_data[i] = [aug_va_sx_i, aug_va_sy_i, auglabel_s_va_i]
            
            # ===== 保持原逻辑：合并所有source数据 =====
            if len(raw_tr_sx) == 0:
                raw_tr_sx = raw_tr_sx_i
                raw_tr_sy = raw_tr_sy_i
                aug_tr_sx = aug_tr_sx_i
                aug_tr_sy = aug_tr_sy_i
                auglabel_s_tr = auglabel_s_tr_i

                raw_va_sx = raw_va_sx_i
                raw_va_sy = raw_va_sy_i
                aug_va_sx = aug_va_sx_i
                aug_va_sy = aug_va_sy_i
                auglabel_s_va = auglabel_s_va_i
            else:
                raw_tr_sx = np.vstack((raw_tr_sx, raw_tr_sx_i))
                raw_tr_sy = np.vstack((raw_tr_sy, raw_tr_sy_i))
                aug_tr_sx = np.vstack((aug_tr_sx, aug_tr_sx_i))
                aug_tr_sy = np.vstack((aug_tr_sy, aug_tr_sy_i))
                auglabel_s_tr = np.vstack((auglabel_s_tr, auglabel_s_tr_i))

                raw_va_sx = np.vstack((raw_va_sx, raw_va_sx_i))
                raw_va_sy = np.vstack((raw_va_sy, raw_va_sy_i))
                aug_va_sx = np.vstack((aug_va_sx, aug_va_sx_i))
                aug_va_sy = np.vstack((aug_va_sy, aug_va_sy_i))
                auglabel_s_va = np.vstack((auglabel_s_va, auglabel_s_va_i))

    raw_tr_sx, raw_va_sx, raw_tr_tx, raw_va_tx, raw_te_tx, aug_tr_sx, aug_va_sx, aug_tr_tx, aug_va_tx, aug_te_tx = data_scaler(scaler_method,
                                                                                                                               raw_tr_sx, raw_va_sx, raw_tr_tx, raw_va_tx, raw_te_tx, aug_tr_sx, aug_va_sx, aug_tr_tx, aug_va_tx, aug_te_tx, dataset)
    
    # ===== change: 对每个client的数据也进行scaling =====
    for client_id in client_raw_train_data.keys():
        client_raw_train_data[client_id][0], client_raw_val_data[client_id][0], _, client_aug_train_data[client_id][0], client_aug_val_data[client_id][0], _ = data_scaler_client(
            scaler_method, 
            client_raw_train_data[client_id][0], 
            client_raw_val_data[client_id][0],
            client_aug_train_data[client_id][0], 
            client_aug_val_data[client_id][0],
            dataset
        )
    
    raw_trs.append(raw_tr_sx)
    raw_trs.append(raw_tr_sy.reshape(-1,))
    aug_trs.append(aug_tr_sx)
    aug_trs.append(aug_tr_sy.reshape(-1,))
    aug_trs.append(auglabel_s_tr.reshape(-1,))

    raw_vas.append(raw_va_sx)
    raw_vas.append(raw_va_sy.reshape(-1,))
    aug_vas.append(aug_va_sx)
    aug_vas.append(aug_va_sy.reshape(-1,))
    aug_vas.append(auglabel_s_va.reshape(-1,))

    raw_trt.append(raw_tr_tx)
    raw_trt.append(raw_tr_ty.reshape(-1,))
    aug_trt.append(aug_tr_tx)
    aug_trt.append(aug_tr_ty.reshape(-1,))
    aug_trt.append(auglabel_t_tr.reshape(-1,))

    raw_vat.append(raw_va_tx)
    raw_vat.append(raw_va_ty.reshape(-1,))
    aug_vat.append(aug_va_tx)
    aug_vat.append(aug_va_ty.reshape(-1,))
    aug_vat.append(auglabel_t_va.reshape(-1,))

    raw_tet.append(raw_te_tx)
    raw_tet.append(raw_te_ty.reshape(-1,))
    aug_tet.append(aug_te_tx)
    aug_tet.append(aug_te_ty.reshape(-1,))
    aug_tet.append(auglabel_t_te.reshape(-1,))
    
    # ===== change：整理client数据格式 =====
    client_raw_trs = {}
    client_aug_trs = {}
    client_raw_vas = {}
    client_aug_vas = {}
    
    for client_id in client_raw_train_data.keys():
        client_raw_trs[client_id] = [
            client_raw_train_data[client_id][0],
            client_raw_train_data[client_id][1].reshape(-1,)
        ]
        client_aug_trs[client_id] = [
            client_aug_train_data[client_id][0],
            client_aug_train_data[client_id][1].reshape(-1,),
            client_aug_train_data[client_id][2].reshape(-1,)
        ]
        client_raw_vas[client_id] = [
            client_raw_val_data[client_id][0],
            client_raw_val_data[client_id][1].reshape(-1,)
        ]
        client_aug_vas[client_id] = [
            client_aug_val_data[client_id][0],
            client_aug_val_data[client_id][1].reshape(-1,),
            client_aug_val_data[client_id][2].reshape(-1,)
        ]
    
    return raw_trs, aug_trs, raw_vas, aug_vas, raw_trt, aug_trt, raw_vat, aug_vat, raw_tet, aug_tet, client_raw_trs, client_aug_trs, client_raw_vas, client_aug_vas


# ===== 新增：单个client的数据scaling函数 =====
def data_scaler_client(scaler_method, x_train_raw, x_val_raw, x_train_aug, x_val_aug, dataset):
    x_train_raw = reshape_data(x_train_raw, dataset, 'begin')
    x_val_raw = reshape_data(x_val_raw, dataset, 'begin') 
    x_train_aug = reshape_data(x_train_aug, dataset, 'begin')
    x_val_aug = reshape_data(x_val_aug, dataset, 'begin')
    
    scaler = MinMaxScaler()
    scaler_aug = MinMaxScaler()
    x_train_raw = scaler.fit_transform(x_train_raw)
    x_val_raw = scaler.transform(x_val_raw)
    
    x_train_aug = scaler_aug.fit_transform(x_train_aug)
    x_val_aug = scaler_aug.transform(x_val_aug)
    
    x_train_raw = reshape_data(x_train_raw, dataset, 'end')
    x_val_raw = reshape_data(x_val_raw, dataset, 'end')
    x_train_aug = reshape_data(x_train_aug, dataset, 'end')
    x_val_aug = reshape_data(x_val_aug, dataset, 'end')
    
    return x_train_raw, x_val_raw, None, x_train_aug, x_val_aug, None


def data_scaler(scaler_method, x_trs_raw, x_vas_raw, x_trt_raw, x_vat_raw, x_tet_raw, x_trs_aug, x_vas_aug, x_trt_aug, x_vat_aug, x_tet_aug, dataset):
    x_trs_raw, x_vas_raw, x_trt_raw, x_vat_raw, x_tet_raw, x_trs_aug, x_vas_aug, x_trt_aug, x_vat_aug, x_tet_aug = reshape_data(x_trs_raw, dataset, 'begin'), reshape_data(x_vas_raw, dataset, 'begin'),\
        reshape_data(x_trt_raw, dataset, 'begin'), reshape_data(x_vat_raw, dataset, 'begin'), reshape_data(x_tet_raw, dataset, 'begin'), reshape_data(x_trs_aug, dataset, 'begin'), reshape_data(
            x_vas_aug, dataset, 'begin'), reshape_data(x_trt_aug, dataset, 'begin'), reshape_data(x_vat_aug, dataset, 'begin'), reshape_data(x_tet_aug, dataset, 'begin')
    scaler = MinMaxScaler()
    scaler_aug = MinMaxScaler()
    x_trs_raw = scaler.fit_transform(x_trs_raw)
    x_vas_raw = scaler.transform(x_vas_raw)

    x_trt_raw = scaler.transform(x_trt_raw)
    x_vat_raw = scaler.transform(x_vat_raw)
    x_tet_raw = scaler.transform(x_tet_raw)
    x_trs_aug = scaler_aug.fit_transform(x_trs_aug)
    x_vas_aug = scaler_aug.transform(x_vas_aug)
    x_trt_aug = scaler_aug.transform(x_trt_aug)
    x_vat_aug = scaler_aug.transform(x_vat_aug)
    x_tet_aug = scaler_aug.transform(x_tet_aug)
    x_trs_raw, x_vas_raw, x_trt_raw, x_vat_raw, x_tet_raw, x_trs_aug, x_vas_aug, x_trt_aug, x_vat_aug, x_tet_aug = reshape_data(x_trs_raw, dataset, 'end'), reshape_data(x_vas_raw, dataset, 'end'),\
        reshape_data(x_trt_raw, dataset, 'end'), reshape_data(x_vat_raw, dataset, 'end'), reshape_data(x_tet_raw, dataset, 'end'), reshape_data(x_trs_aug, dataset, 'end'), reshape_data(
            x_vas_aug, dataset, 'end'), reshape_data(x_trt_aug, dataset, 'end'), reshape_data(x_vat_aug, dataset, 'end'), reshape_data(x_tet_aug, dataset, 'end')
    return x_trs_raw, x_vas_raw, x_trt_raw, x_vat_raw, x_tet_raw, x_trs_aug, x_vas_aug, x_trt_aug, x_vat_aug, x_tet_aug


def reshape_data(x, dataset, when):
    if when == 'begin':
        if dataset == 'dsads':
            x = x.reshape(-1, 45)
        elif dataset == 'uschad':
            x = x.reshape(-1, 6)
        elif dataset == 'pamap':
            x = x.reshape(-1, 27)
    elif when == 'end':
        if dataset == 'dsads':
            x = x.reshape(-1, 125, 45)
        elif dataset == 'uschad':
            x = x.reshape(-1, 500, 6)
        elif dataset == 'pamap':
            x = x.reshape(-1, 512, 27)
    else:
        print("error")
    return x


def pick_data(data, data_type, data_name, src):
    x, y, auglabel = [], [], []
    if data_type == 'raw':
        if data_name == 'train':
            x = data[src][0][0]
            y = data[src][0][1]
        elif data_name == 'val':
            x = data[src][1][0]
            y = data[src][1][1]
        elif data_name == 'test':
            x = data[src][2][0]
            y = data[src][2][1]
        y = y.reshape(-1, 1)
        return x, y
    if data_type == 'aug':
        if data_name == 'train':
            x = data[src][0][0]
            y = data[src][0][1]
            auglabel = data[src][0][2]
        elif data_name == 'val':
            x = data[src][1][0]
            y = data[src][1][1]
            auglabel = data[src][1][2]
        elif data_name == 'test':
            x = data[src][2][0]
            y = data[src][2][1]
            auglabel = data[src][2][2]
        y = y.reshape(-1, 1)
        auglabel = auglabel.reshape(-1, 1)
        return x, y, auglabel


def set_param(dataset):
    if dataset == 'dsads':
        n_domain = 4
    elif dataset == 'pamap':
        n_domain = 4
    elif dataset == 'uschad':
        n_domain = 5
    return n_domain


if __name__ == "__main__":
    args = args_parse()
    root_path = "/home/SHIH0020/robustlearn/datasets/"
    for args.dataset in ['dsads','pamap','uschad']:
        n_domain = set_param(args.dataset)
        for args.scaler_method in ['minmax']:
            for remain_data_rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
                for args.seed in range(1, 4, 1):
                    utils.set_random_seed(args.seed)
                    raw_data, aug_data = raw_to_aug(args.seed, root_path, save_path=None,
                                                    dataset=args.dataset, aug_num=7, remain_data_rate=remain_data_rate)
                    # for target in range(n_domain):
                    target = 0
                    save_path = root_path + \
                        f'FL_{args.dataset}/{args.dataset}_crosssubject_rawaug_rate{remain_data_rate}_t{target}_seed{args.seed}_scaler{args.scaler_method}.pkl'
                    print(save_path)
                    # ===== 修改：接收新的返回值 =====
                    raw_trs, aug_trs, raw_vas, aug_vas, raw_trt, aug_trt, raw_vat, aug_vat, raw_tet, aug_tet, client_raw_trs, client_aug_trs, client_raw_vas, client_aug_vas = load_raw_aug_data(
                        raw_data, aug_data, args.scaler_method, args.dataset, target, n_domain)
                    
                    # ===== 修改：保存时添加client数据 =====
                    raw_and_aug = {
                        'raw_trs': raw_trs,
                        'aug_trs': aug_trs,
                        'raw_vas': raw_vas,
                        'aug_vas': aug_vas,
                        'raw_trt': raw_trt,
                        'aug_trt': aug_trt,
                        'raw_vat': raw_vat,
                        'aug_vat': aug_vat,
                        'raw_tet': raw_tet,
                        'aug_tet': aug_tet,
                        # 新增client数据
                        'client_raw_trs': client_raw_trs,
                        'client_aug_trs': client_aug_trs,
                        'client_raw_vas': client_raw_vas,
                        'client_aug_vas': client_aug_vas
                    }
                    with open(save_path, 'wb') as f:
                        pickle.dump(raw_and_aug, f)