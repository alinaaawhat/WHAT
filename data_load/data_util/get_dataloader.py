# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
sys.path.append('./data_load')
from data_load.data_util.sensor_loader import SensorDataset
from torch.utils.data import DataLoader
from data_load.data_util.fast_data_loader import InfiniteDataLoader
import numpy as np

import torch
def load(args):
    """Load data and get dataloader
    """
    raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = [], [], [], [], [], []
    data_path = args.root_path + args.dataset + "/"+ \
        f'{args.dataset}_crosssubject_rawaug_rate{args.remain_data_rate}_t{args.target}_seed{args.seed}_scaler{args.scaler_method}.pkl'
    # data_raw_aug = np.load(data_path, allow_pickle=True)
    data_path = f'/home/SHIH0020/robustlearn/datasets/FL_{args.dataset}/{args.dataset}_crosssubject_rawaug_rate{args.r}_t0_seed1_scalerminmax.pkl'
    try:
        data_raw_aug  = torch.load(data_path)
    except:
        data_raw_aug = np.load(data_path, allow_pickle=True)

    raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = data_raw_aug['raw_trs'], data_raw_aug[
        'aug_trs'], data_raw_aug['raw_vas'], data_raw_aug['aug_vas'], data_raw_aug['raw_tet'], data_raw_aug['aug_tet']

    train_raw_dataset = SensorDataset(
        raw_trs, aug=False, dataset=args.dataset)
    train_aug_dataset = SensorDataset(
        aug_trs, aug=True, dataset=args.dataset)

    val_raw_dataset = SensorDataset(
        raw_vas, aug=False, dataset=args.dataset)
    val_aug_dataset = SensorDataset(
        aug_vas, aug=True, dataset=args.dataset)

    test_raw_dataset = SensorDataset(
        raw_tet, aug=False, dataset=args.dataset)
    test_aug_dataset = SensorDataset(
        aug_tet, aug=True, dataset=args.dataset)

    tstep_per_epoch = int(len(aug_trs[0])/args.batch_size)
    if tstep_per_epoch < args.step_per_epoch:
        args.step_per_epoch = tstep_per_epoch


    train_raw_loader = DataLoader(
        dataset=train_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=True)

    train_aug_loader = InfiniteDataLoader(
        dataset=train_aug_dataset,
        sample_weights=None,
        batch_size=args.batch_size//2,
        num_workers=args.num_workers)

    val_raw_loader = DataLoader(
        dataset=val_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    val_aug_loader = DataLoader(
        dataset=val_aug_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    test_raw_loader = DataLoader(
        dataset=test_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    test_aug_loader = DataLoader(
        dataset=test_aug_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    return train_raw_loader, train_aug_loader, val_raw_loader, val_aug_loader, test_raw_loader, test_aug_loader


def load_client(args, client_id = 1):
    """Load data and get dataloader
    """
    raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = [], [], [], [], [], []
    data_path = args.root_path + args.dataset + "/"+ \
        f'{args.dataset}_crosssubject_rawaug_rate{args.remain_data_rate}_t{args.target}_seed{args.seed}_scaler{args.scaler_method}.pkl'
    # data_raw_aug = np.load(data_path, allow_pickle=True)
    data_path = f'/home/SHIH0020/robustlearn/datasets/FL_dsads/dsads_crosssubject_rawaug_rate{args.remain_rate}_t0_seed1_scalerminmax.pkl'
    try:
        data_raw_aug  = torch.load(data_path)
    except:
        data_raw_aug = np.load(data_path, allow_pickle=True)

    raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = data_raw_aug['client_raw_trs'][client_id], data_raw_aug[
        'client_aug_trs'][client_id], data_raw_aug['client_raw_vas'][client_id], data_raw_aug['client_aug_vas'][client_id], data_raw_aug['raw_tet'], data_raw_aug['aug_tet']

    train_raw_dataset = SensorDataset(
        raw_trs, aug=False, dataset=args.dataset)
    train_aug_dataset = SensorDataset(
        aug_trs, aug=True, dataset=args.dataset)

    val_raw_dataset = SensorDataset(
        raw_vas, aug=False, dataset=args.dataset)
    val_aug_dataset = SensorDataset(
        aug_vas, aug=True, dataset=args.dataset)

    test_raw_dataset = SensorDataset(
        raw_tet, aug=False, dataset=args.dataset)
    test_aug_dataset = SensorDataset(
        aug_tet, aug=True, dataset=args.dataset)

    tstep_per_epoch = int(len(aug_trs[0])/args.batch_size)
    if tstep_per_epoch < args.step_per_epoch:
        args.step_per_epoch = tstep_per_epoch


    train_raw_loader = DataLoader(
        dataset=train_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=True)

    train_aug_loader = InfiniteDataLoader(
        dataset=train_aug_dataset,
        sample_weights=None,
        batch_size=args.batch_size//2,
        num_workers=args.num_workers)

    val_raw_loader = DataLoader(
        dataset=val_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    val_aug_loader = DataLoader(
        dataset=val_aug_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    test_raw_loader = DataLoader(
        dataset=test_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    test_aug_loader = DataLoader(
        dataset=test_aug_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    return train_raw_loader, train_aug_loader, val_raw_loader, val_aug_loader, test_raw_loader, test_aug_loader






def load_client_fre(args, client_id = 1):
    """Load data and get dataloader
    """
    raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = [], [], [], [], [], []
    data_path = args.root_path + args.dataset + "/"+ \
        f'{args.dataset}_crosssubject_rawaug_rate{args.remain_data_rate}_t{args.target}_seed{args.seed}_scaler{args.scaler_method}.pkl'
    # data_raw_aug = np.load(data_path, allow_pickle=True)
    data_path = f"/home/SHIH0020/robustlearn/datasets/FL_{args.dataset}/{args.dataset}_crosssubject_rawaug_rate{args.remain_rate}_t0_seed1_scalerminmax.pkl"
    if args.dataset == 'emg':
        data_path = f"/home/SHIH0020/robustlearn/datasets/FL_{args.dataset}/{args.dataset}_crosssubject_rawaug_rate{args.remain_rate}_t0_seed1_scalerminmax_clientnum_{args.num_clients}.pkl"
    print(f"数据导入：{data_path}")
    try:
        data_raw_aug  = torch.load(data_path)
    except:
        data_raw_aug = np.load(data_path, allow_pickle=True)

    raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = data_raw_aug['client_raw_trs'][client_id], data_raw_aug[
        'client_aug_trs'][client_id], data_raw_aug['client_raw_vas'][client_id], data_raw_aug['client_aug_vas'][client_id], data_raw_aug['raw_tet'], data_raw_aug['aug_tet']
    print(f'client number:{client_id}')
    for data_list in [raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet]:
        data_list[0] = torch.fft.rfft(torch.from_numpy(data_list[0]), dim=-2).numpy()
        # print(f'fft后数据结构是什么：')
        # print(f'  data_list类型: {type(data_list)}')
        # print(f'  data_list长度: {len(data_list)}')
        # for i, item in enumerate(data_list):
        #     print(f'  元素{i}类型: {type(item)}, 形状: {getattr(item, "shape", "无shape属性")}, 数据类型: {getattr(item, "dtype", "无dtype属性")}')
    train_raw_dataset = SensorDataset(
        raw_trs, aug=False, dataset=args.dataset)
    train_aug_dataset = SensorDataset(
        aug_trs, aug=True, dataset=args.dataset)

    val_raw_dataset = SensorDataset(
        raw_vas, aug=False, dataset=args.dataset)
    val_aug_dataset = SensorDataset(
        aug_vas, aug=True, dataset=args.dataset)

    test_raw_dataset = SensorDataset(
        raw_tet, aug=False, dataset=args.dataset)
    test_aug_dataset = SensorDataset(
        aug_tet, aug=True, dataset=args.dataset)

    tstep_per_epoch = int(len(aug_trs[0])/args.batch_size)
    if tstep_per_epoch < args.step_per_epoch:
        args.step_per_epoch = tstep_per_epoch


    train_raw_loader = DataLoader(
        dataset=train_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=True)

    train_aug_loader = InfiniteDataLoader(
        dataset=train_aug_dataset,
        sample_weights=None,
        batch_size=args.batch_size//2,
        num_workers=args.num_workers)

    val_raw_loader = DataLoader(
        dataset=val_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    val_aug_loader = DataLoader(
        dataset=val_aug_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    test_raw_loader = DataLoader(
        dataset=test_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    test_aug_loader = DataLoader(
        dataset=test_aug_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    return train_raw_loader, train_aug_loader, val_raw_loader, val_aug_loader, test_raw_loader, test_aug_loader


# ###############################
# def load(args):
#     """Load data and get dataloader
#     """
#     raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = [], [], [], [], [], []
#     data_path = args.root_path + args.dataset + "/"+ \
#         f'{args.dataset}_crosssubject_rawaug_rate{args.remain_data_rate}_t{args.target}_seed{args.seed}_scaler{args.scaler_method}.pkl'
#     # data_raw_aug = np.load(data_path, allow_pickle=True)
#     try:
#         data_raw_aug  = torch.load(data_path)
#     except:
#         data_raw_aug = np.load(data_path, allow_pickle=True)

#     raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = data_raw_aug['raw_trs'], data_raw_aug[
#         'aug_trs'], data_raw_aug['raw_vas'], data_raw_aug['aug_vas'], data_raw_aug['raw_tet'], data_raw_aug['aug_tet']

#     train_raw_dataset = SensorDataset(
#         raw_trs, aug=False, dataset=args.dataset)
#     train_aug_dataset = SensorDataset(
#         aug_trs, aug=True, dataset=args.dataset)

#     val_raw_dataset = SensorDataset(
#         raw_vas, aug=False, dataset=args.dataset)
#     val_aug_dataset = SensorDataset(
#         aug_vas, aug=True, dataset=args.dataset)

#     test_raw_dataset = SensorDataset(
#         raw_tet, aug=False, dataset=args.dataset)
#     test_aug_dataset = SensorDataset(
#         aug_tet, aug=True, dataset=args.dataset)

#     tstep_per_epoch = int(len(aug_trs[0])/args.batch_size)
#     if tstep_per_epoch < args.step_per_epoch:
#         args.step_per_epoch = tstep_per_epoch


#     train_raw_loader = DataLoader(
#         dataset=train_raw_dataset,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         drop_last=False,
#         shuffle=True)

#     train_aug_loader = InfiniteDataLoader(
#         dataset=train_aug_dataset,
#         sample_weights=None,
#         batch_size=args.batch_size//2,
#         num_workers=args.num_workers)

#     val_raw_loader = DataLoader(
#         dataset=val_raw_dataset,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         drop_last=False,
#         shuffle=False)

#     val_aug_loader = DataLoader(
#         dataset=val_aug_dataset,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         drop_last=False,
#         shuffle=False)

#     test_raw_loader = DataLoader(
#         dataset=test_raw_dataset,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         drop_last=False,
#         shuffle=False)

#     test_aug_loader = DataLoader(
#         dataset=test_aug_dataset,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         drop_last=False,
#         shuffle=False)

#     return train_raw_loader, train_aug_loader, val_raw_loader, val_aug_loader, test_raw_loader, test_aug_loader
