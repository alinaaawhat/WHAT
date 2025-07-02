# DI2SDiff_FL



# Datasets

We employ the identical preprocessing method as outlined in the work by [Qin et al DDLearn](https://github.com/microsoft/robustlearn/tree/main/ddlearn) on three publicly available HAR datasets, namely [DSADS](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities), [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring), and [USC-HAD](https://sipi.usc.edu/had/). By running its data preprocess code involving deal, divide_domain, and raw_aug_loader, we can obtain the dataset with the specified target and remain rate. The data file should be put in the `data` folder, such as `data/uschad/uschad_crosssubject_rawaug_rate0.2_t0_seed1_scalernorm.pkl`. Here, we only use the preprocessed original data for training.

DSADS data store in 'uschad_crosssubject_rawaug_rate0.2_t0_seed1_scalernorm.pkl' and format, the training data is divided into 3 clients:
—— raw_trs ——
  type: list
  length = 2
  first element type: <class 'numpy.ndarray'>
  shape = (1626, 125, 45)

—— aug_trs ——
  type: list
  length = 3
  first element type: <class 'numpy.ndarray'>
  shape = (11382, 125, 45)

—— raw_vas ——
  type: list
  length = 2
  first element type: <class 'numpy.ndarray'>
  shape = (2712, 125, 45)

—— aug_vas ——
  type: list
  length = 3
  first element type: <class 'numpy.ndarray'>
  shape = (18984, 125, 45)

—— raw_trt ——
  type: list
  length = 2
  first element type: <class 'numpy.ndarray'>
  shape = (542, 125, 45)

—— aug_trt ——
  type: list
  length = 3
  first element type: <class 'numpy.ndarray'>
  shape = (3794, 125, 45)

—— raw_vat ——
  type: list
  length = 2
  first element type: <class 'numpy.ndarray'>
  shape = (904, 125, 45)

—— aug_vat ——
  type: list
  length = 3
  first element type: <class 'numpy.ndarray'>
  shape = (6328, 125, 45)

—— raw_tet ——
  type: list
  length = 2
  first element type: <class 'numpy.ndarray'>
  shape = (905, 125, 45)

—— aug_tet ——
  type: list
  length = 3
  first element type: <class 'numpy.ndarray'>
  shape = (6335, 125, 45)

—— client_raw_trs ——
  type: dict
  number of clients = 3
  client IDs: [1, 2, 3]
  client 1 data:
    type: <class 'list'>
    length: 2
    first element shape: (542, 125, 45)
    first element type: <class 'numpy.ndarray'>
    second element shape: (542,)
    second element type: <class 'numpy.ndarray'>

—— client_aug_trs ——
  type: dict
  number of clients = 3
  client IDs: [1, 2, 3]
  client 1 data:
    type: <class 'list'>
    length: 3
    first element shape: (3794, 125, 45)
    first element type: <class 'numpy.ndarray'>
    second element shape: (3794,)
    second element type: <class 'numpy.ndarray'>

—— client_raw_vas ——
  type: dict
  number of clients = 3
  client IDs: [1, 2, 3]
  client 1 data:
    type: <class 'list'>
    length: 2
    first element shape: (904, 125, 45)
    first element type: <class 'numpy.ndarray'>
    second element shape: (904,)
    second element type: <class 'numpy.ndarray'>

—— client_aug_vas ——
  type: dict
  number of clients = 3
  client IDs: [1, 2, 3]
  client 1 data:
    type: <class 'list'>
    length: 3
    first element shape: (6328, 125, 45)
    first element type: <class 'numpy.ndarray'>
    second element shape: (6328,)
    second element type: <class 'numpy.ndarray'>

=== Client Data Analysis ===

client_raw_trs:
  Client 1:
    Element 0: shape (542, 125, 45), dtype float64
    Element 1: shape (542,), dtype int64
  Client 2:
    Element 0: shape (542, 125, 45), dtype float64
    Element 1: shape (542,), dtype int64
  Client 3:
    Element 0: shape (542, 125, 45), dtype float64
    Element 1: shape (542,), dtype int64

client_aug_trs:
  Client 1:
    Element 0: shape (3794, 125, 45), dtype float64
    Element 1: shape (3794,), dtype float64
    Element 2: shape (3794,), dtype float64
  Client 2:
    Element 0: shape (3794, 125, 45), dtype float64
    Element 1: shape (3794,), dtype float64
    Element 2: shape (3794,), dtype float64
  Client 3:
    Element 0: shape (3794, 125, 45), dtype float64
    Element 1: shape (3794,), dtype float64
    Element 2: shape (3794,), dtype float64

client_raw_vas:
  Client 1:
    Element 0: shape (904, 125, 45), dtype float64
    Element 1: shape (904,), dtype int64
  Client 2:
    Element 0: shape (904, 125, 45), dtype float64
    Element 1: shape (904,), dtype int64
  Client 3:
    Element 0: shape (904, 125, 45), dtype float64
    Element 1: shape (904,), dtype int64

client_aug_vas:
  Client 1:
    Element 0: shape (6328, 125, 45), dtype float64
    Element 1: shape (6328,), dtype float64
    Element 2: shape (6328,), dtype float64
  Client 2:
    Element 0: shape (6328, 125, 45), dtype float64
    Element 1: shape (6328,), dtype float64
    Element 2: shape (6328,), dtype float64
  Client 3:
    Element 0: shape (6328, 125, 45), dtype float64
    Element 1: shape (6328,), dtype float64
    Element 2: shape (6328,), dtype float64

=== Data Structure Summary ===
Target domain: 0 (测试集)
Source domains (参与方): [1, 2, 3]
Total training samples across all clients: 1626
  Client 1: 542 train samples, 904 val samples
  Client 2: 542 train samples, 904 val samples
  Client 3: 542 train samples, 904 val samples

每个client的数据格式是：
pythonclient_data = [X, y]  # 对于raw数据
client_data = [X, y, aug_label]  # 对于aug数据
具体来说：
Raw数据 (client_raw_trs, client_raw_vas)
pythonclient_raw_trs[1] = [
    X,  # shape: (542, 125, 45) - 特征数据 [样本数, 时间窗口, 传感器特征]
    y   # shape: (542,) - 活动标签 [样本数]
]
Aug数据 (client_aug_trs, client_aug_vas)
pythonclient_aug_trs[1] = [
    X,          # shape: (3794, 125, 45) - 增强后的特征数据
    y,          # shape: (3794,) - 活动标签
    aug_label   # shape: (3794,) - 增强方法标签 (0-6表示7种增强方法)
]



# Prerequisites

1. See requirements.txt

```
 pip install -r requirements.txt
```

2. Prepare dataset

Please download the dataset and run the [preprocessing code](https://github.com/microsoft/robustlearn/tree/main/ddlearn). Once you have processed the data, move the resulting file to the data folder. For example, the processed file for the USC-HAD dataset with 20% of the data used for training, the target set to 1, and a seed value of 1 should be saved as `data/uschad/uschad_crosssubject_rawaug_rate0.2_t0_seed1_scalernorm.pkl`.




# Instructions

We provide a brief introduction to each code folder. We change this model into Federated setting. As dsads as example: 

1. **Style_conditioner**: This folder is used to train the style conditioner, which is built based on [TS-TCC](https://github.com/emadeldeen24/TS-TCC/).  To adapt to our training dataset, we added new configs.
2. **Diffusion_model**: This folder is used to train our diffusion model, which is adapted from [denoising-diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch). to achieve the guidance goal, we made adjustments to the model structure and diffusion method, such as adding style embedding and multiple conditional guidance.
3. **Featurenet**: This folder is used to generate synthetic data and train the feature network for classification.

Style_conditioner & Diffusion_model's process is：first, Local Style_conditioner Training. Each client trains its own Style_conditioner on its local data to generate style embeddings.second, Local Diffusion Model Training.Using the local Style_conditioner, each client trains a diffusion model on its own data.Then, Model Aggregation: Clients upload their trained diffusion‐model parameters to the server, which aggregates them (e.g., by averaging) into a single global diffusion model.Then, Broadcast & Continue Training:The server distributes the aggregated global model back to each client. Clients use it as the initialization for the next round of local diffusion‐model training. Last, Iterate Until Convergence
Steps 2–4 are repeated for a fixed number of rounds (or until convergence), so that all clients eventually share the same aggregated diffusion model.
run"python combined_train.py \
    --seed 1 \
    --selected_dataset 'dsads' \
    --remain_rate 0.2 \
    --target 0 \
    --training_mode 'self_supervised' \
    --logs_save_dir './Style_conditioner/conditioner_pth/' \
    --results_folder './Diffusion_model/dm_pth/' \
    --local_training_steps 20000 \
    --aggregation_num 3 \
    --num_clients 3"

Featurenet:
We ran these experiments on a GeForce RTX 3090 Ti. The generation time may take some time, but we believe that existing fast diffusion models can help alleviate this problem.




