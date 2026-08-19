# 提供的可选数据集
nb201_dataset_list = ["cifar10", "cifar100", "ImageNet16-120"]  # NAS-Bench-201
meta_dataset_list = ["aircraft", "pets"]  # MetaD2A

# EDNAG originally generated these with int(time.time()).  Recording the actual
# 20-seed sequence in config.py makes the random protocol exactly repeatable and
# ensures every competing method uses the same seeds.
experiment_random_seeds = [
    1776386043, 1776386100, 5005, 1776385957,1776386117,
]

# 实验超参数设置
nb201_hyper_params_setting = {
    "cifar10": {
        "num_step": 2,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.2,
        "mutate_distri_index": 5,
        "rand_exp_num": 20,
        "max_iter_time": 30,
        "save_dir": "./results/nb201_benchmark/cifar10/",
        "nb201_or_meta": "nb201",
        "seed": [
            1731578139,  # max_acc@1: 94.37
            1731578141,  # max_acc@1: 94.37
            1731578146,  # max_acc@1: 94.37
            1731578150,  # max_acc@1: 94.37
            1731578154,  # max_acc@1: 94.37
        ],
    },
    "cifar100": {
        "num_step": 2,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.2,
        "mutate_distri_index": 5,
        "rand_exp_num": 20,
        "max_iter_time": 30,
        "save_dir": "./results/nb201_benchmark/cifar100/",
        "nb201_or_meta": "nb201",
        "seed": [
            1731578242,  # max_acc@1: 73.51
            1731578247,  # max_acc@1: 73.51
            1731578254,  # max_acc@1: 73.51
            1731578256,  # max_acc@1: 73.51
            1731578258,  # max_acc@1: 73.51
        ],
    },
    "ImageNet16-120": {
        "num_step": 2,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.2,
        "mutate_distri_index": 5,
        "rand_exp_num": 20,
        "max_iter_time": 30,
        "save_dir": "./results/nb201_benchmark/imagenet16_120/",
        "nb201_or_meta": "nb201",
        "seed": [
            1731578516,  # max_acc@1: 47.31
            1731578531,  # max_acc@1: 47.31
            1731578554,  # max_acc@1: 47.31
            1731578556,  # max_acc@1: 47.31
            1731578650,  # max_acc@1: 47.31
        ],
    },
}
meta_hyper_params_setting = {
    "aircraft": {
        "num_step": 20,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.3,
        "mutate_distri_index": 5,
        "rand_exp_num": 10,
        "max_iter_time": 90,
        "save_dir": "./results/meta/aircraft/",
        "nb201_or_meta": "meta",
        "eta_min": 0.0,
        "epochs": 200,
        "warmup": 10,
        "LR": 0.1,
        "decay": 0.0005,
        "momentum": 0.9,
        "nesterov": True,
        "batch_size": 256,
        "image_cutout": 5,
        "topk": 2,
        "early_stop": False,
        "multi_thread": False,
        "seed": [
            1776386125,  # max_acc@1: 61.09375
            555,  # max_acc@1: 60.41852679
            4567,  # max_acc@1: 59.15736607
            1776386043,  # max_acc@1: 59.15736607
            5005,  # max_acc@1: 58.90625 max_acc@2: 53.91
            1776386100,  # max_acc@1: 58.48772321
            1776386117,# max_acc@1: 58.32589286
            333,      # max_acc@1: 58.25892857
            1776386069,# max_acc@1: 58.15290179
            1776385957,# max_acc@1: 58.06919643 max_acc@2: 50.98
        ],
    },
    "pets": {
        "num_step": 20,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.3,
        "mutate_distri_index": 5,
        "rand_exp_num": 10,
        "max_iter_time": 90,
        "save_dir": "./results/meta/pets/",
        "nb201_or_meta": "meta",
        "eta_min": 0.0,
        "epochs": 200,
        "warmup": 10,
        "LR": 0.1,
        "decay": 0.0005,
        "momentum": 0.9,
        "nesterov": True,
        "batch_size": 256,
        "image_cutout": 5,
        "topk": 2,
        "early_stop": False,
        "multi_thread": False,
        "seed": [
            1776386100,  # max_acc@1: 49.57536774, max_acc@2: 0
            1776385957,  # max_acc@1: 36.36, max_acc@2: 48.71231689
            1776386043,  # max_acc@1: 46.52022095, max_acc@2: 0
            1776386057,  # max_acc@1: 46.05239029, max_acc@2: 0
            1776386088,  # max_acc@1: 45.97518387, max_acc@2: 38.08
            5005,  # max_acc@1: 45.11489029, max_acc@2:35.57
            1776386117,  # max_acc@1: 45.03860321, max_acc@2: 0
            1776386069,  # max_acc@1: 44.80330887, max_acc@2:0
            42,          # max_acc@1: 44.49080887, max_acc@2:0
            9012,        # max_acc@1: 44.48621368, max_acc@2:0
        ],
    },
}
