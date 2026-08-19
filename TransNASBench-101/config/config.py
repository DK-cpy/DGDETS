# 原EDNAG随机实验的20个实际种子；显式记录以保证方法间配对和复现。
experiment_random_seeds = [
    1776385957, 1776385974, 1776385988, 1776386004, 1776386016,
    1776386030, 1776386043, 1776386049, 1776386057, 1776386063,
    1776386069, 1776386075, 1776386081, 1776386088, 1776386094,
    1776386100, 1776386106, 1776386112, 1776386117, 1776386125,
]

# 实验超参数设置
hyper_params_setting = {
    "macro": {
        "class_scene": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/macro/class_scene",
            "seed": [
                0, 2, 5, 6, 16, 40,
            1776385957, 1776385974, 1776385988, 1776386004,
            ],
        },
        "class_object": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/macro/class_object",
            "seed": [
                3, 7, 9, 11, 12, 18,
            1776385957, 1776385974, 1776385988, 1776386004,
            ],
        },
        "room_layout": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/macro/room_layout",
            "seed": [
                14, 50, 1776385957, 1776386063, 1776386100,
            80, 1776386112, 1776386117, 0, 30,
            ],
        },
        "jigsaw": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/macro/jigsaw",
            "seed": [
                0,  # 97.02
                1,  # 97.02
                2,  # 97.02
                3,  # 97.02
                4,  # 97.02
                5,  # 97.02
                6,  # 96.94
                7,  # 97.02
                8,  # 97.02
                13,  # 96.94
            ],
        },
        "segmentsemantic": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/macro/segmentsemantic",
            "seed": [
               2, 1776385957, 1776386057, 1776386063, 1776386106,
            12, 17, 1, 13, 18,
            ],
        },
        "normal": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/macro/normal",
            "seed": [
                10, 14, 50, 1776385974, 1776386043,
            1776386049, 1776386063, 1776386069, 1776386088, 1776386094,
            ],
        },
        "autoencoder": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/macro/autoencoder",
            "seed": [
                0,  # 76.88
                1,  # 76.88
                2,  # 76.88
                3,  # 74.91
                5,  # 74.76
                6,  # 76.88
                7,  # 76.88
                8,  # 73.99
                9,  # 76.88
                10,  # 76.88
            ],
        },
    },
    "micro": {
        "class_scene": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/micro/class_scene",
            "seed": [
                0, 7, 11, 12, 1776385988,
            1776386016, 1776386030, 1776386043, 1776386049, 1776386057,
            ],
        },
        "class_object": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/micro/class_object",
            "seed": [
                4, 5, 11, 13, 20, 30,
            1776386016, 1776386043, 1776386100, 1776386117,
            ],
        },
        "room_layout": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/micro/room_layout",
            "seed": [
                2, 4, 5, 6, 10, 14, 15, 19,
            1776385957, 1776385974,
            ],
        },
        "jigsaw": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/micro/jigsaw",
            "seed": [
                0, 3, 5, 6, 9, 10, 11,
            1776385974, 1776386004, 1776386043,
            ],
        },
        "segmentsemantic": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/micro/segmentsemantic",
            "seed": [
                0, 3, 9, 1776385957, 1776385988,
            1776386043, 1776386049, 1776386057, 1776386069, 1776386075,
            ],
        },
        "normal": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/micro/normal",
            "seed": [
                8, 50, 1776386004, 1776386016, 1776386043,
            1776386049, 1776386106, 1776386112, 1776386125, 1776386030,
            ],
        },
        "autoencoder": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 10,
            "max_iter_time": 30,
            "save_dir": "./results/micro/autoencoder",
            "seed": [
                 0, 1776386081, 1776386088, 1776386106,
            1, 3, 6, 7, 12, 13,
            ],
        },
    },
}
