INFO = {
    'mnist': {
        'model_params': {
            'num_classes': 10,
        },
        'shape': (1, 28, 28),
        'moments': [[0.1307], [0.3081]],
        'task': 'image',
    },
    'emnist': {
        'model_params': {
            'num_classes': 10,
        },
        'shape': (1, 28, 28),
        'moments': [[0.1307], [0.3081]],
        'task': 'image',
    },
    'fashionmnist': {
        'model_params': {
            'num_classes': 10,
        },
        'shape': (1, 28, 28),
        'moments': [[0.1307], [0.3081]],
        'task': 'image',
    },
    'svhn': {
        'model_params': {
            'num_classes': 10,
        },
        'shape': (3, 32, 32),
        'moments': [(0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)],
        'task': 'image',
    },
    'cifar10': {
        'model_params': {
            'num_classes': 10,
        },
        'shape': (3, 32, 32),
        #'moments': [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
        'moments': [(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)],
        'task': 'image',
    },
    'cifar100': {
        'model_params': {
            'num_classes': 100,
        },
        'shape': (3, 32, 32),
        'task': 'image',
    },
    'wikitext': {
        'model_params': {
            'ntoken': 100,
            'ninp': 200,
        },
        'task': 'language',
    },
    'synthetic-niid00': {
        'model_params': {
            'num_classes': 10,
        },
    },
    'synthetic-niid0p50p5': {
        'model_params': {
            'num_classes': 10,
        },
    },
    'synthetic-niid11': {
        'model_params': {
            'num_classes': 10,
        },
    },
    'synthetic-iid': {
        'model_params': {
            'num_classes': 10,
        },
    },
}
