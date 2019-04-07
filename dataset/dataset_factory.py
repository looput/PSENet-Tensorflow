"""A factory-pattern class which returns classification image/label pairs."""

class DatasetConfig():
    def __init__(self, file_pattern, split_sizes,dataset_dir=None):
        self.file_pattern = file_pattern
        self.split_sizes = split_sizes
        self.dataset_dir=dataset_dir
        
icdar2013 = DatasetConfig(
        file_pattern = '*_%s.tfrecord', 
        split_sizes = {
            'train': 229,
            'test': 233
        }
)
icdar2015 = DatasetConfig(
        file_pattern = 'icdar2015_%s.tfrecord', 
        split_sizes = {
            'train': 1000,
            'test': 500
        },
        dataset_dir='/home/lupu'
)
ic15mlt = DatasetConfig(
        file_pattern = 'ic15mlt_%s.tfrecord', 
        split_sizes = {
            'train': 9966,
            'test': 500
        },
        dataset_dir='/home/lupu'
)

td500 = DatasetConfig(
        file_pattern = 'TD500_%s.tfrecord', 
        split_sizes = {
            'train': 300,
            'test': 200
        },
        dataset_dir='/home/lupu'
)
tr400 = DatasetConfig(
        file_pattern = 'tr400_%s.tfrecord', 
        split_sizes = {
            'train': 400
        }
)
scut = DatasetConfig(
    file_pattern = 'scut_%s.tfrecord',
    split_sizes = {
        'train': 1715
    }
)

synthtext = DatasetConfig(
    file_pattern = '*.tfrecord',
#     file_pattern = 'SynthText_*.tfrecord',
    split_sizes = {
        'train': 858750
    }
)

ctw1500 = DatasetConfig(
    file_pattern = 'ctw_%s.tfrecord',
    split_sizes = {
        'train': 1000
    },
    dataset_dir='/home/lupu'
)

datasets_map = {
    'icdar2013':icdar2013,
    'icdar2015':icdar2015,
    'ic15mlt':ic15mlt,
    'scut':scut,
    'td500':td500,
    'tr400':tr400,
    'synthtext':synthtext,
    'ctw1500':ctw1500
}