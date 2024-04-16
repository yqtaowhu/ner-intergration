# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taoyanqi <taoyanqi@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/16 21:16:33 by taoyanqi          #+#    #+#              #
#    Updated: 2024/04/16 21:33:05 by taoyanqi         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import os

from fastNLP.io.loader import ConllLoader
from fastNLP import Vocabulary

class Data:
    def __init__(self, flags):
        self.train_path = os.path.join(flags.data_dir, flags.train_path)
        self.valid_path = os.path.join(flags.data_dir, flags.valid_path)
        self.test_path = os.path.join(flags.data_dir, flags.test_path)

        # 数据信息
        self.datasets = {}
        self.chars_vocab = Vocabulary()
        self.target_vocab = Vocabulary(unknown=None, padding=None)

    def load():
        pass

    def get_datasets(self):
        print('# begin init dataset:')
        paths = {}
        paths['train'] = self.train_path
        paths['dev'] = self.valid_path
        paths['test'] = self.test_path

        loader = ConllLoader(['raw_chars', 'target'])
        for k, v in paths.items():
            bundle = loader.load(v)
            self.datasets[k] = bundle.datasets['train']

        # 构造词表
        self.chars_vocab.from_dataset(self.datasets['train'], field_name='raw_chars', no_create_entry_dataset=[self.datasets['dev'], self.datasets['test']])
        self.target_vocab.from_dataset(self.datasets['train'], field_name='target')

        # char to index
        self.chars_vocab.index_dataset(*list(self.datasets.values()), field_name='raw_chars', new_field_name='chars')
        self.target_vocab.index_dataset(*list(self.datasets.values()), field_name='target', new_field_name='target')

        # 新增字段
        for k, v in self.datasets.items():
            v.add_seq_len('ras_chars', new_field_name='seq_len')