#!/usr/bin/env python
import logging
import os.path
import sys
import warnings

import pandas as pd
from language_checker import CheckRussian
from run_filtration import HeuristicsFilter


class Worker:
    def __init__(self, batch, size=1000, offset=0):
        self.batch = batch
        self.df = self.read_dataframe()
        self.offset = offset
        self.size = size
        self.hf = HeuristicsFilter(CheckRussian())

    @staticmethod
    def read_dataframe():
        filename = f'/data/in/{batch}.csv'
        df = pd.read_csv(filename, sep='\t', encoding='utf8')
        df['index'] = df.index
        df = df[df['question'] != 'GENERATION_ERROR']
        df = df[df['answer'] != 'GENERATION_ERROR']
        df.rename(columns={'short_summary': 'summary'}, inplace=True)
        logging.info(df.shape)
        return df

    def get_portion(self):
        logging.info(self.df.shape)
        logging.info(self.offset)
        logging.info(self.size)
        n = min(self.offset + self.size, self.df.shape[0])
        try:
            while n < self.df.shape[0] and self.df['text_index'][n-1] == self.df['text_index'][n]:
                n += 1
        except:
            pass
        portion = self.df[self.offset:n]
        self.offset = n
        return portion

    def run(self):
        while self.offset < self.df.shape[0]:
            logging.info(f'Starting at {self.offset}')
            portion = self.get_portion()
            logging.info(f'Processing {len(portion)} lines')
            out_file = f'/data/out/{self.batch}_{self.offset - len(portion)}_{self.offset}.csv'
            if os.path.exists(out_file):
                logging.warning(f'Skipping existing portion: {out_file}')
                continue
            df_filtered = self.hf.do_filter(portion)
            logging.info(f'Got {len(df_filtered)} rows')
            df_filtered.to_csv(out_file, sep='\t', encoding='utf8', index=False)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    warnings.filterwarnings("ignore")
    batch = sys.argv[1]
    w = Worker(batch, 1000, 0)
    w.run()
