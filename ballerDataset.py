import torch
from torch.utils.data import IterableDataset
from glob import glob

class BallerDataset(IterableDataset):
    def __init__(self, datasetDir, in_seq_len, out_seq_len, includePlayerOut=False):
        self.datasetDir = datasetDir
        self.in_seq_len, self.out_seq_len = in_seq_len, out_seq_len
        self.includePlayerOut = includePlayerOut
        
    def __iter__(self):
        return self.it()
    
    def it(self):
        if self.includePlayerOut:
            for datasetIn, datasetOut, datasetPlayerOut in zip(sorted(glob(self.datasetDir+f'/*_{self.in_seq_len}_{self.out_seq_len}_in.pt')), sorted(glob(self.datasetDir+f'/*_{self.in_seq_len}_{self.out_seq_len}_out.pt')), sorted(glob(self.datasetDir+f'/*_{self.in_seq_len}_{self.out_seq_len}_outPlayer.pt'))):
                # print(f'Opening {datasetIn}')
                if datasetIn.rstrip('_in.pt') != datasetOut.rstrip('_out.pt') or datasetPlayerOut.rstrip('_outPlayer.pt') != datasetOut.rstrip('_out.pt'):
                    raise Exception(f"File name mismatch In:{datasetIn}, Out:{datasetOut} Player:{datasetPlayerOut}")

                with open(datasetIn, 'rb') as fin, open(datasetOut, 'rb') as fout, open(datasetPlayerOut, 'rb') as fplayer:
                    x = torch.load(fin)
                    y = torch.load(fout)
                    player = torch.load(fplayer)
                    for i in range(x.shape[0]):
                        yield x[i], y[i], player[i]
        else:
            for datasetIn, datasetOut in zip(sorted(glob(self.datasetDir+f'/*_{self.in_seq_len}_{self.out_seq_len}_in.pt')), sorted(glob(self.datasetDir+f'/*_{self.in_seq_len}_{self.out_seq_len}_out.pt'))):
                # print(f'Opening {datasetIn}')
                if datasetIn.rstrip('_in.pt') != datasetOut.rstrip('_out.pt'):
                    raise Exception(f"File name mismatch In:{datasetIn}, Out:{datasetOut}")

                with open(datasetIn, 'rb') as fin, open(datasetOut, 'rb') as fout:
                    x = torch.load(fin)
                    y = torch.load(fout)
                    for i in range(x.shape[0]):
                        yield x[i], y[i]

        