from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from utils.tools import pad_1D, pad_2D

class ReturnDataset(Dataset):
    def __init__(
        self, filename, configs, sort=False, drop_last=False, offset=0
    ):
        
        self.batch_size = configs["optimizer"]["batch_size"]
        self.n_window_timesteps = configs["model"]["n_window_timesteps"]

        self.timestep, self.date, self.returns = self.process_data(
            filename, offset=offset
        )
    
        self.sort = sort
        self.drop_last = drop_last       

    def __len__(self):
        return len(self.date)

    def __getitem__(self, idx):
        timestep = self.timestep[idx]
        date = self.date[idx]
        returns = self.returns[idx]

        n = len(timestep)//2

        sample = {
            "enc_timestep": timestep[:n],
            "enc_date": date[:n],
            "enc_returns": returns[:n],
            "dec_timestep": timestep[n:2*n],
            "dec_date": date[n:2*n],
            "dec_returns": returns[n:2*n],
            "target_returns": returns[n+1:2*n+1]
        }

        return sample

    def process_data(self, filename, offset, index_col="Date"):
            data = pd.read_csv(filename, index_col=index_col)
            timestep = []
            dates = []
            returns = []

            for i in range(data.shape[0] - 2*self.n_window_timesteps):
                data_slice = data.iloc[i:i+2*self.n_window_timesteps+1] 
                timestep.append(list(range(offset+i, offset+i+2*self.n_window_timesteps+1)))
                dates.append(list(data_slice.index))
                returns.append(data_slice.to_numpy().astype(np.float32))

            return timestep, dates, returns
    
    def _make_batch(self, data, idxs, prefix):
        _timestep = np.array([data[idx][f"{prefix}_timestep"] for idx in idxs]).astype(np.float32)
        _date = [data[idx][f"{prefix}_date"] for idx in idxs]
        _returns = [data[idx][f"{prefix}_returns"] for idx in idxs]

        return _timestep, _date, _returns

    def reprocess(self, data, idxs):
        enc_timestep, enc_date, enc_returns = self._make_batch(data, idxs, "enc")
        dec_timestep, dec_date, dec_returns = self._make_batch(data, idxs, "dec")

        target_returns =  [data[idx]["target_returns"] for idx in idxs]

        enc_inp_lens = np.array([ret.shape[0] for ret in enc_returns])
        dec_inp_lens = np.array([ret.shape[0] for ret in dec_returns])

        enc_timestep = pad_1D(enc_timestep)
        enc_returns = pad_2D(enc_returns)

        dec_timestep = pad_1D(dec_timestep)
        dec_returns = pad_2D(dec_returns)

        target_returns = pad_2D(target_returns)
        
        return ( #ToDo - refactor
            enc_date,
            dec_date,
            enc_returns,
            enc_timestep,
            enc_inp_lens,
            max(enc_inp_lens),
            dec_returns,
            dec_timestep,
            dec_inp_lens,
            max(dec_inp_lens),
            target_returns
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["enc_timestep"][0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output