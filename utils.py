import os
import torch


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def stream_print(f, string, pipe_to_file=True):
    print(string)
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

def open_file(fname):
    if fname is None:
        return None
    else:
        i = 0
        while os.path.isfile('{:s}.{:d}'.format(fname, i)):
            i += 1
        return open('{:s}.{:d}'.format(fname, i), 'w')