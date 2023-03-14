# some parameters #
import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current_path = os.getcwd()

