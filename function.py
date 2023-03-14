import torch

def onehot_encoding(label_list):
    encoded_label_list = torch.empty(0, 50)

    for label in label_list:
        tmp = torch.zeros(1, 50)
        tmp[:, int(label)] = 1
        encoded_label_list = torch.cat([encoded_label_list, tmp], dim=0)

    return encoded_label_list

def turn_to_score(score_vector): # transform the score vector(50-dim vector) into the original label
    maximum = score_vector[0]
    idx = 0
    max_index = 0
    while idx < len(score_vector):
        if score_vector[idx] > maximum:
            maximum = score_vector[idx]
            max_index = idx
        idx = idx + 1
    return max_index
    

def accurate_num(label, score_list): # calculate accurate number of a batch
    num = 0
    for idx in range(len(label)):
        if label[idx] == score_list[idx]:
            num = num + 1
    return num
