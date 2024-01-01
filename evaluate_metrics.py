import numpy as np
import pandas as pd
import torch
def hamming_loss(y_GT, predict):
    GT_size = np.sum(y_GT, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    temp = 0

    for i in range(GT_size.shape[0]):
        index=sorted[i][-GT_size[i]:][::-1]
        predict_label[i][index]=1
        temp = temp + np.sum(y_GT[i] ^ predict_label[i])

    hmloss = temp/(y_GT.shape[0]*y_GT.shape[1])
    return hmloss

def FR(y_GT, predict):
    GT_size = np.sum(y_GT, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    temp = 0

    for i in range(GT_size.shape[0]):
        index = sorted[i][-GT_size[i]:][::-1]
        predict_label[i][index] = 1
        if np.sum(y_GT[i] ^ predict_label[i])>0:
            temp = temp + 1

    fr = temp / y_GT.shape[0]
    return fr

# def TP_index(y_targets, predict):
#     GT_size = np.sum(y_targets, axis=1)
#     predict_label = np.zeros(predict.shape, dtype=int)
#     sorted = predict.argsort()
#     tp_list = []
#
#     for i in range(GT_size.shape[0]):
#
#         index = sorted[i][-GT_size[i]:][::-1]
#         predict_label[i][index] = 1
#         if np.sum(y_targets[i] ^ predict_label[i])==0:
#             tp_list.append(i)
#
#     return tp_list, predict_label

def TP_index(y_targets, predict):
    GT_size = np.sum(y_targets, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    tp_flag = False

    for i in range(GT_size.shape[0]):
        index = sorted[i][-GT_size[i]:][::-1]
        predict_label[i][index] = 1
        if np.sum(y_targets[i] ^ predict_label[i])==0:
            tp_flag = True
    return tp_flag, predict_label


def nontargeted_TP_index(y_GT, predict, kvalue):
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    flag = True

    for i in range(predict.shape[0]):

        index = sorted[i][-kvalue:][::-1]
        # index = torch.flip(index, [0])
        predict_label[i][index] = 1
        for j in index:#
            if y_GT[i][j] == 1:
                flag = False
    
    return flag, predict_label

def UAP_TP_index(y_GT, predict, kvalue):
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    index = sorted[:, -kvalue:][:, ::-1]
    b_new = np.where(y_GT == 1)
    data = pd.DataFrame({'id': list(b_new[0]), 'value': list(np.char.mod('%d', b_new[1]))})
    data_new = data.groupby(by='id').apply(lambda x: [' '.join(x['value'])])
    count = 0
    for i in range(y_GT.shape[0]):
        predict_label[i][index[i]] = 1
        set_a = set(index[i])
        set_b = set(list(map(int, data_new[i][0].split(' '))))
        if len(set_a.intersection(set_b)) == 0:
            count = count + 1
    fooling_rate = count / y_GT.shape[0]
    return fooling_rate, predict_label

def label_match(y_GT, predict,k_value):
    GT_size = np.sum(y_GT, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    tp_list = []

    for i in range(GT_size.shape[0]):
        index = sorted[i][-k_value:][::-1]
        for j in index:
            if y_GT[j]==1:
                return False
        return True

def topk_acc_metric(y_GT, predict, kvalue):
    count = 0
    all_list = set(list(range(0, y_GT.shape[1])))
    for i in range(y_GT.shape[0]):
        GT_label = set(np.transpose(np.argwhere(y_GT[i]==1))[0])
        predict_index = predict[i].argsort()[-kvalue:][::-1]
        predict_complement_set = all_list - set(predict_index)
        count = count + int(predict_complement_set.issuperset(GT_label))
    return count / y_GT.shape[0]


def topk_acc_metric_1_to_10(y_GT, predict):
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_10 = 0

    all_list = set(list(range(0, y_GT.shape[1])))

    for i in range(y_GT.shape[0]):
        GT_label = set(np.transpose(np.argwhere(y_GT[i]==1))[0])
        predict_index_1 = predict[i].argsort()[-1:][::-1]
        predict_index_2 = predict[i].argsort()[-2:][::-1]
        predict_index_3 = predict[i].argsort()[-3:][::-1]
        predict_index_4 = predict[i].argsort()[-4:][::-1]
        predict_index_5 = predict[i].argsort()[-5:][::-1]
        predict_index_10 = predict[i].argsort()[-10:][::-1]

        predict_complement_set_1 = all_list - set(predict_index_1)
        predict_complement_set_2 = all_list - set(predict_index_2)
        predict_complement_set_3 = all_list - set(predict_index_3)
        predict_complement_set_4 = all_list - set(predict_index_4)
        predict_complement_set_5 = all_list - set(predict_index_5)
        predict_complement_set_10 = all_list - set(predict_index_10)

        count_1 = count_1 + int(predict_complement_set_1.issuperset(GT_label))
        count_2 = count_2 + int(predict_complement_set_2.issuperset(GT_label))
        count_3 = count_3 + int(predict_complement_set_3.issuperset(GT_label))
        count_4 = count_4 + int(predict_complement_set_4.issuperset(GT_label))
        count_5 = count_5 + int(predict_complement_set_5.issuperset(GT_label))
        count_10 = count_10 + int(predict_complement_set_10.issuperset(GT_label))

    return count_1 / y_GT.shape[0],count_2 / y_GT.shape[0],count_3 / y_GT.shape[0],count_4 / y_GT.shape[0],count_5 / y_GT.shape[0],count_10 / y_GT.shape[0]

def filp_label(model, inputs, purtabed_out):

    before_pred_label = (model(inputs).detach().cpu() > 0).long()
    pred_idx = torch.nonzero(before_pred_label.squeeze() == 1, as_tuple=False).squeeze(0).detach().cpu()
    after_pred_pos = (purtabed_out[0, pred_idx] > 0).long()

    before_sum = before_pred_label.sum().item()
    after_sum = before_pred_label.sum().item() - after_pred_pos.sum().item()
    if before_sum != 0:
        f = after_sum / before_sum
    else:
        f = 1
    return before_sum, after_sum, f

def New_E(GT, purtabed_out, k):

    new_po = purtabed_out.detach().cpu()
    GT_idx = torch.nonzero(GT.squeeze(0) == 1, as_tuple=False).squeeze(0).detach().cpu()
    if len(GT_idx.shape) > 1:
        GT_idx = GT_idx.squeeze(1)
    
    GT_list = GT_idx.detach().cpu().numpy().tolist()
    _, pred_idx = torch.topk(new_po[0], k)
    pred_idx_l = pred_idx.detach().cpu().numpy().tolist()

    iou1 = len(list(set(GT_list) & set(pred_idx_l))) / len(list(set(GT_list) | set(pred_idx_l)))
    iou2 = len(list(set(GT_list) & set(pred_idx_l))) / min(len(GT_list), k)
    iou3 = len(list(set(GT_list) & set(pred_idx_l))) / max(len(GT_list), k)
    iou4 = len(list(set(GT_list) & set(pred_idx_l))) / len(GT_list)
    return iou1, iou2, iou3, iou4, GT_list, pred_idx_l

def New_E_T(target_idx, pred):
    sorted = pred.argsort()[0]
    k = len(target_idx)
    target_idx_set = set(target_idx)
    pred_topk = sorted[-k:].detach().cpu().numpy().tolist()
    Eious_iou = len(list(target_idx_set & set(pred_topk))) / len(list(target_idx_set | set(pred_topk[-k:]))) 
    Eious_min_or_GT_max = len(list(target_idx_set & set(pred_topk))) / k
    return Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk

def label_flip_T(Pd_idx1, logits):
    logit_idx = logits[0, Pd_idx1]
    after = (logit_idx < 0).long().sum().item()
    frac = 1.0
    if Pd_idx1:
        frac = after / len(Pd_idx1)
    return len(Pd_idx1), after, frac

def other_class_influence(B_set1, B_set0, logits):
    logit1 = logits[0, B_set1]
    logit0 = logits[0, B_set0]
    
    s1 = (logit1 < 0).long().sum().item()
    s2 = (logit0 >= 0).long().sum().item()

    return len(B_set0 + B_set1), s1 + s2, (s1 + s2) / len(B_set0 + B_set1)