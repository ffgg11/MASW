import copy
import gc
import os
#from torch.autograd.gradcheck import zero_gradients
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from evaluate_metrics import (FR, New_E_T, TP_index, hamming_loss,
                              label_flip_T, nontargeted_TP_index,
                              other_class_influence, topk_acc_metric,
                              topk_acc_metric_1_to_10, filp_label, New_E)
from utils import generate_target_zeros_3_cases
from advertorch.utils import normalize_by_pnorm
import matplotlib.pyplot as plt

def get_jacobianU(model, x, noutputs, True_idx):
    num_instaces = x.size()[0]
    v = torch.eye(noutputs).cuda()
    jac = []
    x = x.clone().data.cpu()
    
    if torch.cuda.is_available():
        x = x.cuda()
    x.requires_grad = True
    y = torch.nn.functional.sigmoid(model(x)) 
    retain_graph = True
    n_trueidx = len(True_idx)
    for i in range(n_trueidx):
        idx = True_idx[i]
        if i == n_trueidx - 1:
            retain_graph = False
        y.backward(torch.unsqueeze(v[idx], 0).repeat(num_instaces, 1), retain_graph=retain_graph)
        g = x.grad.cpu().detach().numpy()
        x.grad.zero_()
        jac.append(g)
        
    jac = np.asarray(jac)
    y = y.cpu().detach().numpy()
    return jac, y

def generate_pert_DFU(x, model, GT, True_idx):

    #x = x.detach().cpu()
    num_labels = GT.shape[1]
    x_shape = np.shape(x)[1:]

    nb_labels = num_labels

    w = np.squeeze(np.zeros(x.shape[1:]))  # same shape as original image
    r_tot = np.zeros(x.shape)
    gradients, output = get_jacobianU(model, x, num_labels, True_idx)
    gradients = np.asarray(gradients)
    gradients = gradients.swapaxes(1, 0)
    predictions_val = output
    nb_labels = len(True_idx)
    for idx in range(np.shape(x)[0]):
        f = predictions_val[idx]
        f_true = -f[True_idx]
        c = np.array([0.5] * len(True_idx)) + f_true
        w = gradients[idx].reshape(nb_labels, -1)
        w_true = w
        P = w_true.T
    
        q = np.reshape(c, (-1, 1))
        
        temp = np.matmul(P.T, P)
        zeros = np.zeros(temp.shape[1])
        delete_idx = []
        for j in range(temp.shape[0]):
            if np.all(temp[j] == zeros):
                delete_idx.append(j)
        P = np.delete(P, delete_idx, axis=1)
        q = np.delete(q, delete_idx, axis=0)
        try:
            delta_r = np.matmul(np.matmul(P, np.linalg.inv(np.matmul(P.T, P))), q)
        except:
            continue
        delta_r = np.reshape(delta_r, x_shape)

        r_tot[idx] = delta_r
    return r_tot

def DFU(index, index_success, args, model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, device, Projection_flag = False, lr=1e-2):
    # trick from CW, normalize to [boxin, boxmin]
    model.eval()
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    GT_idx = torch.nonzero(label.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    Pd_ori = model(inputs)
    Pd_ori = (Pd_ori >= 0).long()
    Pd_idx = torch.nonzero(Pd_ori.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    timg = inputs
    tlab = label
    const = eps

    purtabed_img = (torch.tanh(inputs) * boxmul + boxplus).clone().data.cpu()
    purtabed_img = purtabed_img.cuda()
    converge_iter = maxiter
    attack_success = False
    c_list = [0] * 6
    start_time = time.time()
    sub_res = {}
    for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:
        sub_res[iter_num] = {}
    purtabed_img = torch.tanh(timg) * boxmul + boxplus
    True_idx = torch.nonzero(label.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    if len(True_idx.shape) != 1:
        True_idx = True_idx.squeeze()
    True_idx= True_idx.numpy().tolist()
    True_idx = set(True_idx)
    
    for iteration in range(1, maxiter + 1):

        
        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)

        purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta
        purtabed_out = model(purtabed_img)
        pert_i = generate_pert_DFU(purtabed_img, model, label, list(True_idx))
        purtabed_img = purtabed_img + torch.from_numpy(pert_i).float().cuda()

        if Projection_flag:
            purtabed_img = (torch.tanh(timg) * boxmul + boxplus + eta).clone().data.cpu()
            purtabed_img = purtabed_img.cuda()
            #purtabed_img = purtabed_img.detach_()
        
        Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)

        # If attack success terminate and return
        if Flag:
            converge_iter = iteration
            attack_success = True
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = topk_acc_metric_1_to_10(tlab.cpu().detach().numpy(), 
                                                                                                        purtabed_out.cpu().detach().numpy())


            end_time = time.time()
            before, after, bafrac = filp_label(model, inputs, purtabed_out)
            iou1, iou2, iou3, iou4, GT_list, pred_idx_l = New_E(label, purtabed_out, k_value)
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:

                if iter_num >= iteration:
                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['bafrac'] = bafrac

                    sub_res[iter_num]['iou1'] = iou1
                    sub_res[iter_num]['iou2'] = iou2
                    sub_res[iter_num]['iou3'] = iou3
                    sub_res[iter_num]['iou4'] = iou4
                    sub_res[iter_num]['GT_list'] = GT_list[:]
                    sub_res[iter_num]['pred_idx_l'] = pred_idx_l[:]

                    if c_list[0]==1:
                        sub_res[iter_num]['index_success_0'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_0'] = None
                    if c_list[1]==1:
                        sub_res[iter_num]['index_success_1'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_1'] = None
                    if c_list[2]==1:
                        sub_res[iter_num]['index_success_2'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_2'] = None
                    if c_list[3]==1:
                        sub_res[iter_num]['index_success_3'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_3'] = None
                    if c_list[4]==1:
                        sub_res[iter_num]['index_success_4'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_4'] = None
                    if c_list[5]==1:
                        sub_res[iter_num]['index_success_5'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size))
                    sub_res[iter_num]['c_list'] = c_list[:]
            break
        else:
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:
                if iter_num == iteration:
                    end_time = time.time()
                    before, after, bafrac = filp_label(model, inputs, purtabed_out)
                    iou1, iou2, iou3, iou4, GT_list, pred_idx_l = New_E(label, purtabed_out, k_value)

                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['bafrac'] = bafrac

                    sub_res[iter_num]['iou1'] = iou1
                    sub_res[iter_num]['iou2'] = iou2
                    sub_res[iter_num]['iou3'] = iou3
                    sub_res[iter_num]['iou4'] = iou4
                    sub_res[iter_num]['GT_list'] = GT_list[:]
                    sub_res[iter_num]['pred_idx_l'] = pred_idx_l[:]

                    sub_res[iter_num]['index_success_0'] = None
                    sub_res[iter_num]['index_success_1'] = None
                    sub_res[iter_num]['index_success_2'] = None
                    sub_res[iter_num]['index_success_3'] = None
                    sub_res[iter_num]['index_success_4'] = None
                    sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = None
                    sub_res[iter_num]['c_list'] = c_list[:]
            
    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    
    return purtabed_img_out, modifier_out, converge_iter, attack_success, c_list, sub_res, iteration


def MIFGSM_T(args, model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, GT, device, lr=3e-2):
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    Allidx = set([i for i in range(label.shape[1])])
    targets_zeros_cuda = label

    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape

    model.eval()
    start_time = time.time()

    purtabed_img = torch.zeros(*shape)
    attack_success = False
    c_list = [0] * 6
    Pd_ori = model(inputs)
    Pd_ori_ge0 = (Pd_ori >= 0).long()
    Pd_idx_ge0 = torch.nonzero(Pd_ori_ge0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()
    Pd_ori_le0 = (Pd_ori < 0).long()
    Pd_idx_le0 = torch.nonzero(Pd_ori_le0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()

    Targetidx = torch.nonzero(targets_zeros_cuda.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    Targetidx = Targetidx.numpy().tolist()
    Targetidx = set(Targetidx)
    Com_cup = Allidx - Targetidx
    Com_cup = list(Com_cup)

    True_idx = torch.nonzero(GT.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    if len(True_idx.shape) != 1:
        True_idx = True_idx.squeeze()
    True_idx= True_idx.numpy().tolist()
    True_idx = set(True_idx)
    np.random.seed(1)
    if len(list(Allidx - Targetidx - True_idx)) > args.Bs:
        B_set = np.random.choice(list(Allidx - Targetidx - True_idx), args.Bs, replace=False).tolist()
    else:
        B_set = list(Allidx - Targetidx - True_idx)

    B_set1 = list(set(B_set) & set(Pd_idx_ge0))
    B_set0 = list(set(B_set) & set(Pd_idx_le0))

    Pd_idx = list(set(Pd_idx_ge0) - set(B_set)) 

    Targetidx = list(Targetidx)
    True_idx = list(True_idx)
    delta = torch.zeros_like(inputs)
    g = torch.zeros_like(inputs)
    delta = torch.nn.Parameter(delta)
    purtabed_img = torch.tanh(delta + timg) * boxmul + boxplus
    purtabed_out = model(purtabed_img)
    
    for iteration in range(1, 40 + 1):
        if delta.grad is not None:
            delta.grad.detach_()
            delta.grad.zero_()

        logprob = torch.nn.functional.logsigmoid(purtabed_out)
        loss = -logprob[0][True_idx].sum() + logprob[0][Targetidx].sum()
        loss.backward()
        g = g + normalize_by_pnorm(delta.grad.data, p=1)

        delta.data += lr * normalize_by_pnorm(g, p=2)
        purtabed_img = torch.tanh(delta + timg) * boxmul + boxplus
        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)
        purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta

        purtabed_out = model(purtabed_img)

        TP_flag, predict_label = TP_index(targets_zeros_cuda.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
        eta_norm = np.linalg.norm(eta.cpu().detach().numpy())

        if TP_flag==True:
            attack_success = True
            c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5] = \
                topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
            break
    
    end_time = time.time()

    Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk = New_E_T(Targetidx, purtabed_out)

    before, after, frac = label_flip_T(Pd_idx, purtabed_out)
    Bs_before, Bs_after, Bs_frac = other_class_influence(B_set1, B_set0, purtabed_out)
    
    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul* 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    return purtabed_img_out, modifier_out, attack_success, c_list, Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk, before, after, frac, Bs_before, Bs_after, Bs_frac, end_time - start_time

def MI_FGSM(model, inputs, label, maxiter, k_value, eps, boxmax, boxmin, device, lr=3e-2):
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    GT_idx = torch.nonzero(label.squeeze() == 1, as_tuple=True)[0].detach().cpu()

    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape

    model.eval()
    start_time = time.time()
    purtabed_img = torch.zeros(*shape)
    attack_success = False
    c_list = [0] * 6
    delta = torch.zeros_like(inputs)
    g = torch.zeros_like(inputs)
    delta = torch.nn.Parameter(delta)
    purtabed_img = torch.tanh(delta + timg) * boxmul + boxplus
    purtabed_out = model(purtabed_img)
    
    for iteration in range(1, 40 + 1):
        if delta.grad is not None:
            delta.grad.detach_()
            delta.grad.zero_()

        logprob = torch.nn.functional.logsigmoid(purtabed_out)
        loss = -logprob[0][GT_idx].sum()
        loss.backward()
        g = g + normalize_by_pnorm(delta.grad.data, p=1)

        delta.data += lr * normalize_by_pnorm(g, p=2)
        purtabed_img = torch.tanh(delta + timg) * boxmul + boxplus
        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)
        purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta

        purtabed_out = model(purtabed_img)

        Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)

        # If attack success terminate and return
        if Flag:
            converge_iter = iteration
            attack_success = True
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = topk_acc_metric_1_to_10(tlab.cpu().detach().numpy(), 
                                                                                                        purtabed_out.cpu().detach().numpy())

    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    end_time = time.time()
    before, after, bafrac = filp_label(model, inputs, purtabed_out)
    iou1, iou2, iou3, iou4, GT_list, pred_idx_l = New_E(label, purtabed_out, k_value)

    return purtabed_img_out, modifier_out, attack_success, c_list, before, after, bafrac, iou1, iou2, iou3, iou4, GT_list, pred_idx_l, end_time - start_time

def FGSM(model, inputs, label, k_value, eps, boxmax, boxmin, device, lr=3e-2):
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    GT_idx = torch.nonzero(label.squeeze() == 1, as_tuple=True)[0].detach().cpu()

    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape

    model.eval()
    start_time = time.time()
    purtabed_img = torch.zeros(*shape)
    attack_success = False
    c_list = [0] * 6

    purtabed_img = torch.tanh(timg) * boxmul + boxplus
    purtabed_img = purtabed_img.clone().cpu().numpy()
    purtabed_img = torch.from_numpy(purtabed_img).cuda()
    purtabed_img.requires_grad = True

    purtabed_out = model(purtabed_img)
    logprob = torch.nn.functional.logsigmoid(purtabed_out)
    loss = -logprob[0][GT_idx].sum()
    loss.backward()
    grad_sign = purtabed_img.grad.detach().sign()
    eta = lr * grad_sign
    eta = clip_eta(eta, 2, const)

    purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta
    purtabed_out = model(purtabed_img)

    Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
    # If attack success terminate and return
    if Flag:
        attack_success = True
        c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
            topk_acc_metric_1_to_10(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())

        print('loss= ', "{}".format(loss), 'attacked: ', Flag, 'predict_label:', predict_label, \
                'GT:', label.cpu().detach().numpy(), 'norm:', "{:.5f}".format(np.linalg.norm(eta.cpu().detach().numpy())))

    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    end_time = time.time()
    before, after, bafrac = filp_label(model, inputs, purtabed_out)
    iou1, iou2, iou3, iou4, GT_list, pred_idx_l = New_E(label, purtabed_out, k_value)

    return purtabed_img_out, modifier_out, attack_success, c_list, before, after, bafrac, iou1, iou2, iou3, iou4, GT_list, pred_idx_l, end_time - start_time

def FGSM_T(args, model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, GT, device, lr=3e-2):
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    Allidx = set([i for i in range(label.shape[1])])
    targets_zeros_cuda = label

    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape

    model.eval()
    start_time = time.time()

    purtabed_img = torch.zeros(*shape)
    attack_success = False
    c_list = [0] * 6
    Pd_ori = model(inputs)
    Pd_ori_ge0 = (Pd_ori >= 0).long()
    Pd_idx_ge0 = torch.nonzero(Pd_ori_ge0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()
    Pd_ori_le0 = (Pd_ori < 0).long()
    Pd_idx_le0 = torch.nonzero(Pd_ori_le0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()

    Targetidx = torch.nonzero(targets_zeros_cuda.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    Targetidx = Targetidx.numpy().tolist()
    Targetidx = set(Targetidx)
    Com_cup = Allidx - Targetidx
    Com_cup = list(Com_cup)

    True_idx = torch.nonzero(GT.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    if len(True_idx.shape) != 1:
        True_idx = True_idx.squeeze()
    True_idx= True_idx.numpy().tolist()
    True_idx = set(True_idx)
    np.random.seed(1)
    if len(list(Allidx - Targetidx - True_idx)) > args.Bs:
        B_set = np.random.choice(list(Allidx - Targetidx - True_idx), args.Bs, replace=False).tolist()
    else:
        B_set = list(Allidx - Targetidx - True_idx)
    B_set1 = list(set(B_set) & set(Pd_idx_ge0))
    B_set0 = list(set(B_set) & set(Pd_idx_le0))

    Pd_idx = list(set(Pd_idx_ge0) - set(B_set)) 
    Targetidx = list(Targetidx)
    True_idx = list(True_idx)

    purtabed_img = torch.tanh(timg) * boxmul + boxplus
    purtabed_img = purtabed_img.clone().cpu().numpy()
    purtabed_img = torch.from_numpy(purtabed_img).cuda()
    purtabed_img.requires_grad = True

    purtabed_out = model(purtabed_img)
    logprob = torch.nn.functional.logsigmoid(purtabed_out)
    loss = -logprob[0][True_idx].sum() + logprob[0][Targetidx].sum()
    loss.backward()
    grad_sign = purtabed_img.grad.detach().sign()
    eta = lr * grad_sign
    eta = clip_eta(eta, 2, const)

    purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta
    purtabed_out = model(purtabed_img)

    TP_flag, predict_label = TP_index(targets_zeros_cuda.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
    eta_norm = np.linalg.norm(eta.cpu().detach().numpy())

    if TP_flag==True:
        attack_success = True
        c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5] = \
            topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
    end_time = time.time()
    Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk = New_E_T(Targetidx, purtabed_out)
    before, after, frac = label_flip_T(Pd_idx, purtabed_out)
    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul* 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    Bs_before, Bs_after, Bs_frac = other_class_influence(B_set1, B_set0, purtabed_out)
    return purtabed_img_out, modifier_out, attack_success, c_list, Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk, before, after, frac, Bs_before, Bs_after, Bs_frac, end_time - start_time


def PGD_T(index_success, args, model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, GT, device, lr=1e-2):
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    Allidx = set([i for i in range(label.shape[1])])
    targets_zeros_cuda = label

    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape
    model.eval()
    best_norm = 1e10
    purtabed_img_out = torch.zeros(*shape)
    modifier_out = torch.zeros(*shape)
    attack_success = False
    c_list = [0] * 6
    Pd_ori = model(inputs)
    Pd_ori_ge0 = (Pd_ori >= 0).long()
    Pd_idx_ge0 = torch.nonzero(Pd_ori_ge0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()
    Pd_ori_le0 = (Pd_ori < 0).long()
    Pd_idx_le0 = torch.nonzero(Pd_ori_le0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()

    Targetidx = torch.nonzero(targets_zeros_cuda.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    Targetidx = Targetidx.numpy().tolist()
    Targetidx = set(Targetidx)
    Com_cup = Allidx - Targetidx
    Com_cup = list(Com_cup)
    True_idx = torch.nonzero(GT.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    if len(True_idx.shape) != 1:
        True_idx = True_idx.squeeze()
    True_idx= True_idx.numpy().tolist()
    True_idx = set(True_idx)
    np.random.seed(1)
    if len(list(Allidx - Targetidx - True_idx)) > args.Bs:
        B_set = np.random.choice(list(Allidx - Targetidx - True_idx), args.Bs, replace=False).tolist()
    else:
        B_set = list(Allidx - Targetidx - True_idx)
    B_set1 = list(set(B_set) & set(Pd_idx_ge0))
    B_set0 = list(set(B_set) & set(Pd_idx_le0))

    Pd_idx = list(set(Pd_idx_ge0) - set(B_set))
    Targetidx = list(Targetidx)
    True_idx = list(True_idx)
    
    start_time = time.time()
    sub_res = {}
    sub_res = {}
    for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
        sub_res[iter_num] = {}
    purtabed_img = (torch.tanh(inputs) * boxmul + boxplus).clone().data.cpu()
    purtabed_img = purtabed_img.cuda()

    for iteration in range(1, maxiter+1):
        purtabed_img.requires_grad = True
        purtabed_out = model(purtabed_img)
        logprob = torch.nn.functional.logsigmoid(purtabed_out)
        loss = -logprob[0][True_idx].sum() + logprob[0][Targetidx].sum()
        loss.backward()

        purtabed_img = purtabed_img + lr * purtabed_img.grad.sign()

        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)
        purtabed_img = (torch.tanh(timg) * boxmul + boxplus + eta).clone().data.cpu()
        purtabed_img = purtabed_img.cuda()

        TP_flag, predict_label = TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
        eta_norm = np.linalg.norm(eta.cpu().detach().numpy())
        if TP_flag==True and eta_norm<best_norm:
            best_norm = eta_norm
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul* 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            attack_success = True

            c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5] = \
                topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
            end_time = time.time()
            Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk = New_E_T(Targetidx, purtabed_out)
            before, after, frac = label_flip_T(Pd_idx, purtabed_out)
            Bs_before, Bs_after, Bs_frac = other_class_influence(B_set1, B_set0, purtabed_out)
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                if iter_num >= iteration:
                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['Eious_iou'] = Eious_iou
                    sub_res[iter_num]['Eious_min_or_GT_max'] = Eious_min_or_GT_max
                    sub_res[iter_num]['target_idx'] = target_idx[:]
                    sub_res[iter_num]['pred_topk'] = pred_topk[:]
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['frac'] = frac
                    sub_res[iter_num]['Bs_before'] = Bs_before
                    sub_res[iter_num]['Bs_after'] = Bs_after
                    sub_res[iter_num]['Bs_frac'] = Bs_frac
                    if c_list[0]==1:
                        sub_res[iter_num]['index_success_0'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_0'] = None
                    if c_list[1]==1:
                        sub_res[iter_num]['index_success_1'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_1'] = None
                    if c_list[2]==1:
                        sub_res[iter_num]['index_success_2'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_2'] = None
                    if c_list[3]==1:
                        sub_res[iter_num]['index_success_3'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_3'] = None
                    if c_list[4]==1:
                        sub_res[iter_num]['index_success_4'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_4'] = None
                    if c_list[5]==1:
                        sub_res[iter_num]['index_success_5'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size))
                    sub_res[iter_num]['c_list'] = c_list[:]
            break
        else:
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                if iter_num == iteration:
                    end_time = time.time()
                    Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk = New_E_T(Targetidx, purtabed_out)
                    before, after, frac = label_flip_T(Pd_idx, purtabed_out)
                    Bs_before, Bs_after, Bs_frac = other_class_influence(B_set1, B_set0, purtabed_out)
                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['Eious_iou'] = Eious_iou
                    sub_res[iter_num]['Eious_min_or_GT_max'] = Eious_min_or_GT_max
                    sub_res[iter_num]['target_idx'] = target_idx[:]
                    sub_res[iter_num]['pred_topk'] = pred_topk[:]
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['frac'] = frac
                    sub_res[iter_num]['Bs_before'] = Bs_before
                    sub_res[iter_num]['Bs_after'] = Bs_after
                    sub_res[iter_num]['Bs_frac'] = Bs_frac

                    sub_res[iter_num]['index_success_0'] = None
                    sub_res[iter_num]['index_success_1'] = None
                    sub_res[iter_num]['index_success_2'] = None
                    sub_res[iter_num]['index_success_3'] = None
                    sub_res[iter_num]['index_success_4'] = None
                    sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = None
                    sub_res[iter_num]['c_list'] = c_list[:]
            

    return purtabed_img_out, modifier_out, attack_success, c_list, sub_res, iteration


def CWU(index, index_success, args, model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, device, Projection_flag = False, lr=1e-2):
    # trick from CW, normalize to [boxin, boxmin]
    model.eval()
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    GT_idx = torch.nonzero(label.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    Pd_ori = model(inputs)
    Pd_ori = (Pd_ori >= 0).long()
    Pd_idx = torch.nonzero(Pd_ori.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    timg = inputs
    tlab = label
    const = eps
    purtabed_img = (torch.tanh(inputs) * boxmul + boxplus).clone().data.cpu()
    purtabed_img = purtabed_img.cuda()
    converge_iter = maxiter
    attack_success = False
    c_list = [0] * 6
    start_time = time.time()
    sub_res = {}
    for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
        sub_res[iter_num] = {}
    loss_data = []
    
    for iteration in range(1, maxiter + 1):
        purtabed_img.requires_grad = True
        purtabed_out = model(purtabed_img)
        loss = torch.sum(torch.clamp(purtabed_out[:,GT_idx], min=0.0))
        loss_data.append(loss.data.item())
        loss.backward()

        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)

        if Projection_flag:
            purtabed_img = (torch.tanh(timg) * boxmul + boxplus + eta).clone().data.cpu()
            purtabed_img = purtabed_img.cuda()
        
        Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)

        # If attack success terminate and return
        if Flag:
            converge_iter = iteration
            attack_success = True
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = topk_acc_metric_1_to_10(tlab.cpu().detach().numpy(), 
                                                                                                        purtabed_out.cpu().detach().numpy())


            end_time = time.time()
            before, after, bafrac = filp_label(model, inputs, purtabed_out)
            iou1, iou2, iou3, iou4, GT_list, pred_idx_l = New_E(label, purtabed_out, k_value)
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                if iter_num >= iteration:
                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['bafrac'] = bafrac

                    sub_res[iter_num]['iou1'] = iou1
                    sub_res[iter_num]['iou2'] = iou2
                    sub_res[iter_num]['iou3'] = iou3
                    sub_res[iter_num]['iou4'] = iou4
                    sub_res[iter_num]['GT_list'] = GT_list[:]
                    sub_res[iter_num]['pred_idx_l'] = pred_idx_l[:]

                    if c_list[0]==1:
                        sub_res[iter_num]['index_success_0'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_0'] = None
                    if c_list[1]==1:
                        sub_res[iter_num]['index_success_1'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_1'] = None
                    if c_list[2]==1:
                        sub_res[iter_num]['index_success_2'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_2'] = None
                    if c_list[3]==1:
                        sub_res[iter_num]['index_success_3'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_3'] = None
                    if c_list[4]==1:
                        sub_res[iter_num]['index_success_4'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_4'] = None
                    if c_list[5]==1:
                        sub_res[iter_num]['index_success_5'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size))
                    sub_res[iter_num]['c_list'] = c_list[:]
            # break
        else:
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                if iter_num == iteration:
                    end_time = time.time()
                    before, after, bafrac = filp_label(model, inputs, purtabed_out)
                    iou1, iou2, iou3, iou4, GT_list, pred_idx_l = New_E(label, purtabed_out, k_value)

                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['bafrac'] = bafrac

                    sub_res[iter_num]['iou1'] = iou1
                    sub_res[iter_num]['iou2'] = iou2
                    sub_res[iter_num]['iou3'] = iou3
                    sub_res[iter_num]['iou4'] = iou4
                    sub_res[iter_num]['GT_list'] = GT_list[:]
                    sub_res[iter_num]['pred_idx_l'] = pred_idx_l[:]

                    sub_res[iter_num]['index_success_0'] = None
                    sub_res[iter_num]['index_success_1'] = None
                    sub_res[iter_num]['index_success_2'] = None
                    sub_res[iter_num]['index_success_3'] = None
                    sub_res[iter_num]['index_success_4'] = None
                    sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = None
                    sub_res[iter_num]['c_list'] = c_list[:]
            
    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    
    return purtabed_img_out, modifier_out, converge_iter, attack_success, c_list, sub_res, iteration, loss_data

def PGD(index, index_success, args, model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, device, Projection_flag = False, lr=1e-2):
    # trick from CW, normalize to [boxin, boxmin]
    model.eval()
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    GT_idx = torch.nonzero(label.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    Pd_ori = model(inputs)
    Pd_ori = (Pd_ori >= 0).long()
    Pd_idx = torch.nonzero(Pd_ori.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    timg = inputs
    tlab = label
    const = eps
    purtabed_img = (torch.tanh(inputs) * boxmul + boxplus).clone().data.cpu()
    purtabed_img = purtabed_img.cuda()
    converge_iter = maxiter
    attack_success = False
    c_list = [0] * 6
    start_time = time.time()
    sub_res = {}
    for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
        sub_res[iter_num] = {}
    loss_data = []
    
    for iteration in range(1, maxiter + 1):
        purtabed_img.requires_grad = True
        purtabed_out = model(purtabed_img)
        logprob = torch.nn.functional.logsigmoid(purtabed_out)
        loss = -logprob[0][GT_idx].sum()
        loss_data.append(loss.data.item())
        loss.backward()

        purtabed_img = purtabed_img + lr * purtabed_img.grad.sign()

        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)

        if Projection_flag:
            purtabed_img = (torch.tanh(timg) * boxmul + boxplus + eta).clone().data.cpu()
            purtabed_img = purtabed_img.cuda()

        Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)

        if Flag:
            converge_iter = iteration
            attack_success = True
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = topk_acc_metric_1_to_10(tlab.cpu().detach().numpy(), 
                                                                                                        purtabed_out.cpu().detach().numpy())


            end_time = time.time()
            before, after, bafrac = filp_label(model, inputs, purtabed_out)
            iou1, iou2, iou3, iou4, GT_list, pred_idx_l = New_E(label, purtabed_out, k_value)
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                if iter_num >= iteration:
                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['bafrac'] = bafrac

                    sub_res[iter_num]['iou1'] = iou1
                    sub_res[iter_num]['iou2'] = iou2
                    sub_res[iter_num]['iou3'] = iou3
                    sub_res[iter_num]['iou4'] = iou4
                    sub_res[iter_num]['GT_list'] = GT_list[:]
                    sub_res[iter_num]['pred_idx_l'] = pred_idx_l[:]

                    if c_list[0]==1:
                        sub_res[iter_num]['index_success_0'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_0'] = None
                    if c_list[1]==1:
                        sub_res[iter_num]['index_success_1'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_1'] = None
                    if c_list[2]==1:
                        sub_res[iter_num]['index_success_2'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_2'] = None
                    if c_list[3]==1:
                        sub_res[iter_num]['index_success_3'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_3'] = None
                    if c_list[4]==1:
                        sub_res[iter_num]['index_success_4'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_4'] = None
                    if c_list[5]==1:
                        sub_res[iter_num]['index_success_5'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size))
                    sub_res[iter_num]['c_list'] = c_list[:]
            # break
        else:
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                if iter_num == iteration:
                    end_time = time.time()
                    before, after, bafrac = filp_label(model, inputs, purtabed_out)
                    iou1, iou2, iou3, iou4, GT_list, pred_idx_l = New_E(label, purtabed_out, k_value)

                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['bafrac'] = bafrac

                    sub_res[iter_num]['iou1'] = iou1
                    sub_res[iter_num]['iou2'] = iou2
                    sub_res[iter_num]['iou3'] = iou3
                    sub_res[iter_num]['iou4'] = iou4
                    sub_res[iter_num]['GT_list'] = GT_list[:]
                    sub_res[iter_num]['pred_idx_l'] = pred_idx_l[:]

                    sub_res[iter_num]['index_success_0'] = None
                    sub_res[iter_num]['index_success_1'] = None
                    sub_res[iter_num]['index_success_2'] = None
                    sub_res[iter_num]['index_success_3'] = None
                    sub_res[iter_num]['index_success_4'] = None
                    sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = None
                    sub_res[iter_num]['c_list'] = c_list[:]
            
    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    
    return purtabed_img_out, modifier_out, converge_iter, attack_success, c_list, sub_res, iteration, loss_data

def get_jacobian(model, x, noutputs):
    num_instaces = x.size()[0]
    v = torch.eye(noutputs).cuda()
    jac = []
    x = x.clone().data.cpu()
    
    if torch.cuda.is_available():
        x = x.cuda()
    x.requires_grad = True
    y = torch.nn.functional.sigmoid(model(x)) 
    retain_graph = True
    for i in range(noutputs):
        if i == noutputs - 1:
            retain_graph = False
        y.backward(torch.unsqueeze(v[i], 0).repeat(num_instaces, 1), retain_graph=retain_graph)
        g = x.grad.cpu().detach().numpy()
        x.grad.zero_()
        jac.append(g)
        
    jac = np.asarray(jac)
    y = y.cpu().detach().numpy()
    return jac, y

def generate_pert_DF(x, model, y_target, Target_idx, True_idx):

    num_labels = y_target.shape[1]
    x_shape = np.shape(x)[1:]
    y_target = y_target.detach().cpu().numpy()
    nb_labels = num_labels

    w = np.squeeze(np.zeros(x.shape[1:]))  # same shape as original image
    r_tot = np.zeros(x.shape)
    gradients, output = get_jacobian(model, x, num_labels)
    gradients = np.asarray(gradients)
    gradients = gradients.swapaxes(1, 0)
    predictions_val = output

    for idx in range(np.shape(x)[0]):
        f = predictions_val[idx]
        f_true = -f[True_idx]
        f_tar = f[Target_idx]
        c = np.array([0.5] * len(True_idx) + [-0.5] * len(Target_idx)) + np.concatenate([f_true, f_tar])
        w = gradients[idx].reshape(nb_labels, -1)
        w_true = w[True_idx]
        w_tar = -w[Target_idx]
        P = np.concatenate([w_true, w_tar], axis = 0).T
    
        q = np.reshape(c, (-1, 1))
        
        temp = np.matmul(P.T, P)
        zeros = np.zeros(temp.shape[1])
        delete_idx = []
        for j in range(temp.shape[0]):
            if np.all(temp[j] == zeros):
                delete_idx.append(j)
        P = np.delete(P, delete_idx, axis=1)
        q = np.delete(q, delete_idx, axis=0)
        try:
            delta_r = np.matmul(np.matmul(P, np.linalg.inv(np.matmul(P.T, P))), q)
        except:
            continue
        delta_r = np.reshape(delta_r, x_shape)

        r_tot[idx] = delta_r
    return r_tot

def DF(index_success, args, model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, GT, device, lr=1e-2):
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    Allidx = set([i for i in range(label.shape[1])])
    targets_zeros_cuda = label

    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape

    model.eval()

    best_norm = 1e10
    purtabed_img_out = torch.zeros(*shape)
    modifier_out = torch.zeros(*shape)
    attack_success = False
    c_list = [0] * 6
    Pd_ori = model(inputs)
    Pd_ori_ge0 = (Pd_ori >= 0).long()
    Pd_idx_ge0 = torch.nonzero(Pd_ori_ge0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()
    Pd_ori_le0 = (Pd_ori < 0).long()
    Pd_idx_le0 = torch.nonzero(Pd_ori_le0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()

    Targetidx = torch.nonzero(targets_zeros_cuda.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    Targetidx = Targetidx.numpy().tolist()
    Targetidx = set(Targetidx)
    Com_cup = Allidx - Targetidx
    Com_cup = list(Com_cup)
    True_idx = torch.nonzero(GT.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    if len(True_idx.shape) != 1:
        True_idx = True_idx.squeeze()
    True_idx= True_idx.numpy().tolist()
    True_idx = set(True_idx)

    np.random.seed(1)
    if len(list(Allidx - Targetidx - True_idx)) > args.Bs:
        B_set = np.random.choice(list(Allidx - Targetidx - True_idx), args.Bs, replace=False).tolist()
    else:
        B_set = list(Allidx - Targetidx - True_idx)
    B_set1 = list(set(B_set) & set(Pd_idx_ge0))
    B_set0 = list(set(B_set) & set(Pd_idx_le0))

    Pd_idx = list(set(Pd_idx_ge0) - set(B_set))
    Targetidx = list(Targetidx)
    start_time = time.time()
    sub_res = {}
    sub_res = {}
    purtabed_img = torch.tanh(timg) * boxmul + boxplus
    for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:#, 300, 400, 500
        sub_res[iter_num] = {}

    for iteration in range(1, maxiter+1):
        
        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)

        purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta
        purtabed_out = model(purtabed_img)
        pert_i = generate_pert_DF(purtabed_img, model, label, Targetidx, list(True_idx))
        purtabed_img = purtabed_img + torch.from_numpy(pert_i).float().cuda()

        TP_flag, predict_label = TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
        eta_norm = np.linalg.norm(eta.cpu().detach().numpy())
        if TP_flag==True and eta_norm<best_norm:
            best_norm = eta_norm
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul* 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            attack_success = True

            c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5] = \
                topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
            end_time = time.time()
            Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk = New_E_T(Targetidx, purtabed_out)
            before, after, frac = label_flip_T(Pd_idx, purtabed_out)
            Bs_before, Bs_after, Bs_frac = other_class_influence(B_set1, B_set0, purtabed_out)
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:#, 300, 400, 500
                if iter_num >= iteration:
                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['Eious_iou'] = Eious_iou
                    sub_res[iter_num]['Eious_min_or_GT_max'] = Eious_min_or_GT_max
                    sub_res[iter_num]['target_idx'] = target_idx[:]
                    sub_res[iter_num]['pred_topk'] = pred_topk[:]
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['frac'] = frac
                    sub_res[iter_num]['Bs_before'] = Bs_before
                    sub_res[iter_num]['Bs_after'] = Bs_after
                    sub_res[iter_num]['Bs_frac'] = Bs_frac
                    norm = np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size))
                    sub_res[iter_num]['norm'] = None if np.isnan(norm) else norm
                    
                    if c_list[0]==1 and not np.isnan(norm):
                        sub_res[iter_num]['index_success_0'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_0'] = None
                        
                    if c_list[1]==1 and not np.isnan(norm):
                        sub_res[iter_num]['index_success_1'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_1'] = None
                        
                    if c_list[2]==1 and not np.isnan(norm):
                        sub_res[iter_num]['index_success_2'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_2'] = None
                        
                    if c_list[3]==1 and not np.isnan(norm):
                        sub_res[iter_num]['index_success_3'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_3'] = None
                        
                    if c_list[4]==1 and not np.isnan(norm):
                        sub_res[iter_num]['index_success_4'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_4'] = None
                        
                    if c_list[5]==1 and not np.isnan(norm):
                        sub_res[iter_num]['index_success_5'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_5'] = None
                    
                    sub_res[iter_num]['c_list'] = c_list[:]
            break
        else:
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:#, 300, 400, 500
                if iter_num == iteration:
                    end_time = time.time()
                    Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk = New_E_T(Targetidx, purtabed_out)
                    before, after, frac = label_flip_T(Pd_idx, purtabed_out)
                    Bs_before, Bs_after, Bs_frac = other_class_influence(B_set1, B_set0, purtabed_out)
                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['Eious_iou'] = Eious_iou
                    sub_res[iter_num]['Eious_min_or_GT_max'] = Eious_min_or_GT_max
                    sub_res[iter_num]['target_idx'] = target_idx[:]
                    sub_res[iter_num]['pred_topk'] = pred_topk[:]
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['frac'] = frac
                    sub_res[iter_num]['Bs_before'] = Bs_before
                    sub_res[iter_num]['Bs_after'] = Bs_after
                    sub_res[iter_num]['Bs_frac'] = Bs_frac

                    sub_res[iter_num]['index_success_0'] = None
                    sub_res[iter_num]['index_success_1'] = None
                    sub_res[iter_num]['index_success_2'] = None
                    sub_res[iter_num]['index_success_3'] = None
                    sub_res[iter_num]['index_success_4'] = None
                    sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = None
                    sub_res[iter_num]['c_list'] = c_list[:]
            
    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul* 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    norm_i = np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size))
    return purtabed_img_out, modifier_out, attack_success, c_list, sub_res, iteration, norm_i

def rank2(index_success, args, model, inputs, label, k_value, eps, maxiter, boxmax, boxmin, GT, device, lr=1e-2):
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0
    Allidx = set([i for i in range(label.shape[1])])
    targets_zeros_cuda = label

    timg = inputs
    tlab = label
    const = eps
    shape = inputs.shape

    # variables we are going to optimize over
    if device.type == 'cuda':
        modifier = Variable(torch.zeros(*shape).cuda(), requires_grad=True).to(device)
    else:
        modifier = Variable(torch.zeros(*shape), requires_grad=True).to(device)
    model.eval()

    optimizer = torch.optim.SGD([{'params': modifier}], lr=lr)

    best_norm = 1e10
    purtabed_img_out = torch.zeros(*shape)
    modifier_out = torch.zeros(*shape)
    attack_success = False
    c_list = [0] * 6
    Pd_ori = model(inputs)
    Pd_ori_ge0 = (Pd_ori >= 0).long()
    Pd_idx_ge0 = torch.nonzero(Pd_ori_ge0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()
    Pd_ori_le0 = (Pd_ori < 0).long()
    Pd_idx_le0 = torch.nonzero(Pd_ori_le0.squeeze() == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()

    Targetidx = torch.nonzero(targets_zeros_cuda.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    Targetidx = Targetidx.numpy().tolist()
    Targetidx = set(Targetidx)
    Com_cup = Allidx - Targetidx
    Com_cup = list(Com_cup)

    True_idx = torch.nonzero(GT.squeeze() == 1, as_tuple=True)[0].detach().cpu()
    if len(True_idx.shape) != 1:
        True_idx = True_idx.squeeze()
    True_idx= True_idx.numpy().tolist()
    True_idx = set(True_idx)
    np.random.seed(1)
    if len(list(Allidx - Targetidx - True_idx)) > args.Bs:
        B_set = np.random.choice(list(Allidx - Targetidx - True_idx), args.Bs, replace=False).tolist()
    else:
        B_set = list(Allidx - Targetidx - True_idx)
    B_set1 = list(set(B_set) & set(Pd_idx_ge0))
    B_set0 = list(set(B_set) & set(Pd_idx_le0))

    Pd_idx = list(set(Pd_idx_ge0) - set(B_set))
    Targetidx = list(Targetidx)
    start_time = time.time()
    sub_res = {}
    sub_res = {}
    for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
        sub_res[iter_num] = {}

    for iteration in range(1, maxiter+1):
        optimizer.zero_grad()

        purtabed_img = torch.tanh(modifier + timg) * boxmul + boxplus
        eta = purtabed_img - (torch.tanh(timg) * boxmul + boxplus)
        eta = clip_eta(eta, 2, const)

        purtabed_img = torch.tanh(timg) * boxmul + boxplus + eta
        purtabed_out = model(purtabed_img)

        real = torch.max((1 - tlab) * purtabed_out - tlab * 10000)
        other = torch.min(tlab * purtabed_out + (1 - tlab) * 10000)
        loss = torch.sum(torch.clamp(real - other, min=0.0))

        # Calculate gradient
        loss.backward()
        optimizer.step()

        TP_flag, predict_label = TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
        eta_norm = np.linalg.norm(eta.cpu().detach().numpy())
        if TP_flag==True and eta_norm<best_norm:
            best_norm = eta_norm
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul* 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            attack_success = True

            c_list[0],c_list[1],c_list[2],c_list[3],c_list[4],c_list[5] = \
                topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy())
            end_time = time.time()
            Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk = New_E_T(Targetidx, purtabed_out)
            before, after, frac = label_flip_T(Pd_idx, purtabed_out)
            Bs_before, Bs_after, Bs_frac = other_class_influence(B_set1, B_set0, purtabed_out)
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                if iter_num >= iteration:
                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['Eious_iou'] = Eious_iou
                    sub_res[iter_num]['Eious_min_or_GT_max'] = Eious_min_or_GT_max
                    sub_res[iter_num]['target_idx'] = target_idx[:]
                    sub_res[iter_num]['pred_topk'] = pred_topk[:]
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['frac'] = frac
                    sub_res[iter_num]['Bs_before'] = Bs_before
                    sub_res[iter_num]['Bs_after'] = Bs_after
                    sub_res[iter_num]['Bs_frac'] = Bs_frac
                    if c_list[0]==1:
                        sub_res[iter_num]['index_success_0'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_0'] = None
                    if c_list[1]==1:
                        sub_res[iter_num]['index_success_1'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_1'] = None
                    if c_list[2]==1:
                        sub_res[iter_num]['index_success_2'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_2'] = None
                    if c_list[3]==1:
                        sub_res[iter_num]['index_success_3'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_3'] = None
                    if c_list[4]==1:
                        sub_res[iter_num]['index_success_4'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_4'] = None
                    if c_list[5]==1:
                        sub_res[iter_num]['index_success_5'] = index_success[iter_num]
                    else:
                        sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size))
                    sub_res[iter_num]['c_list'] = c_list[:]
            break
        else:
            for iter_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                if iter_num == iteration:
                    end_time = time.time()
                    Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk = New_E_T(Targetidx, purtabed_out)
                    before, after, frac = label_flip_T(Pd_idx, purtabed_out)
                    Bs_before, Bs_after, Bs_frac = other_class_influence(B_set1, B_set0, purtabed_out)
                    sub_res[iter_num]['cost_time'] = end_time - start_time
                    sub_res[iter_num]['Eious_iou'] = Eious_iou
                    sub_res[iter_num]['Eious_min_or_GT_max'] = Eious_min_or_GT_max
                    sub_res[iter_num]['target_idx'] = target_idx[:]
                    sub_res[iter_num]['pred_topk'] = pred_topk[:]
                    sub_res[iter_num]['before'] = before
                    sub_res[iter_num]['after'] = after
                    sub_res[iter_num]['frac'] = frac
                    sub_res[iter_num]['Bs_before'] = Bs_before
                    sub_res[iter_num]['Bs_after'] = Bs_after
                    sub_res[iter_num]['Bs_frac'] = Bs_frac

                    sub_res[iter_num]['index_success_0'] = None
                    sub_res[iter_num]['index_success_1'] = None
                    sub_res[iter_num]['index_success_2'] = None
                    sub_res[iter_num]['index_success_3'] = None
                    sub_res[iter_num]['index_success_4'] = None
                    sub_res[iter_num]['index_success_5'] = None
                    sub_res[iter_num]['norm'] = None
                    sub_res[iter_num]['c_list'] = c_list[:]
            

    return purtabed_img_out, modifier_out, attack_success, c_list, sub_res, iteration

def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives

def kfool(model, inputs, GT, k_value, boxmax, boxmin,maxiter, device, lr=1e-2):

    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = GT
    shape = inputs.shape
    purtabed_img_out = np.zeros(shape)
    modifier_out = np.zeros(shape)

    all_list = set(list(range(0, tlab.cpu().detach().numpy().shape[1])))
    with torch.no_grad():
        F = model(torch.tanh(timg) * boxmul + boxplus)
    nb_classes = F.size(-1)

    purtabed_img = (torch.tanh(timg) * boxmul + boxplus).clone().requires_grad_()

    loop_i = 0
    attack_success = False
    c_list = [0] * 6
    F = model(purtabed_img)
    max_label = torch.argmax(tlab * F - (1 - tlab) * 10000)
    p = torch.argsort(F, dim=1, descending=True)
    tlab_all = ((tlab == 1).nonzero(as_tuple=True)[1]).cpu().detach().numpy()
    complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())

    r_tot = torch.zeros(timg.size()).to(device)

    while (complement_set.issuperset(set(tlab_all))) == False and loop_i < maxiter:
        w = torch.squeeze(torch.zeros(timg.size()[1:])).to(device)
        f = 0
        top_F = F.topk(k_value + 1)[0]
        max_F = F[0][max_label].reshape((1,1))
        gradients_top = torch.stack(jacobian(top_F, purtabed_img, k_value + 1), dim=1)
        gradients_max = torch.stack(jacobian(max_F, purtabed_img, 1), dim=1)
        # gradients = torch.stack(jacobian(F, purtabed_img, nb_classes), dim=1)
        with torch.no_grad():
            for idx in range(inputs.size(0)):
                for k in range(k_value + 1):
                    if torch.all(torch.eq(gradients_top[idx, k, ...], gradients_max[idx,0,...]))==False and p[0][k]!=max_label:
                        norm = torch.div(1, torch.norm(gradients_top[idx, k, ...] - gradients_max[idx,0,...]))
                        w = w + (gradients_top[idx, k, ...] - gradients_max[idx,0,...]) * norm
                        f = f + (F[idx, p[0][k]] - F[idx, max_label]) * norm
                r_tot[idx, ...] = r_tot[idx, ...] + torch.abs(f) * w / torch.norm(w)
        purtabed_img = (torch.tanh(r_tot + timg) * boxmul + boxplus).requires_grad_()
        F = model(purtabed_img)
        p = torch.argsort(F, dim=1, descending=True)

        # complement_set = all_list - set(p[:k_value])
        complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())
        # Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
        if complement_set.issuperset(set(tlab_all)):
            Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
            purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
            modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
            attack_success = True
            c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
                topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), F.cpu().detach().numpy())
            print('iter:', loop_i + 1, \
                  'attacked: ', attack_success, \
                  'predict_label:', predict_label, \
                  'GT:', tlab.cpu().detach().numpy(), \
                  'min:', "{:.5f}".format(r_tot.min().cpu()), \
                  'max:', "{:.5f}".format(r_tot.max().cpu()), \
                  'norm:', "{:.5f}".format(np.linalg.norm(r_tot.cpu().detach().numpy())))
        loop_i = loop_i + 1
    
    return purtabed_img_out, modifier_out, attack_success, c_list

def kUAP(args, model, device, val_loader, sample_list, train_index_end):
    global_delta = torch.zeros(iter(val_loader).next()[0].shape).to(device)
    fooling_rate = 0
    itr = 0
    boxmul = (args.boxmax - args.boxmin) / 2.
    boxplus = (args.boxmin + args.boxmax) / 2.
    while fooling_rate < args.ufr_lower_bound and itr < args.max_iter_uni:
        print('Starting pass number ', itr)
        index1 =0
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:train_index_end]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                print('ith:', index1 + 1)
                GT = GT.int()
                print('\n')
                data, GT = data.to(device), GT.to(device)
                pred = model(torch.tanh(data + global_delta) * boxmul + boxplus)
                Flag = bool(topk_acc_metric(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value))
                if Flag == False:
                    purtabed_img_out, modifiered, attack_success, c_list = \
                        kfool(model, data, GT, args.k_value, args.boxmax, args.boxmin, args.maxiter, device=device,
                              lr=args.lr_attack)
                    if attack_success == True:
                        global_delta = global_delta + torch.from_numpy(modifiered).to(device)
                        print('pre_global_delta:', np.linalg.norm(global_delta.cpu().detach().numpy()))
                        print(args.uap_norm, args.uap_eps)
                        global_delta = clip_eta(global_delta, args.uap_norm, args.uap_eps)
                        print('after_global_delta:', np.linalg.norm(global_delta.cpu().detach().numpy()))
                else:
                    print('attack had successed')

                index1 = index1 + 1
            if index1 == train_index_end:
                break
        itr = itr + 1
        count = 0
        data_num = 0
        index2 = 0
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:train_index_end]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                data, GT = data.to(device), GT.to(device)
                data_num = data_num + 1
                pred = model(torch.tanh(data + global_delta) * boxmul + boxplus)
                count = count + topk_acc_metric(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value)
                index2 = index2 + 1
            if index2 == train_index_end:
                break

        fooling_rate = count / data_num
        print('FOOLING RATE = ', fooling_rate)
    return global_delta

def save_result_FGSMT(success_c_list, index, args, success_img_index, success_modifier_norm_list, success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3,success_modifier_norm_index_4, success_modifier_norm_index_5,\
                success_blurring_index_1, success_blurring_index_2, success_blurring_index_3, Eious_oris_T, Eious_min_or_GT_maxs_T, Target_idxs_T, Pred_Topks_T,\
                Befores_T, Afters_T, Label_Fracs_T, Bs_befores_T, Bs_afters_T, Bs_fracs_T, times_costs):

    final_all = []
    final_all.append(success_modifier_norm_list)
    final_all.append(success_img_index)
    final_all.append(success_c_list)
    final_all.append(success_modifier_norm_index_0)
    final_all.append(success_modifier_norm_index_1)
    final_all.append(success_modifier_norm_index_2)
    final_all.append(success_modifier_norm_index_3)
    final_all.append(success_modifier_norm_index_4)
    final_all.append(success_modifier_norm_index_5)
    final_all.append(success_blurring_index_1)
    final_all.append(success_blurring_index_2)
    final_all.append(success_blurring_index_3)
    final_all.append(Eious_oris_T)
    final_all.append(Eious_min_or_GT_maxs_T)
    final_all.append(Target_idxs_T)
    final_all.append(Pred_Topks_T)
    final_all.append(Befores_T)
    final_all.append(Afters_T)
    final_all.append(Bs_befores_T)
    final_all.append(Bs_afters_T)
    final_all.append(Bs_fracs_T)
    final_all.append(times_costs)
    with open(args.myresults, 'a+') as f:
        f.write('k is :' + str(args.k_value) + '\n')
        
    c_list_sum = np.sum(np.asarray(success_c_list), 0) / index
    if success_c_list != []:
        print('attack_type= ', "{}".format(args.app), 'label_difficult= ', "{}".format(args.label_difficult))
        print('FR_1= ', "{:.5f}".format(c_list_sum[0]), \
              'FR_2= ', "{:.5f}".format(c_list_sum[1]), \
              'FR_3= ', "{:.5f}".format(c_list_sum[2]), \
              'FR_4= ', "{:.5f}".format(c_list_sum[3]), \
              'FR_5= ', "{:.5f}".format(c_list_sum[4]), \
              'FR_10= ', "{:.5f}".format(c_list_sum[5]))
        print('avg_norm_1= ', "{}".format(
            np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))), \
              'avg_norm_2= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))), \
              'avg_norm_3= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))), \
              'avg_norm_4= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))), \
              'avg_norm_5= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))), \
              'avg_norm_10= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))))
        print('')
        print('Eious_oris_T is :', 1 - sum(Eious_oris_T) / len(Eious_oris_T))
        print('Eious_min_or_GT_maxs_T is :', 1 - sum(Eious_min_or_GT_maxs_T) / len(Eious_min_or_GT_maxs_T))
        print('Label flip rate is :', 1 - sum(Label_Fracs_T) / len(Label_Fracs_T))
        print('Label keep same Rate is :', 1 - sum(Bs_fracs_T) / len(Bs_fracs_T))

        with open(args.myresults, 'a+') as f:
            f.write('attack type is :' + str(args.app)+ '\n')
            f.write('label difficult is :' + str(args.label_difficult) + '\n')
            f.write('k is :' + str(args.k_value) + '\n')
            
            f.write('FR_1 is :' + str("{:.5f}".format(c_list_sum[0])) + '\n')
            f.write('FR_2 is :' + str("{:.5f}".format(c_list_sum[1])) + '\n')
            f.write('FR_3 is :' + str("{:.5f}".format(c_list_sum[2])) + '\n')
            f.write('FR_4 is :' + str("{:.5f}".format(c_list_sum[3])) + '\n')
            f.write('FR_5 is :' + str("{:.5f}".format(c_list_sum[4])) + '\n')
            f.write('FR_10 is :' + str("{:.5f}".format(c_list_sum[5])) + '\n')

            avg_norm_1 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))
            avg_norm_2 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))
            avg_norm_3 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))
            avg_norm_4 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))
            avg_norm_5 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))
            avg_norm_10 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))

            f.write('avg_norm_1 is :' + str(avg_norm_1 * 255) + '\n')
            f.write('avg_norm_2 is :' + str(avg_norm_2 * 255) + '\n')
            f.write('avg_norm_3 is :' + str(avg_norm_3 * 255) + '\n')
            f.write('avg_norm_4 is :' + str(avg_norm_4 * 255) + '\n')
            f.write('avg_norm_5 is :' + str(avg_norm_5 * 255) + '\n')
            f.write('avg_norm_10 is :' + str(avg_norm_10 * 255) + '\n')

            
    else:
        print('All False')

    if success_blurring_index_1!= []:
        success_blurring_ratio_1 = len(success_blurring_index_1)/ index
    else:
        success_blurring_ratio_1 = 0
    if success_blurring_index_2 != []:
        success_blurring_ratio_2 = len(success_blurring_index_2) / index
    else:
        success_blurring_ratio_2 = 0
    if success_blurring_index_3 != []:
        success_blurring_ratio_3 = len(success_blurring_index_3) / index
    else:
        success_blurring_ratio_3 = 0
    print('blurring_1_ratio = {:.5f}'.format(success_blurring_ratio_1),\
          'blurring_2_ratio = {:.5f}'.format(success_blurring_ratio_2),\
          'blurring_3_ratio = {:.5f}'.format(success_blurring_ratio_3))

    with open(args.myresults, 'a+') as f:
        f.write('Eious_oris_T is :' + str(sum(Eious_oris_T) / len(Eious_oris_T)) + '\n')
        f.write('Eious_min_or_GT_maxs_T is :' + str(sum(Eious_min_or_GT_maxs_T) / len(Eious_min_or_GT_maxs_T)) + '\n')
        f.write('Label flip rate is :' + str(sum(Label_Fracs_T) / len(Label_Fracs_T)) + '\n')
        f.write('Label keep same Rate is :' + str(1 - sum(Bs_fracs_T) / len(Bs_fracs_T)) + '\n')

        f.write('Total time cost is :' + str(sum(times_costs)) + '\n')
        f.write('Average time cost is :' + str(sum(times_costs) / len(times_costs)) + '\n')
        
        f.write('blurring_1_ratio is :' + str(success_blurring_ratio_1)+ '\n')
        f.write('blurring_2_ratio is :' + str(success_blurring_ratio_2)+ '\n')
        f.write('blurring_3_ratio is :' + str(success_blurring_ratio_3)+ '\n')

        f.write('\n')
        f.write('\n')
        f.write('\n')

    os.makedirs('./result/{}/{}/{}/eps_{}'.format(args.dataset,args.label_difficult, args.app, args.eps), exist_ok=True)
    final_all = np.asarray(final_all, dtype=object)
    np.save('./result/{}/{}/{}/eps_{}/k_{}'.format(args.dataset,args.label_difficult, args.app, args.eps, args.k_value), final_all)

def save_result_FGSM(success_c_list, 
                    index, 
                    args, 
                    success_img_index, 
                    success_modifier_norm_list, 
                    success_modifier_norm_index_0, 
                    success_modifier_norm_index_1,
                    success_modifier_norm_index_2, 
                    success_modifier_norm_index_3,
                    success_modifier_norm_index_4, 
                    success_modifier_norm_index_5,
                    success_blurring_index_1, 
                    success_blurring_index_2, 
                    success_blurring_index_3, 
                    Eious1, Eious2, Eious3, Eious4,\
                    bafracs, befores, afters, GT_lists,
                    pred_idx_ls, times_costs):

    final_all = []
    with open(args.myresults, 'a+') as f:
        f.write('k is :' + str(args.k_value) + '\n')
        
    c_list_sum = np.sum(np.asarray(success_c_list), 0) / index
    if success_c_list != []:
        print('attack_type= ', "{}".format(args.app), 'label_difficult= ', "{}".format(args.label_difficult))
        print('FR_1= ', "{:.5f}".format(c_list_sum[0]), \
              'FR_2= ', "{:.5f}".format(c_list_sum[1]), \
              'FR_3= ', "{:.5f}".format(c_list_sum[2]), \
              'FR_4= ', "{:.5f}".format(c_list_sum[3]), \
              'FR_5= ', "{:.5f}".format(c_list_sum[4]), \
              'FR_10= ', "{:.5f}".format(c_list_sum[5]))

        print('avg_norm_1= ', "{}".format(
            np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))), \
              'avg_norm_2= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))), \
              'avg_norm_3= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))), \
              'avg_norm_4= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))), \
              'avg_norm_5= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))), \
              'avg_norm_10= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))))

        print('Eiou ori is :', 1 - sum(Eious1) / len(Eious1))
        print('Eiou min is :', 1 - sum(Eious2) / len(Eious2))
        print('Eiou max is :', 1 - sum(Eious3) / len(Eious3))
        print('Eiou GT_list is :', 1 - sum(Eious4) / len(Eious4))
        print('labels flip rate is :', 1 - sum(bafracs) / len(bafracs))
        print('total time cost is :', sum(times_costs))
        print('average time cost is :', sum(times_costs) / len(times_costs))

        with open(args.myresults, 'a+') as f:
            f.write('attack type is :' + str(args.app)+ '\n')
            f.write('label difficult is :' + str(args.label_difficult) + '\n')
            f.write('k is :' + str(args.k_value) + '\n')

            f.write('FR_1 is :' + str("{:.5f}".format(c_list_sum[0])) + '\n')
            f.write('FR_2 is :' + str("{:.5f}".format(c_list_sum[1])) + '\n')
            f.write('FR_3 is :' + str("{:.5f}".format(c_list_sum[2])) + '\n')
            f.write('FR_4 is :' + str("{:.5f}".format(c_list_sum[3])) + '\n')
            f.write('FR_5 is :' + str("{:.5f}".format(c_list_sum[4])) + '\n')
            f.write('FR_10 is :' + str("{:.5f}".format(c_list_sum[5])) + '\n')

            avg_norm_1 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))
            avg_norm_2 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))
            avg_norm_3 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))
            avg_norm_4 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))
            avg_norm_5 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))
            avg_norm_10 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))

            f.write('avg_norm_1 is :' + str(avg_norm_1 * 255) + '\n')
            f.write('avg_norm_2 is :' + str(avg_norm_2 * 255) + '\n')
            f.write('avg_norm_3 is :' + str(avg_norm_3 * 255) + '\n')
            f.write('avg_norm_4 is :' + str(avg_norm_4 * 255) + '\n')
            f.write('avg_norm_5 is :' + str(avg_norm_5 * 255) + '\n')
            f.write('avg_norm_10 is :' + str(avg_norm_10 * 255) + '\n')

            
    else:
        print('All False')

    if success_blurring_index_1!= []:
        success_blurring_ratio_1 = len(success_blurring_index_1)/ index
    else:
        success_blurring_ratio_1 = 0
    if success_blurring_index_2 != []:
        success_blurring_ratio_2 = len(success_blurring_index_2) / index
    else:
        success_blurring_ratio_2 = 0
    if success_blurring_index_3 != []:
        success_blurring_ratio_3 = len(success_blurring_index_3) / index
    else:
        success_blurring_ratio_3 = 0
    print('blurring_1_ratio = {:.5f}'.format(success_blurring_ratio_1),\
          'blurring_2_ratio = {:.5f}'.format(success_blurring_ratio_2),\
          'blurring_3_ratio = {:.5f}'.format(success_blurring_ratio_3))
        
    with open(args.myresults, 'a+') as f:
        f.write('Eious1 ori is :' + str(1 - sum(Eious1) / len(Eious1)) + '\n')
        f.write('Eious1 min is :' + str(1 - sum(Eious2) / len(Eious2)) + '\n')
        f.write('Eious1 max is :' + str(1 - sum(Eious3) / len(Eious3)) + '\n')
        f.write('Eious1 GT_list is :' + str(1 - sum(Eious4) / len(Eious4)) + '\n')
        f.write('labels flip rate is :' + str(1 - sum(bafracs) / len(bafracs)) + '\n')

        f.write('Total time cost is :' + str(sum(times_costs)) + '\n')
        f.write('Average time cost is :' + str(sum(times_costs) / len(times_costs)) + '\n')
        f.write('blurring_1_ratio is :' + str(success_blurring_ratio_1)+ '\n')
        f.write('blurring_2_ratio is :' + str(success_blurring_ratio_2)+ '\n')
        f.write('blurring_3_ratio is :' + str(success_blurring_ratio_3)+ '\n')

        f.write('\n')
        f.write('\n')
        f.write('\n')
    
    # final_all.append(success_perturbated_img_list)
    # final_all.append(success_modifier_list)
    final_all.append(success_modifier_norm_list)
    final_all.append(success_img_index)
    final_all.append(success_c_list)
    final_all.append(success_modifier_norm_index_0)
    final_all.append(success_modifier_norm_index_1)
    final_all.append(success_modifier_norm_index_2)
    final_all.append(success_modifier_norm_index_3)
    final_all.append(success_modifier_norm_index_4)
    final_all.append(success_modifier_norm_index_5)
    final_all.append(success_blurring_index_1)
    final_all.append(success_blurring_index_2)
    final_all.append(success_blurring_index_3)

    final_all.append(Eious1)
    final_all.append(Eious2)
    final_all.append(Eious3)
    final_all.append(Eious4)

    final_all.append(bafracs)
    final_all.append(befores)
    final_all.append(afters)
    final_all.append(GT_lists)
    final_all.append(pred_idx_ls)

    final_all.append(times_costs)

    os.makedirs('./result/{}/{}/{}/eps_{}'.format(args.dataset,args.label_difficult, args.app, args.eps), exist_ok=True)
    final_all = np.asarray(final_all, dtype=object)
    np.save('./result/{}/{}/{}/eps_{}/k_{}_MI_{}'.format(args.dataset,args.label_difficult, args.app, args.eps, args.k_value,args.maxiter), final_all)
    

def save_result(key, success_c_list, 
                index, 
                args, 
                success_img_index, 
                success_modifier_norm_list, 
                success_modifier_norm_index_0, 
                success_modifier_norm_index_1, 
                success_modifier_norm_index_2, 
                success_modifier_norm_index_3,
                success_modifier_norm_index_4, 
                success_modifier_norm_index_5,
                success_blurring_index_1, 
                success_blurring_index_2, 
                success_blurring_index_3,
                Eious1, 
                Eious2, 
                Eious3, 
                Eious4,
                bafracs, 
                befores, 
                afters, 
                GT_lists, 
                pred_idx_ls, 
                times_costs):
    final_all = []
    c_list_sum = np.sum(np.asarray(success_c_list), 0) / index
    with open(args.myresults, 'a+') as f:
        f.write('MI is :' + str(key)+ '\n')
        f.write('k value is :' + str(args.k_value) + '\n')
    if success_c_list != []:
        print('attack_type= ', "{}".format(args.app), 'label_difficult= ', "{}".format(args.label_difficult))

        print('FR_1= ', "{:.5f}".format(c_list_sum[0]), \
              'FR_2= ', "{:.5f}".format(c_list_sum[1]), \
              'FR_3= ', "{:.5f}".format(c_list_sum[2]), \
              'FR_4= ', "{:.5f}".format(c_list_sum[3]), \
              'FR_5= ', "{:.5f}".format(c_list_sum[4]), \
              'FR_10= ', "{:.5f}".format(c_list_sum[5]))


        print('avg_norm_1= ', "{}".format(
            np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))), \
              'avg_norm_2= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))), \
              'avg_norm_3= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))), \
              'avg_norm_4= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))), \
              'avg_norm_5= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))), \
              'avg_norm_10= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))))
        print('Eiou ori is :', 1 - sum(Eious1) / len(Eious1))
        print('Eiou min is :', 1 - sum(Eious2) / len(Eious2))
        print('Eiou max is :', 1 - sum(Eious3) / len(Eious3))
        print('Eiou GT_list is :', 1 - sum(Eious4) / len(Eious4))
        print('labels flip rate is :', sum(bafracs) / len(bafracs))
        print('total time cost is :', sum(times_costs))
        print('average time cost is :', sum(times_costs) / len(times_costs))

        with open(args.myresults, 'a+') as f:
            f.write('attack type is :' + str(args.app)+ '\n')
            f.write('label difficult is :' + str(args.label_difficult) + '\n')
            f.write('FR_1 is :' + str("{:.5f}".format(c_list_sum[0])) + '\n')
            f.write('FR_2 is :' + str("{:.5f}".format(c_list_sum[1])) + '\n')
            f.write('FR_3 is :' + str("{:.5f}".format(c_list_sum[2])) + '\n')
            f.write('FR_4 is :' + str("{:.5f}".format(c_list_sum[3])) + '\n')
            f.write('FR_5 is :' + str("{:.5f}".format(c_list_sum[4])) + '\n')
            f.write('FR_10 is :' + str("{:.5f}".format(c_list_sum[5])) + '\n')

            avg_norm_1 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))
            avg_norm_2 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))
            avg_norm_3 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))
            avg_norm_4 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))
            avg_norm_5 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))
            avg_norm_10 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))

            f.write('avg_norm_1 is :' + str(avg_norm_1 * 255) + '\n')
            f.write('avg_norm_2 is :' + str(avg_norm_2 * 255) + '\n')
            f.write('avg_norm_3 is :' + str(avg_norm_3 * 255) + '\n')
            f.write('avg_norm_4 is :' + str(avg_norm_4 * 255) + '\n')
            f.write('avg_norm_5 is :' + str(avg_norm_5 * 255) + '\n')
            f.write('avg_norm_10 is :' + str(avg_norm_10 * 255) + '\n')

            f.write('Eious1 ori is :' + str(1 - sum(Eious1) / len(Eious1)) + '\n')
            f.write('Eious1 min is :' + str(1 - sum(Eious2) / len(Eious2)) + '\n')
            f.write('Eious1 max is :' + str(1 - sum(Eious3) / len(Eious3)) + '\n')
            f.write('Eious1 GT_list is :' + str(1 - sum(Eious4) / len(Eious4)) + '\n')
            f.write('labels flip rate is :' + str(sum(bafracs) / len(bafracs)) + '\n')

            f.write('Total time cost is :' + str(sum(times_costs)) + '\n')
            f.write('Average time cost is :' + str(sum(times_costs) / len(times_costs)) + '\n')

    else:
        print('All False')

    if success_blurring_index_1!= []:
        success_blurring_ratio_1 = len(success_blurring_index_1)/ index
    else:
        success_blurring_ratio_1 = 0
    if success_blurring_index_2 != []:
        success_blurring_ratio_2 = len(success_blurring_index_2) / index
    else:
        success_blurring_ratio_2 = 0
    if success_blurring_index_3 != []:
        success_blurring_ratio_3 = len(success_blurring_index_3) / index
    else:
        success_blurring_ratio_3 = 0
    print('blurring_1_ratio = {:.5f}'.format(success_blurring_ratio_1),\
          'blurring_2_ratio = {:.5f}'.format(success_blurring_ratio_2),\
          'blurring_3_ratio = {:.5f}'.format(success_blurring_ratio_3))
    
    with open(args.myresults, 'a+') as f:
        f.write('blurring_1_ratio is :' + str(success_blurring_ratio_1)+ '\n')
        f.write('blurring_2_ratio is :' + str(success_blurring_ratio_2)+ '\n')
        f.write('blurring_3_ratio is :' + str(success_blurring_ratio_3)+ '\n')

        f.write('\n')
        f.write('\n')
        f.write('\n')



            # final_all.append(success_perturbated_img_list)
    # final_all.append(success_modifier_list)
    final_all.append(success_modifier_norm_list)
    final_all.append(success_img_index)
    final_all.append(success_c_list)
    final_all.append(success_modifier_norm_index_0)
    final_all.append(success_modifier_norm_index_1)
    final_all.append(success_modifier_norm_index_2)
    final_all.append(success_modifier_norm_index_3)
    final_all.append(success_modifier_norm_index_4)
    final_all.append(success_modifier_norm_index_5)
    final_all.append(success_blurring_index_1)
    final_all.append(success_blurring_index_2)
    final_all.append(success_blurring_index_3)
    final_all.append(Eious1)
    final_all.append(Eious2)
    final_all.append(Eious3)
    final_all.append(Eious4)

    final_all.append(bafracs)
    final_all.append(befores)
    final_all.append(afters)
    final_all.append(GT_lists)
    final_all.append(pred_idx_ls)

    final_all.append(times_costs)
    os.makedirs('./result/{}/{}/{}/eps_{}'.format(args.dataset,args.label_difficult, args.app, args.eps), exist_ok=True)
    final_all = np.asarray(final_all, dtype=object)
    np.save('./result/{}/{}/{}/eps_{}/k_{}_MI_{}'.format(args.dataset,args.label_difficult, args.app, args.eps, args.k_value, args.maxiter), final_all)

def save_result_T(key,
                  success_c_list, 
                  index, 
                  args, 
                  success_img_index, 
                  success_modifier_norm_list, 
                  success_modifier_norm_index_0, 
                  success_modifier_norm_index_1,
                  success_modifier_norm_index_2, 
                  success_modifier_norm_index_3,
                  success_modifier_norm_index_4, 
                  success_modifier_norm_index_5,
                  success_blurring_index_1, 
                  success_blurring_index_2, 
                  success_blurring_index_3, 
                  Eious_oris_T, 
                  Eious_min_or_GT_maxs_T, 
                  Target_idxs_T, Pred_Topks_T,
                  Befores_T, 
                  Afters_T, 
                  Label_Fracs_T, 
                  Bs_befores_T, 
                  Bs_afters_T, 
                  Bs_fracs_T, 
                  times_costs):


    final_all = []
    final_all.append(success_modifier_norm_list)
    final_all.append(success_img_index)
    final_all.append(success_c_list)
    final_all.append(success_modifier_norm_index_0)
    final_all.append(success_modifier_norm_index_1)
    final_all.append(success_modifier_norm_index_2)
    final_all.append(success_modifier_norm_index_3)
    final_all.append(success_modifier_norm_index_4)
    final_all.append(success_modifier_norm_index_5)
    final_all.append(success_blurring_index_1)
    final_all.append(success_blurring_index_2)
    final_all.append(success_blurring_index_3)
    final_all.append(Eious_oris_T)
    final_all.append(Eious_min_or_GT_maxs_T)
    final_all.append(Target_idxs_T)
    final_all.append(Pred_Topks_T)
    final_all.append(Befores_T)
    final_all.append(Afters_T)
    final_all.append(Bs_befores_T)
    final_all.append(Bs_afters_T)
    final_all.append(Bs_fracs_T)
    final_all.append(times_costs)
    with open(args.myresults, 'a+') as f:
        f.write('############# maxiter iteration is :' + str(key)+ '#############'+'\n')
        f.write('############# k value is :' + str(args.k_value) + '#############' + '\n')
    c_list_sum = np.sum(np.asarray(success_c_list), 0) / index
    if success_c_list != []:
        print('attack_type= ', "{}".format(args.app), 'label_difficult= ', "{}".format(args.label_difficult))
        print('FR_1= ', "{:.5f}".format(c_list_sum[0]), \
              'FR_2= ', "{:.5f}".format(c_list_sum[1]), \
              'FR_3= ', "{:.5f}".format(c_list_sum[2]), \
              'FR_4= ', "{:.5f}".format(c_list_sum[3]), \
              'FR_5= ', "{:.5f}".format(c_list_sum[4]), \
              'FR_10= ', "{:.5f}".format(c_list_sum[5]))
        print('avg_norm_1= ', "{}".format(
            np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))), \
              'avg_norm_2= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))), \
              'avg_norm_3= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))), \
              'avg_norm_4= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))), \
              'avg_norm_5= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))), \
              'avg_norm_10= ', "{}".format(np.average(
                np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))))
        print('')
        print('Eious_oris_T is :', 1 - sum(Eious_oris_T) / len(Eious_oris_T))
        print('Eious_min_or_GT_maxs_T is :', 1 - sum(Eious_min_or_GT_maxs_T) / len(Eious_min_or_GT_maxs_T))
        print('Label flip rate is :', 1 - sum(Label_Fracs_T) / len(Label_Fracs_T))
        print('Label keep same Rate is :', 1 - sum(Bs_fracs_T) / len(Bs_fracs_T))
        with open(args.myresults, 'a+') as f:
            f.write('before_nums_pos is : ' + str(sum(Befores_T)) + '\n')
            f.write('after nums neg is : '+ str(sum(Afters_T)) + '\n')
        with open(args.myresults, 'a+') as f:
            f.write('attack type is :' + str(args.app)+ '\n')
            f.write('label difficult is :' + str(args.label_difficult) + '\n')
            f.write('FR_1 is :' + str("{:.5f}".format(c_list_sum[0])) + '\n')
            f.write('FR_2 is :' + str("{:.5f}".format(c_list_sum[1])) + '\n')
            f.write('FR_3 is :' + str("{:.5f}".format(c_list_sum[2])) + '\n')
            f.write('FR_4 is :' + str("{:.5f}".format(c_list_sum[3])) + '\n')
            f.write('FR_5 is :' + str("{:.5f}".format(c_list_sum[4])) + '\n')
            f.write('FR_10 is :' + str("{:.5f}".format(c_list_sum[5])) + '\n')

            avg_norm_1 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_0))))
            avg_norm_2 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_1))))
            avg_norm_3 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_2))))
            avg_norm_4 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_3))))
            avg_norm_5 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_4))))
            avg_norm_10 = np.average(np.asarray(list(map(success_modifier_norm_list.__getitem__, success_modifier_norm_index_5))))

            f.write('avg_norm_1 is :' + str(avg_norm_1 * 255) + '\n')
            f.write('avg_norm_2 is :' + str(avg_norm_2 * 255) + '\n')
            f.write('avg_norm_3 is :' + str(avg_norm_3 * 255) + '\n')
            f.write('avg_norm_4 is :' + str(avg_norm_4 * 255) + '\n')
            f.write('avg_norm_5 is :' + str(avg_norm_5 * 255) + '\n')
            f.write('avg_norm_10 is :' + str(avg_norm_10 * 255) + '\n')

            f.write('Eious_oris_T is :' + str(sum(Eious_oris_T) / len(Eious_oris_T)) + '\n')
            f.write('Eious_min_or_GT_maxs_T is :' + str(sum(Eious_min_or_GT_maxs_T) / len(Eious_min_or_GT_maxs_T)) + '\n')
            f.write('Label flip rate is :' + str(sum(Label_Fracs_T) / len(Label_Fracs_T)) + '\n')
            f.write('Label keep same Rate is :' + str(1 - sum(Bs_fracs_T) / len(Bs_fracs_T)) + '\n')

            f.write('Total time cost is :' + str(sum(times_costs)) + '\n')
            f.write('Average time cost is :' + str(sum(times_costs) / len(times_costs)) + '\n')
    else:
        print('All False')

    if success_blurring_index_1!= []:
        success_blurring_ratio_1 = len(success_blurring_index_1)/ index
    else:
        success_blurring_ratio_1 = 0
    if success_blurring_index_2 != []:
        success_blurring_ratio_2 = len(success_blurring_index_2) / index
    else:
        success_blurring_ratio_2 = 0
    if success_blurring_index_3 != []:
        success_blurring_ratio_3 = len(success_blurring_index_3) / index
    else:
        success_blurring_ratio_3 = 0
    print('blurring_1_ratio = {:.5f}'.format(success_blurring_ratio_1),\
          'blurring_2_ratio = {:.5f}'.format(success_blurring_ratio_2),\
          'blurring_3_ratio = {:.5f}'.format(success_blurring_ratio_3))

    with open(args.myresults, 'a+') as f:
        f.write('blurring_1_ratio is :' + str(success_blurring_ratio_1)+ '\n')
        f.write('blurring_2_ratio is :' + str(success_blurring_ratio_2)+ '\n')
        f.write('blurring_3_ratio is :' + str(success_blurring_ratio_3)+ '\n')

        f.write('\n')
        f.write('\n')
        f.write('\n')

    os.makedirs('./result/{}/{}/{}/eps_{}'.format(args.dataset,args.label_difficult, args.app, args.eps), exist_ok=True)
    final_all = np.asarray(final_all, dtype=object)
    np.save('./result/{}/{}/{}/eps_{}/k_{}'.format(args.dataset,args.label_difficult, args.app, args.eps, args.k_value), final_all)


def baselineap(args, model, device, val_loader):
    print('kvalue: ',args.k_value, 'label_difficult',args.label_difficult, 'app_type:', args.app,\
          'uap_norm:', args.uap_norm, 'uap_eps:', args.uap_eps)
    model.eval()
    success_count = 0
    index = 0
    index_success = {}
    # success_perturbated_img_list = []
    # success_modifier_list = []
    success_modifier_norm_list = []
    success_img_index = []
    success_c_list = []
    success_modifier_norm_index_0 = []
    success_modifier_norm_index_1 = []
    success_modifier_norm_index_2 = []
    success_modifier_norm_index_3 = []
    success_modifier_norm_index_4 = []
    success_modifier_norm_index_5 = []
    success_blurring_index_1 = []
    success_blurring_index_2 = []
    success_blurring_index_3 = []
    sample_list = np.load('ap_{}_list.npy'.format(args.dataset))
    
    Eious1 = []
    Eious2 = []
    Eious3 = []
    Eious4 = []
    bafracs = []
    befores = []
    afters = []
    GT_lists = []
    pred_idx_ls = []
    times_costs = []
    Eious_oris_T = []
    Eious_min_or_GT_maxs_T = []
    Target_idxs_T = []
    Pred_Topks_T = []
    Befores_T = []
    Afters_T = []
    Label_Fracs_T = []
    Bs_befores_T = []
    Bs_afters_T = []
    Bs_fracs_T = []
    
    if args.app == 'baseline_MIFGSMT':
        all_samples = 1000
        index_success = 0
        before_nums_pos = 0
        after_nums_neg = 0
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:all_samples]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                GT = GT.int()
                if index < all_samples:
                    # target attack
                    print('\n')
                    data = data.to(device)
                    target, targets_zeros = generate_target_zeros_3_cases(model, data, GT.cpu().detach().numpy(), args.label_difficult, k=args.k_value)
                    targets_zeros = torch.from_numpy(targets_zeros).to(device)
                    purtabed_img_out, modifier_out, attack_success, c_list, Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk, before, after, frac, Bs_before, Bs_after, Bs_frac, time_cost=\
                        MIFGSM_T(args, model, data, targets_zeros, args.k_value, args.eps, args.maxiter, args.boxmax, args.boxmin, GT, device=device)
                    times_costs.append(time_cost)
                    Eious_oris_T.append(Eious_iou)
                    Eious_min_or_GT_maxs_T.append(Eious_min_or_GT_max)
                    Target_idxs_T.append(target_idx)
                    Pred_Topks_T.append(pred_topk)
                    Befores_T.append(before)
                    Afters_T.append(after)
                    Label_Fracs_T.append(frac)
                    Bs_befores_T.append(Bs_before)
                    Bs_afters_T.append(Bs_after)
                    Bs_fracs_T.append(Bs_frac)
                    before_nums_pos += before
                    after_nums_neg += after
                        
                    if attack_success:
                        success_count = success_count +1
                        success_modifier_norm_list.append(np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size)))
                        success_img_index.append(ith)
                        success_c_list.append(c_list)
                        if c_list[0]==1:
                            success_modifier_norm_index_0.append(index_success)
                        if c_list[1]==1:
                            success_modifier_norm_index_1.append(index_success)
                        if c_list[2]==1:
                            success_modifier_norm_index_2.append(index_success)
                        if c_list[3]==1:
                            success_modifier_norm_index_3.append(index_success)
                        if c_list[4]==1:
                            success_modifier_norm_index_4.append(index_success)
                        if c_list[5]==1:
                            success_modifier_norm_index_5.append(index_success)
                        index_success = index_success + 1
                    print('success:{}/{}'.format(success_count, index+1))
                    print('index is :', index)
                    index = index + 1
                if index == all_samples:
                    break
        save_result_FGSMT(success_c_list, index, args, success_img_index, success_modifier_norm_list, success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3,success_modifier_norm_index_4, success_modifier_norm_index_5,\
                success_blurring_index_1, success_blurring_index_2, success_blurring_index_3, Eious_oris_T, Eious_min_or_GT_maxs_T, Target_idxs_T, Pred_Topks_T,\
                Befores_T, Afters_T, Label_Fracs_T, Bs_befores_T, Bs_afters_T, Bs_fracs_T, times_costs)

    if args.app == 'baseline_FGSMT':
        all_samples = 1000
        index_success = 0
        before_nums_pos = 0
        after_nums_neg = 0
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:all_samples]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                GT = GT.int()
                if index < all_samples:
                    # target attack
                    print('\n')
                    data = data.to(device)
                    target, targets_zeros = generate_target_zeros_3_cases(model, data, GT.cpu().detach().numpy(), args.label_difficult, k=args.k_value)
                    targets_zeros = torch.from_numpy(targets_zeros).to(device)
                    purtabed_img_out, modifier_out, attack_success, c_list, Eious_iou, Eious_min_or_GT_max, target_idx, pred_topk, before, after, frac, Bs_before, Bs_after, Bs_frac, time_cost=\
                        FGSM_T(args, model, data, targets_zeros, args.k_value, args.eps, args.maxiter, args.boxmax, args.boxmin, GT, device=device)
                    times_costs.append(time_cost)
                    Eious_oris_T.append(Eious_iou)
                    Eious_min_or_GT_maxs_T.append(Eious_min_or_GT_max)
                    Target_idxs_T.append(target_idx)
                    Pred_Topks_T.append(pred_topk)
                    Befores_T.append(before)
                    Afters_T.append(after)
                    Label_Fracs_T.append(frac)
                    Bs_befores_T.append(Bs_before)
                    Bs_afters_T.append(Bs_after)
                    Bs_fracs_T.append(Bs_frac)
                    before_nums_pos += before
                    after_nums_neg += after
                        
                    if attack_success:
                        success_count = success_count +1
                        success_modifier_norm_list.append(np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size)))
                        success_img_index.append(ith)
                        success_c_list.append(c_list)
                        if c_list[0]==1:
                            success_modifier_norm_index_0.append(index_success)
                        if c_list[1]==1:
                            success_modifier_norm_index_1.append(index_success)
                        if c_list[2]==1:
                            success_modifier_norm_index_2.append(index_success)
                        if c_list[3]==1:
                            success_modifier_norm_index_3.append(index_success)
                        if c_list[4]==1:
                            success_modifier_norm_index_4.append(index_success)
                        if c_list[5]==1:
                            success_modifier_norm_index_5.append(index_success)
                        index_success = index_success + 1
                    print('success:{}/{}'.format(success_count, index+1))
                    print('index is :', index)
                    index = index + 1
                if index == all_samples:
                    break
        print('#################')
        print('#################')
        print('#################')
        print('Eious_min_or_GT_maxs_T is :', sum(Eious_min_or_GT_maxs_T) / len(Eious_min_or_GT_maxs_T))
        print('#################')
        print('#################')
        print('#################')
        save_result_FGSMT(success_c_list, index, args, success_img_index, success_modifier_norm_list, success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3,success_modifier_norm_index_4, success_modifier_norm_index_5,\
                success_blurring_index_1, success_blurring_index_2, success_blurring_index_3, Eious_oris_T, Eious_min_or_GT_maxs_T, Target_idxs_T, Pred_Topks_T,\
                Befores_T, Afters_T, Label_Fracs_T, Bs_befores_T, Bs_afters_T, Bs_fracs_T, times_costs)

    if args.app == 'baseline_MIFGSM':
        # none target attack
        index_success = 0
        all_samples = 1000#20675, 4774
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:all_samples]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                GT = GT.int()

                print('\n')
                data, GT = data.to(device), GT.to(device)
                purtabed_img_out, modifier_out, attack_success, c_list, pos, neg, bafrac, iou1, iou2, iou3, iou4, GT_list, pred_idx_l, time_cost = \
                    MI_FGSM(model, data, GT,args.maxiter, args.k_value, args.eps, args.boxmax, args.boxmin, device)
                befores.append(pos)
                afters.append(neg)
                bafracs.append(bafrac)
                Eious1.append(iou1)
                Eious2.append(iou2)
                Eious3.append(iou3)
                Eious4.append(iou4)
                GT_lists.append(GT_list)
                pred_idx_ls.append(pred_idx_l)
                times_costs.append(time_cost)
                if attack_success:
                    ######Gaussian Blurring 3,5,7
                    img_1_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(3, 3), sigmaX=0)
                    img_2_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(5, 5), sigmaX=0)
                    img_3_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(7, 7), sigmaX=0)
                    img_1_ori = np.expand_dims(img_1_ori, axis=0)
                    img_2_ori = np.expand_dims(img_2_ori, axis=0)
                    img_3_ori = np.expand_dims(img_3_ori, axis=0)
                    predict_1 = model(torch.from_numpy(img_1_ori).cuda())
                    Flag_1 = bool(topk_acc_metric(GT.cpu().numpy(), predict_1.cpu().detach().numpy(), args.k_value))
                    predict_2 = model(torch.from_numpy(img_2_ori).cuda())
                    Flag_2 = bool(topk_acc_metric(GT.cpu().numpy(), predict_2.cpu().detach().numpy(), args.k_value))
                    predict_3 = model(torch.from_numpy(img_3_ori).cuda())
                    Flag_3 = bool(topk_acc_metric(GT.cpu().numpy(), predict_3.cpu().detach().numpy(), args.k_value))
                    ############
                    success_count = success_count + 1
                    success_modifier_norm_list.append(np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size)))
                    success_img_index.append(ith)
                    if Flag_1:
                        success_blurring_index_1.append(ith)
                    if Flag_2:
                        success_blurring_index_2.append(ith)
                    if Flag_3:
                        success_blurring_index_3.append(ith)
                    success_c_list.append(c_list)
                    if c_list[0] == 1:
                        success_modifier_norm_index_0.append(index_success)
                    if c_list[1] == 1:
                        success_modifier_norm_index_1.append(index_success)
                    if c_list[2] == 1:
                        success_modifier_norm_index_2.append(index_success)
                    if c_list[3] == 1:
                        success_modifier_norm_index_3.append(index_success)
                    if c_list[4] == 1:
                        success_modifier_norm_index_4.append(index_success)
                    if c_list[5] == 1:
                        success_modifier_norm_index_5.append(index_success)
                    index_success = index_success + 1
                print('success:{}/{}'.format(success_count, index + 1))
                print('index is :', index)
                index = index + 1
                if index == all_samples:
                    break

        save_result_FGSM(success_c_list, index, args, success_img_index, success_modifier_norm_list,
                success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3, \
                success_modifier_norm_index_4, success_modifier_norm_index_5,\
                success_blurring_index_1, success_blurring_index_2,\
                success_blurring_index_3, Eious1, Eious2, Eious3, Eious4,\
                bafracs, befores, afters, GT_lists, pred_idx_ls, times_costs)

    if args.app == 'baseline_FGSM':
        # none target attack
        index_success = 0
        all_samples = 1000#20675, 4774
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:all_samples]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                GT = GT.int()

                print('\n')
                data, GT = data.to(device), GT.to(device)
                purtabed_img_out, modifier_out, attack_success, c_list, pos, neg, bafrac, iou1, iou2, iou3, iou4, GT_list, pred_idx_l, time_cost = \
                    FGSM(model, data, GT, args.k_value, args.eps, args.boxmax, args.boxmin, device)
                befores.append(pos)
                afters.append(neg)
                bafracs.append(bafrac)
                Eious1.append(iou1)
                Eious2.append(iou2)
                Eious3.append(iou3)
                Eious4.append(iou4)
                GT_lists.append(GT_list)
                pred_idx_ls.append(pred_idx_l)
                times_costs.append(time_cost)
                if attack_success:
                    ######Gaussian Blurring 3,5,7
                    img_1_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(3, 3), sigmaX=0)
                    img_2_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(5, 5), sigmaX=0)
                    img_3_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(7, 7), sigmaX=0)
                    img_1_ori = np.expand_dims(img_1_ori, axis=0)
                    img_2_ori = np.expand_dims(img_2_ori, axis=0)
                    img_3_ori = np.expand_dims(img_3_ori, axis=0)
                    predict_1 = model(torch.from_numpy(img_1_ori).cuda())
                    Flag_1 = bool(topk_acc_metric(GT.cpu().numpy(), predict_1.cpu().detach().numpy(), args.k_value))
                    predict_2 = model(torch.from_numpy(img_2_ori).cuda())
                    Flag_2 = bool(topk_acc_metric(GT.cpu().numpy(), predict_2.cpu().detach().numpy(), args.k_value))
                    predict_3 = model(torch.from_numpy(img_3_ori).cuda())
                    Flag_3 = bool(topk_acc_metric(GT.cpu().numpy(), predict_3.cpu().detach().numpy(), args.k_value))
                    ############
                    success_count = success_count + 1
                    success_modifier_norm_list.append(np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size)))
                    success_img_index.append(ith)
                    if Flag_1:
                        success_blurring_index_1.append(ith)
                    if Flag_2:
                        success_blurring_index_2.append(ith)
                    if Flag_3:
                        success_blurring_index_3.append(ith)
                    success_c_list.append(c_list)
                    if c_list[0] == 1:
                        success_modifier_norm_index_0.append(index_success)
                    if c_list[1] == 1:
                        success_modifier_norm_index_1.append(index_success)
                    if c_list[2] == 1:
                        success_modifier_norm_index_2.append(index_success)
                    if c_list[3] == 1:
                        success_modifier_norm_index_3.append(index_success)
                    if c_list[4] == 1:
                        success_modifier_norm_index_4.append(index_success)
                    if c_list[5] == 1:
                        success_modifier_norm_index_5.append(index_success)
                    index_success = index_success + 1
                print('success:{}/{}'.format(success_count, index + 1))
                print('index is :', index)
                index = index + 1
                if index == all_samples:
                    break

        save_result_FGSM(success_c_list, index, args, success_img_index, success_modifier_norm_list,
                success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3, \
                success_modifier_norm_index_4, success_modifier_norm_index_5,\
                success_blurring_index_1, success_blurring_index_2,\
                success_blurring_index_3, Eious1, Eious2, Eious3, Eious4,\
                bafracs, befores, afters, GT_lists, pred_idx_ls, times_costs)
        

    if args.app == 'baseline_kUAP':
        ###Train
        global_delta = kUAP(args, model, device, val_loader, sample_list, args.uap_train_index_end)
        np.save('./result/{}/{}/{}/eps_{}/perturbation_{}.npy'.format(args.dataset, args.label_difficult, args.app,
                                                                      args.eps, args.k_value),
                global_delta.cpu().detach().numpy())
        ###Test
        boxmul = (args.boxmax - args.boxmin) / 2.
        boxplus = (args.boxmin + args.boxmax) / 2.
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[args.uap_test_index_start:args.uap_test_index_end]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                print('ith2:', index + 1)
                data, GT = data.to(device), GT.to(device)
                pred = model(torch.tanh(data + global_delta) * boxmul + boxplus)
                # Flag, predict_label = nontargeted_TP_index(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value)
                Flag = bool(topk_acc_metric(GT.cpu().numpy(), pred.cpu().detach().numpy(), args.k_value))
                if Flag == True:
                    success_count = success_count + 1
                    c_list = [0] * 6
                    c_list[0], c_list[1], c_list[2], c_list[3], c_list[4], c_list[5] = \
                        topk_acc_metric_1_to_10(GT.cpu().detach().numpy(), pred.cpu().detach().numpy())
                    success_modifier_norm_list.append(np.linalg.norm(global_delta.cpu().detach().numpy())/((args.image_size)*(args.image_size)))
                    success_img_index.append(ith)
                    success_c_list.append(c_list)
                    if c_list[0] == 1:
                        success_modifier_norm_index_0.append(index_success)
                    if c_list[1] == 1:
                        success_modifier_norm_index_1.append(index_success)
                    if c_list[2] == 1:
                        success_modifier_norm_index_2.append(index_success)
                    if c_list[3] == 1:
                        success_modifier_norm_index_3.append(index_success)
                    if c_list[4] == 1:
                        success_modifier_norm_index_4.append(index_success)
                    if c_list[5] == 1:
                        success_modifier_norm_index_5.append(index_success)
                    index_success = index_success + 1
                print('success:{}/{}'.format(success_count, index + 1))
                index = index + 1
            if index == (args.uap_test_index_end - args.uap_test_index_start):
                break
        save_result(success_c_list, index, args, success_img_index, success_modifier_norm_list,
                success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3, \
                success_modifier_norm_index_4, success_modifier_norm_index_5, \
                success_blurring_index_1, success_blurring_index_2, \
                success_blurring_index_3
                )

    if args.app == 'baseline_kfool':

        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:1000]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                GT = GT.int()
                if index < 1000:
                    print('\n')
                    data, GT = data.to(device), GT.to(device)
                    purtabed_img_out, modifier_out, attack_success, c_list =\
                        kfool(model, data, GT, args.k_value, args.boxmax, args.boxmin, args.maxiter, device=device, lr=args.lr_attack)



                    if attack_success:
                        ######Gaussian Blurring 3,5,7
                        img_1_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(3, 3), sigmaX=0)
                        img_2_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(5, 5), sigmaX=0)
                        img_3_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(7, 7), sigmaX=0)
                        img_1_ori = np.expand_dims(img_1_ori, axis=0)
                        img_2_ori = np.expand_dims(img_2_ori, axis=0)
                        img_3_ori = np.expand_dims(img_3_ori, axis=0)
                        predict_1 = model(torch.from_numpy(img_1_ori).cuda())
                        Flag_1 = bool(topk_acc_metric(GT.cpu().numpy(), predict_1.cpu().detach().numpy(), args.k_value))
                        predict_2 = model(torch.from_numpy(img_2_ori).cuda())
                        Flag_2 = bool(topk_acc_metric(GT.cpu().numpy(), predict_2.cpu().detach().numpy(), args.k_value))
                        predict_3 = model(torch.from_numpy(img_3_ori).cuda())
                        Flag_3 = bool(topk_acc_metric(GT.cpu().numpy(), predict_3.cpu().detach().numpy(), args.k_value))
                        #############
                        success_count = success_count + 1
                        # success_perturbated_img_list.append(purtabed_img_out)
                        # success_modifier_list.append(modifier_out)
                        success_modifier_norm_list.append(np.linalg.norm(modifier_out)/((args.image_size)*(args.image_size)))
                        success_img_index.append(ith)
                        if Flag_1:
                            success_blurring_index_1.append(ith)
                        if Flag_2:
                            success_blurring_index_2.append(ith)
                        if Flag_3:
                            success_blurring_index_3.append(ith)
                        success_c_list.append(c_list)
                        if c_list[0] == 1:
                            success_modifier_norm_index_0.append(index_success)
                        if c_list[1] == 1:
                            success_modifier_norm_index_1.append(index_success)
                        if c_list[2] == 1:
                            success_modifier_norm_index_2.append(index_success)
                        if c_list[3] == 1:
                            success_modifier_norm_index_3.append(index_success)
                        if c_list[4] == 1:
                            success_modifier_norm_index_4.append(index_success)
                        if c_list[5] == 1:
                            success_modifier_norm_index_5.append(index_success)
                        index_success = index_success + 1

                    print('success:{}/{}'.format(success_count, index + 1))
                    index = index + 1
            if index == 1000:
                break
        save_result(success_c_list, index, args, success_img_index, success_modifier_norm_list,
                success_modifier_norm_index_0, success_modifier_norm_index_1, \
                success_modifier_norm_index_2, success_modifier_norm_index_3, \
                success_modifier_norm_index_4, success_modifier_norm_index_5, \
                success_blurring_index_1, success_blurring_index_2, \
                success_blurring_index_3
                )
    
    if args.app == 'baseline_DF':
        print('now is the baseline CW !!')
        samples_all = 1000
        All_res = {}
        for iter in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:#, 300, 400, 500
            index_success[iter] = 0
            All_res[iter] = {}
            for key in ['success_c_list',
                        'success_modifier_norm_list', 
                        'success_modifier_norm_index_0', 
                        'success_modifier_norm_index_1', 
                        'success_modifier_norm_index_2',
                        'success_modifier_norm_index_3', 
                        'success_modifier_norm_index_4', 
                        'success_modifier_norm_index_5', 
                        'Eious_oris_T', 
                        'Eious_min_or_GT_maxs_T',
                        'Target_idxs_T',
                        'Pred_Topks_T', 
                        'Befores_T', 
                        'Afters_T', 
                        'Label_Fracs_T', 
                        'Bs_befores_T', 
                        'Bs_afters_T', 
                        'Bs_fracs_T',
                        'times_costs']:
                        All_res[iter][key] = []

        
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if ith in sample_list[:samples_all]:
                if len(GT.shape) == 3:
                    GT = GT.max(dim=1)[0]
                else:
                    pass
                GT = GT.int()
                print('\n')
                data = data.to(device)
                target, targets_zeros = generate_target_zeros_3_cases(model, data, GT.cpu().detach().numpy(),
                                                                        args.label_difficult, k=args.k_value)
                targets_zeros = torch.from_numpy(targets_zeros).to(device)

                ######MLAP rank I baseline from the paper 'Multi-Label Adversarial Perturbations'
                purtabed_img_out, modifier_out, attack_success, c_list, sub_res, iteration, norm_i = \
                    DF(index_success, args, model, data, targets_zeros, args.k_value, args.eps, args.maxiter, args.boxmax, args.boxmin, \
                            GT, device=device, lr=args.lr_attack)
                for key in sub_res.keys():
                    All_res[key]['success_c_list'].append(sub_res[key]['c_list'])

                    if sub_res[key]['norm'] is not None:
                        All_res[key]['success_modifier_norm_list'].append(sub_res[key]['norm'])

                    if sub_res[key]['index_success_0'] is not None:
                        All_res[key]['success_modifier_norm_index_0'].append(sub_res[key]['index_success_0'])
                    if sub_res[key]['index_success_1'] is not None:
                        All_res[key]['success_modifier_norm_index_1'].append(sub_res[key]['index_success_1'])
                    if sub_res[key]['index_success_2'] is not None:
                        All_res[key]['success_modifier_norm_index_2'].append(sub_res[key]['index_success_2'])
                    if sub_res[key]['index_success_3'] is not None:
                        All_res[key]['success_modifier_norm_index_3'].append(sub_res[key]['index_success_3'])
                    if sub_res[key]['index_success_4'] is not None:
                        All_res[key]['success_modifier_norm_index_4'].append(sub_res[key]['index_success_4'])
                    if sub_res[key]['index_success_5'] is not None:
                        All_res[key]['success_modifier_norm_index_5'].append(sub_res[key]['index_success_5'])

                    All_res[key]['Eious_oris_T'].append(sub_res[key]['Eious_iou'])
                    All_res[key]['Eious_min_or_GT_maxs_T'].append(sub_res[key]['Eious_min_or_GT_max'])
                    All_res[key]['Target_idxs_T'].append(sub_res[key]['target_idx'])
                    All_res[key]['Pred_Topks_T'].append(sub_res[key]['pred_topk'])
                    All_res[key]['Befores_T'].append(sub_res[key]['before'])
                    All_res[key]['Afters_T'].append(sub_res[key]['after'])
                    All_res[key]['Label_Fracs_T'].append(sub_res[key]['frac'])
                    All_res[key]['Bs_befores_T'].append(sub_res[key]['Bs_before'])
                    All_res[key]['Bs_afters_T'].append(sub_res[key]['Bs_after'])
                    All_res[key]['Bs_fracs_T'].append(sub_res[key]['Bs_frac'])
                    All_res[key]['times_costs'].append(sub_res[key]['cost_time'])


                if attack_success and not np.isnan(norm_i):
                    success_count = success_count + 1
                    success_img_index.append(ith)
                    for key in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:#, 300, 400, 500
                        if key >= iteration:
                            index_success[key] += 1
                print('success:{}/{}'.format(success_count, index + 1))
                index = index + 1
                if index == samples_all:
                    break

        for key in All_res.keys():
            save_result_T(key,
                          All_res[key]['success_c_list'],
                          index,
                          args,
                          success_img_index,
                          All_res[key]['success_modifier_norm_list'],
                          All_res[key]['success_modifier_norm_index_0'],
                          All_res[key]['success_modifier_norm_index_1'],
                          All_res[key]['success_modifier_norm_index_2'], 
                          All_res[key]['success_modifier_norm_index_3'],
                          All_res[key]['success_modifier_norm_index_4'], 
                          All_res[key]['success_modifier_norm_index_5'],
                          success_blurring_index_1, 
                          success_blurring_index_2,
                          success_blurring_index_3,
                          All_res[key]['Eious_oris_T'],
                          All_res[key]['Eious_min_or_GT_maxs_T'],
                          All_res[key]['Target_idxs_T'],
                          All_res[key]['Pred_Topks_T'],
                          All_res[key]['Befores_T'],
                          All_res[key]['Afters_T'],
                          All_res[key]['Label_Fracs_T'],
                          All_res[key]['Bs_befores_T'],
                          All_res[key]['Bs_afters_T'],
                          All_res[key]['Bs_fracs_T'],
                          All_res[key]['times_costs'])
    
    if args.app == 'baseline_PGDT':
        samples_all = 1000
        All_res = {}
        for iter in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:#
            index_success[iter] = 0
            All_res[iter] = {}
            for key in ['success_c_list',
                        'success_modifier_norm_list', 
                        'success_modifier_norm_index_0', 
                        'success_modifier_norm_index_1', 
                        'success_modifier_norm_index_2',
                        'success_modifier_norm_index_3', 
                        'success_modifier_norm_index_4', 
                        'success_modifier_norm_index_5', 
                        'Eious_oris_T', 
                        'Eious_min_or_GT_maxs_T',
                        'Target_idxs_T',
                        'Pred_Topks_T', 
                        'Befores_T', 
                        'Afters_T', 
                        'Label_Fracs_T', 
                        'Bs_befores_T', 
                        'Bs_afters_T', 
                        'Bs_fracs_T',
                        'times_costs']:
                        All_res[iter][key] = []
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()
            print('\n')
            data = data.to(device)
            target, targets_zeros = generate_target_zeros_3_cases(model, data, GT.cpu().detach().numpy(),
                                                                    args.label_difficult, k=args.k_value)
            targets_zeros = torch.from_numpy(targets_zeros).to(device)

            ######MLAP rank I baseline from the paper 'Multi-Label Adversarial Perturbations'
            purtabed_img_out, modifier_out, attack_success, c_list, sub_res, iteration = \
                PGD_T(index_success, args, model, data, targets_zeros, args.k_value, args.eps, args.maxiter, args.boxmax, args.boxmin, \
                        GT, device=device, lr=args.lr_attack)
            for key in sub_res.keys():
                All_res[key]['success_c_list'].append(sub_res[key]['c_list'])

                if sub_res[key]['norm'] is not None:
                    All_res[key]['success_modifier_norm_list'].append(sub_res[key]['norm'])

                if sub_res[key]['index_success_0'] is not None:
                    All_res[key]['success_modifier_norm_index_0'].append(sub_res[key]['index_success_0'])
                if sub_res[key]['index_success_1'] is not None:
                    All_res[key]['success_modifier_norm_index_1'].append(sub_res[key]['index_success_1'])
                if sub_res[key]['index_success_2'] is not None:
                    All_res[key]['success_modifier_norm_index_2'].append(sub_res[key]['index_success_2'])
                if sub_res[key]['index_success_3'] is not None:
                    All_res[key]['success_modifier_norm_index_3'].append(sub_res[key]['index_success_3'])
                if sub_res[key]['index_success_4'] is not None:
                    All_res[key]['success_modifier_norm_index_4'].append(sub_res[key]['index_success_4'])
                if sub_res[key]['index_success_5'] is not None:
                    All_res[key]['success_modifier_norm_index_5'].append(sub_res[key]['index_success_5'])

                All_res[key]['Eious_oris_T'].append(sub_res[key]['Eious_iou'])
                All_res[key]['Eious_min_or_GT_maxs_T'].append(sub_res[key]['Eious_min_or_GT_max'])
                All_res[key]['Target_idxs_T'].append(sub_res[key]['target_idx'])
                All_res[key]['Pred_Topks_T'].append(sub_res[key]['pred_topk'])
                All_res[key]['Befores_T'].append(sub_res[key]['before'])
                All_res[key]['Afters_T'].append(sub_res[key]['after'])
                All_res[key]['Label_Fracs_T'].append(sub_res[key]['frac'])
                All_res[key]['Bs_befores_T'].append(sub_res[key]['Bs_before'])
                All_res[key]['Bs_afters_T'].append(sub_res[key]['Bs_after'])
                All_res[key]['Bs_fracs_T'].append(sub_res[key]['Bs_frac'])
                All_res[key]['times_costs'].append(sub_res[key]['cost_time'])


            if attack_success:
                success_count = success_count + 1
                success_img_index.append(ith)
                for key in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                    if key >= iteration:
                        index_success[key] += 1
            print('success:{}/{}'.format(success_count, index + 1))
            index = index + 1
            if index == samples_all:
                break

        for key in All_res.keys():
            save_result_T(key,
                          All_res[key]['success_c_list'],
                          index,
                          args,
                          success_img_index,
                          All_res[key]['success_modifier_norm_list'],
                          All_res[key]['success_modifier_norm_index_0'],
                          All_res[key]['success_modifier_norm_index_1'],
                          All_res[key]['success_modifier_norm_index_2'], 
                          All_res[key]['success_modifier_norm_index_3'],
                          All_res[key]['success_modifier_norm_index_4'], 
                          All_res[key]['success_modifier_norm_index_5'],
                          success_blurring_index_1, 
                          success_blurring_index_2,
                          success_blurring_index_3,
                          All_res[key]['Eious_oris_T'],
                          All_res[key]['Eious_min_or_GT_maxs_T'],
                          All_res[key]['Target_idxs_T'],
                          All_res[key]['Pred_Topks_T'],
                          All_res[key]['Befores_T'],
                          All_res[key]['Afters_T'],
                          All_res[key]['Label_Fracs_T'],
                          All_res[key]['Bs_befores_T'],
                          All_res[key]['Bs_afters_T'],
                          All_res[key]['Bs_fracs_T'],
                          All_res[key]['times_costs'])
            
    if args.app == 'baseline_PGD':
        # none target attack
        all_samples = 40#20675, 4774
        before_nums_pos = 0
        after_nums_neg = 0
        All_res = {}
        for iter in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:#
            index_success[iter] = 0
            All_res[iter] = {}
            for key in ['success_c_list',
                        'success_modifier_norm_list', 
                        'success_modifier_norm_index_0', 
                        'success_modifier_norm_index_1', 
                        'success_modifier_norm_index_2',
                        'success_modifier_norm_index_3', 
                        'success_modifier_norm_index_4', 
                        'success_modifier_norm_index_5', 
                        'Eious1', 
                        'Eious2',
                        'Eious3',
                        'Eious4', 
                        'bafracs', 
                        'befores', 
                        'afters', 
                        'GT_lists',
                        'pred_idx_ls',
                        'times_costs']:
                        All_res[iter][key] = []
        loss_S = []
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()
            print('\n')
            data, GT = data.to(device), GT.to(device)
            purtabed_img_out, modifier_out,converge_iter, attack_success, c_list, sub_res, iteration, loss_data = \
                PGD(index, index_success, args, model, data, GT, args.k_value, args.eps, args.maxiter, args.boxmax, args.boxmin, \
                                        device, Projection_flag=True, lr=args.lr_attack)
            loss_S.append(loss_data)
            plt.figure()
            plt.plot(loss_data)
            plt.savefig('./flucts/PGDU-loss-{}-k-{}-iter-{}-DS-{}.pdf'.format(index, args.k_value, str(converge_iter), args.dataset))
            loss_S.append(loss_data)
            for key in sub_res.keys():
                All_res[key]['success_c_list'].append(sub_res[key]['c_list'])

                if sub_res[key]['norm'] is not None:
                    All_res[key]['success_modifier_norm_list'].append(sub_res[key]['norm'])

                if sub_res[key]['index_success_0'] is not None:
                    All_res[key]['success_modifier_norm_index_0'].append(sub_res[key]['index_success_0'])
                if sub_res[key]['index_success_1'] is not None:
                    All_res[key]['success_modifier_norm_index_1'].append(sub_res[key]['index_success_1'])
                if sub_res[key]['index_success_2'] is not None:
                    All_res[key]['success_modifier_norm_index_2'].append(sub_res[key]['index_success_2'])
                if sub_res[key]['index_success_3'] is not None:
                    All_res[key]['success_modifier_norm_index_3'].append(sub_res[key]['index_success_3'])
                if sub_res[key]['index_success_4'] is not None:
                    All_res[key]['success_modifier_norm_index_4'].append(sub_res[key]['index_success_4'])
                if sub_res[key]['index_success_5'] is not None:
                    All_res[key]['success_modifier_norm_index_5'].append(sub_res[key]['index_success_5'])
                
                All_res[key]['Eious1'].append(sub_res[key]['iou1'])
                All_res[key]['Eious2'].append(sub_res[key]['iou2'])
                All_res[key]['Eious3'].append(sub_res[key]['iou3'])
                All_res[key]['Eious4'].append(sub_res[key]['iou4'])
                All_res[key]['bafracs'].append(sub_res[key]['bafrac'])
                All_res[key]['befores'].append(sub_res[key]['before'])
                All_res[key]['afters'].append(sub_res[key]['after'])
                All_res[key]['GT_lists'].append(sub_res[key]['GT_list'])
                All_res[key]['pred_idx_ls'].append(sub_res[key]['pred_idx_l'])
                All_res[key]['times_costs'].append(sub_res[key]['cost_time'])
                
            
            if attack_success:
                img_1_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(3, 3), sigmaX=0)
                img_2_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(5, 5), sigmaX=0)
                img_3_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(7, 7), sigmaX=0)

                img_1_ori = np.expand_dims(img_1_ori, axis=0)
                img_2_ori = np.expand_dims(img_2_ori, axis=0)
                img_3_ori = np.expand_dims(img_3_ori, axis=0)

                predict_1 = model(torch.from_numpy(img_1_ori).cuda())
                Flag_1 = bool(topk_acc_metric(GT.cpu().numpy(), predict_1.cpu().detach().numpy(), args.k_value))
                predict_2 = model(torch.from_numpy(img_2_ori).cuda())
                Flag_2 = bool(topk_acc_metric(GT.cpu().numpy(), predict_2.cpu().detach().numpy(), args.k_value))
                predict_3 = model(torch.from_numpy(img_3_ori).cuda())
                Flag_3 = bool(topk_acc_metric(GT.cpu().numpy(), predict_3.cpu().detach().numpy(), args.k_value))
                ############
                success_count = success_count + 1
                
                success_img_index.append(ith)
                if Flag_1:
                    success_blurring_index_1.append(ith)
                if Flag_2:
                    success_blurring_index_2.append(ith)
                if Flag_3:
                    success_blurring_index_3.append(ith)
                for key in All_res.keys():
                    if key >= iteration:
                        index_success[key] += 1
            index = index + 1
            if index == all_samples:
                np.save('./flucts/PGDU-lossarr-k-{}.npy'.format(args.k_value), loss_S)
                break


        for key in All_res.keys():
            save_result(key, All_res[key]['success_c_list'], 
                    index, 
                    args, 
                    success_img_index, 
                    All_res[key]['success_modifier_norm_list'],
                    All_res[key]['success_modifier_norm_index_0'], 
                    All_res[key]['success_modifier_norm_index_1'],
                    All_res[key]['success_modifier_norm_index_2'], 
                    All_res[key]['success_modifier_norm_index_3'],
                    All_res[key]['success_modifier_norm_index_4'], 
                    All_res[key]['success_modifier_norm_index_5'],
                    success_blurring_index_1, 
                    success_blurring_index_2,
                    success_blurring_index_3,
                    All_res[key]['Eious1'],
                    All_res[key]['Eious2'], 
                    All_res[key]['Eious3'], 
                    All_res[key]['Eious4'],
                    All_res[key]['bafracs'], 
                    All_res[key]['befores'], 
                    All_res[key]['afters'], 
                    All_res[key]['GT_lists'], 
                    All_res[key]['pred_idx_ls'], 
                    All_res[key]['times_costs'])
            
    if args.app == 'baseline_CWU':
        # none target attack
        all_samples = 100#20675, 4774
        before_nums_pos = 0
        after_nums_neg = 0
        All_res = {}
        for iter in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:#
            index_success[iter] = 0
            All_res[iter] = {}
            for key in ['success_c_list',
                        'success_modifier_norm_list', 
                        'success_modifier_norm_index_0', 
                        'success_modifier_norm_index_1', 
                        'success_modifier_norm_index_2',
                        'success_modifier_norm_index_3', 
                        'success_modifier_norm_index_4', 
                        'success_modifier_norm_index_5', 
                        'Eious1', 
                        'Eious2',
                        'Eious3',
                        'Eious4', 
                        'bafracs', 
                        'befores', 
                        'afters', 
                        'GT_lists',
                        'pred_idx_ls',
                        'times_costs']:
                        All_res[iter][key] = []
        loss_S = []
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()
            print('\n')
            data, GT = data.to(device), GT.to(device)
            purtabed_img_out, modifier_out,converge_iter, attack_success, c_list, sub_res, iteration, loss_data= \
                CWU(index, index_success, args, model, data, GT, args.k_value, args.eps, args.maxiter, args.boxmax, args.boxmin, \
                                        device, Projection_flag=True, lr=args.lr_attack)
            loss_S.append(loss_data)
            plt.figure()
            plt.plot(loss_data)
            plt.savefig('./flucts/CWU/CWU-loss-{}-k-{}-iter-{}-DS-{}.pdf'.format(index, args.k_value, str(converge_iter), args.dataset))
            loss_S.append(loss_data)
            
            for key in sub_res.keys():
                All_res[key]['success_c_list'].append(sub_res[key]['c_list'])

                if sub_res[key]['norm'] is not None:
                    All_res[key]['success_modifier_norm_list'].append(sub_res[key]['norm'])

                if sub_res[key]['index_success_0'] is not None:
                    All_res[key]['success_modifier_norm_index_0'].append(sub_res[key]['index_success_0'])
                if sub_res[key]['index_success_1'] is not None:
                    All_res[key]['success_modifier_norm_index_1'].append(sub_res[key]['index_success_1'])
                if sub_res[key]['index_success_2'] is not None:
                    All_res[key]['success_modifier_norm_index_2'].append(sub_res[key]['index_success_2'])
                if sub_res[key]['index_success_3'] is not None:
                    All_res[key]['success_modifier_norm_index_3'].append(sub_res[key]['index_success_3'])
                if sub_res[key]['index_success_4'] is not None:
                    All_res[key]['success_modifier_norm_index_4'].append(sub_res[key]['index_success_4'])
                if sub_res[key]['index_success_5'] is not None:
                    All_res[key]['success_modifier_norm_index_5'].append(sub_res[key]['index_success_5'])
                
                All_res[key]['Eious1'].append(sub_res[key]['iou1'])
                All_res[key]['Eious2'].append(sub_res[key]['iou2'])
                All_res[key]['Eious3'].append(sub_res[key]['iou3'])
                All_res[key]['Eious4'].append(sub_res[key]['iou4'])
                All_res[key]['bafracs'].append(sub_res[key]['bafrac'])
                All_res[key]['befores'].append(sub_res[key]['before'])
                All_res[key]['afters'].append(sub_res[key]['after'])
                All_res[key]['GT_lists'].append(sub_res[key]['GT_list'])
                All_res[key]['pred_idx_ls'].append(sub_res[key]['pred_idx_l'])
                All_res[key]['times_costs'].append(sub_res[key]['cost_time'])
                
            
            if attack_success:
                img_1_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(3, 3), sigmaX=0)
                img_2_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(5, 5), sigmaX=0)
                img_3_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(7, 7), sigmaX=0)

                img_1_ori = np.expand_dims(img_1_ori, axis=0)
                img_2_ori = np.expand_dims(img_2_ori, axis=0)
                img_3_ori = np.expand_dims(img_3_ori, axis=0)

                predict_1 = model(torch.from_numpy(img_1_ori).cuda())
                Flag_1 = bool(topk_acc_metric(GT.cpu().numpy(), predict_1.cpu().detach().numpy(), args.k_value))
                predict_2 = model(torch.from_numpy(img_2_ori).cuda())
                Flag_2 = bool(topk_acc_metric(GT.cpu().numpy(), predict_2.cpu().detach().numpy(), args.k_value))
                predict_3 = model(torch.from_numpy(img_3_ori).cuda())
                Flag_3 = bool(topk_acc_metric(GT.cpu().numpy(), predict_3.cpu().detach().numpy(), args.k_value))
                ############
                success_count = success_count + 1
                
                success_img_index.append(ith)
                if Flag_1:
                    success_blurring_index_1.append(ith)
                if Flag_2:
                    success_blurring_index_2.append(ith)
                if Flag_3:
                    success_blurring_index_3.append(ith)
                for key in All_res.keys():
                    if key >= iteration:
                        index_success[key] += 1
            index = index + 1
            if index == all_samples:
                np.save('./flucts/CWU/CWU-lossarr-k-{}-{}.npy'.format(args.k_value, args.dataset), loss_S)
                break


        for key in All_res.keys():
            save_result(key, All_res[key]['success_c_list'], 
                    index, 
                    args, 
                    success_img_index, 
                    All_res[key]['success_modifier_norm_list'],
                    All_res[key]['success_modifier_norm_index_0'], 
                    All_res[key]['success_modifier_norm_index_1'],
                    All_res[key]['success_modifier_norm_index_2'], 
                    All_res[key]['success_modifier_norm_index_3'],
                    All_res[key]['success_modifier_norm_index_4'], 
                    All_res[key]['success_modifier_norm_index_5'],
                    success_blurring_index_1, 
                    success_blurring_index_2,
                    success_blurring_index_3,
                    All_res[key]['Eious1'],
                    All_res[key]['Eious2'], 
                    All_res[key]['Eious3'], 
                    All_res[key]['Eious4'],
                    All_res[key]['bafracs'], 
                    All_res[key]['befores'], 
                    All_res[key]['afters'], 
                    All_res[key]['GT_lists'], 
                    All_res[key]['pred_idx_ls'], 
                    All_res[key]['times_costs'])
            
    if args.app == 'baseline_DFU':
        # none target attack
        all_samples = 1000#20675, 4774
        before_nums_pos = 0
        after_nums_neg = 0
        All_res = {}
        for iter in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:#
            index_success[iter] = 0
            All_res[iter] = {}
            for key in ['success_c_list',
                        'success_modifier_norm_list', 
                        'success_modifier_norm_index_0', 
                        'success_modifier_norm_index_1', 
                        'success_modifier_norm_index_2',
                        'success_modifier_norm_index_3', 
                        'success_modifier_norm_index_4', 
                        'success_modifier_norm_index_5', 
                        'Eious1', 
                        'Eious2',
                        'Eious3',
                        'Eious4', 
                        'bafracs', 
                        'befores', 
                        'afters', 
                        'GT_lists',
                        'pred_idx_ls',
                        'times_costs']:
                        All_res[iter][key] = []

        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()
            print('\n')
            data, GT = data.to(device), GT.to(device)
            purtabed_img_out, modifier_out,converge_iter, attack_success, c_list, sub_res, iteration = \
                DFU(index, index_success, args, model, data, GT, args.k_value, args.eps, args.maxiter, args.boxmax, args.boxmin, \
                                        device, Projection_flag=True, lr=args.lr_attack)
            for key in sub_res.keys():
                All_res[key]['success_c_list'].append(sub_res[key]['c_list'])

                if sub_res[key]['norm'] is not None:
                    All_res[key]['success_modifier_norm_list'].append(sub_res[key]['norm'])

                if sub_res[key]['index_success_0'] is not None:
                    All_res[key]['success_modifier_norm_index_0'].append(sub_res[key]['index_success_0'])
                if sub_res[key]['index_success_1'] is not None:
                    All_res[key]['success_modifier_norm_index_1'].append(sub_res[key]['index_success_1'])
                if sub_res[key]['index_success_2'] is not None:
                    All_res[key]['success_modifier_norm_index_2'].append(sub_res[key]['index_success_2'])
                if sub_res[key]['index_success_3'] is not None:
                    All_res[key]['success_modifier_norm_index_3'].append(sub_res[key]['index_success_3'])
                if sub_res[key]['index_success_4'] is not None:
                    All_res[key]['success_modifier_norm_index_4'].append(sub_res[key]['index_success_4'])
                if sub_res[key]['index_success_5'] is not None:
                    All_res[key]['success_modifier_norm_index_5'].append(sub_res[key]['index_success_5'])
                
                All_res[key]['Eious1'].append(sub_res[key]['iou1'])
                All_res[key]['Eious2'].append(sub_res[key]['iou2'])
                All_res[key]['Eious3'].append(sub_res[key]['iou3'])
                All_res[key]['Eious4'].append(sub_res[key]['iou4'])
                All_res[key]['bafracs'].append(sub_res[key]['bafrac'])
                All_res[key]['befores'].append(sub_res[key]['before'])
                All_res[key]['afters'].append(sub_res[key]['after'])
                All_res[key]['GT_lists'].append(sub_res[key]['GT_list'])
                All_res[key]['pred_idx_ls'].append(sub_res[key]['pred_idx_l'])
                All_res[key]['times_costs'].append(sub_res[key]['cost_time'])
                
            
            if attack_success:
                img_1_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(3, 3), sigmaX=0)
                img_2_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(5, 5), sigmaX=0)
                img_3_ori = cv2.GaussianBlur(purtabed_img_out[0], ksize=(7, 7), sigmaX=0)

                img_1_ori = np.expand_dims(img_1_ori, axis=0)
                img_2_ori = np.expand_dims(img_2_ori, axis=0)
                img_3_ori = np.expand_dims(img_3_ori, axis=0)

                predict_1 = model(torch.from_numpy(img_1_ori).cuda())
                Flag_1 = bool(topk_acc_metric(GT.cpu().numpy(), predict_1.cpu().detach().numpy(), args.k_value))
                predict_2 = model(torch.from_numpy(img_2_ori).cuda())
                Flag_2 = bool(topk_acc_metric(GT.cpu().numpy(), predict_2.cpu().detach().numpy(), args.k_value))
                predict_3 = model(torch.from_numpy(img_3_ori).cuda())
                Flag_3 = bool(topk_acc_metric(GT.cpu().numpy(), predict_3.cpu().detach().numpy(), args.k_value))
                ############
                success_count = success_count + 1
                
                success_img_index.append(ith)
                if Flag_1:
                    success_blurring_index_1.append(ith)
                if Flag_2:
                    success_blurring_index_2.append(ith)
                if Flag_3:
                    success_blurring_index_3.append(ith)
                for key in All_res.keys():
                    if key >= iteration:
                        index_success[key] += 1
            index = index + 1
            if index == all_samples:
                break


        for key in All_res.keys():
            save_result(key, All_res[key]['success_c_list'], 
                    index, 
                    args, 
                    success_img_index, 
                    All_res[key]['success_modifier_norm_list'],
                    All_res[key]['success_modifier_norm_index_0'], 
                    All_res[key]['success_modifier_norm_index_1'],
                    All_res[key]['success_modifier_norm_index_2'], 
                    All_res[key]['success_modifier_norm_index_3'],
                    All_res[key]['success_modifier_norm_index_4'], 
                    All_res[key]['success_modifier_norm_index_5'],
                    success_blurring_index_1, 
                    success_blurring_index_2,
                    success_blurring_index_3,
                    All_res[key]['Eious1'],
                    All_res[key]['Eious2'], 
                    All_res[key]['Eious3'], 
                    All_res[key]['Eious4'],
                    All_res[key]['bafracs'], 
                    All_res[key]['befores'], 
                    All_res[key]['afters'], 
                    All_res[key]['GT_lists'], 
                    All_res[key]['pred_idx_ls'], 
                    All_res[key]['times_costs'])
                   
    if args.app == 'baseline_rank':
        All_res = {}
        for iter in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:#
            index_success[iter] = 0
            All_res[iter] = {}
            for key in ['success_c_list',
                        'success_modifier_norm_list', 
                        'success_modifier_norm_index_0', 
                        'success_modifier_norm_index_1', 
                        'success_modifier_norm_index_2',
                        'success_modifier_norm_index_3', 
                        'success_modifier_norm_index_4', 
                        'success_modifier_norm_index_5', 
                        'Eious_oris_T', 
                        'Eious_min_or_GT_maxs_T',
                        'Target_idxs_T',
                        'Pred_Topks_T', 
                        'Befores_T', 
                        'Afters_T', 
                        'Label_Fracs_T', 
                        'Bs_befores_T', 
                        'Bs_afters_T', 
                        'Bs_fracs_T',
                        'times_costs']:
                        All_res[iter][key] = []

        
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()
            print('\n')
            data = data.to(device)
            target, targets_zeros = generate_target_zeros_3_cases(model, data, GT.cpu().detach().numpy(),
                                                                    args.label_difficult, k=args.k_value)
            targets_zeros = torch.from_numpy(targets_zeros).to(device)

            ######MLAP rank I baseline from the paper 'Multi-Label Adversarial Perturbations'
            purtabed_img_out, modifier_out, attack_success, c_list, sub_res, iteration = \
                rank2(index_success, args, model, data, targets_zeros, args.k_value, args.eps, args.maxiter, args.boxmax, args.boxmin, \
                        GT, device=device, lr=args.lr_attack)
            for key in sub_res.keys():
                All_res[key]['success_c_list'].append(sub_res[key]['c_list'])

                if sub_res[key]['norm'] is not None:
                    All_res[key]['success_modifier_norm_list'].append(sub_res[key]['norm'])

                if sub_res[key]['index_success_0'] is not None:
                    All_res[key]['success_modifier_norm_index_0'].append(sub_res[key]['index_success_0'])
                if sub_res[key]['index_success_1'] is not None:
                    All_res[key]['success_modifier_norm_index_1'].append(sub_res[key]['index_success_1'])
                if sub_res[key]['index_success_2'] is not None:
                    All_res[key]['success_modifier_norm_index_2'].append(sub_res[key]['index_success_2'])
                if sub_res[key]['index_success_3'] is not None:
                    All_res[key]['success_modifier_norm_index_3'].append(sub_res[key]['index_success_3'])
                if sub_res[key]['index_success_4'] is not None:
                    All_res[key]['success_modifier_norm_index_4'].append(sub_res[key]['index_success_4'])
                if sub_res[key]['index_success_5'] is not None:
                    All_res[key]['success_modifier_norm_index_5'].append(sub_res[key]['index_success_5'])

                All_res[key]['Eious_oris_T'].append(sub_res[key]['Eious_iou'])
                All_res[key]['Eious_min_or_GT_maxs_T'].append(sub_res[key]['Eious_min_or_GT_max'])
                All_res[key]['Target_idxs_T'].append(sub_res[key]['target_idx'])
                All_res[key]['Pred_Topks_T'].append(sub_res[key]['pred_topk'])
                All_res[key]['Befores_T'].append(sub_res[key]['before'])
                All_res[key]['Afters_T'].append(sub_res[key]['after'])
                All_res[key]['Label_Fracs_T'].append(sub_res[key]['frac'])
                All_res[key]['Bs_befores_T'].append(sub_res[key]['Bs_before'])
                All_res[key]['Bs_afters_T'].append(sub_res[key]['Bs_after'])
                All_res[key]['Bs_fracs_T'].append(sub_res[key]['Bs_frac'])
                All_res[key]['times_costs'].append(sub_res[key]['cost_time'])


            if attack_success:
                success_count = success_count + 1
                success_img_index.append(ith)
                for key in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
                    if key >= iteration:
                        index_success[key] += 1
            print('success:{}/{}'.format(success_count, index + 1))
            index = index + 1
            if index == 3000:
                break

        for key in All_res.keys():
            save_result_T(key,
                          All_res[key]['success_c_list'],
                          index,
                          args,
                          success_img_index,
                          All_res[key]['success_modifier_norm_list'],
                          All_res[key]['success_modifier_norm_index_0'],
                          All_res[key]['success_modifier_norm_index_1'],
                          All_res[key]['success_modifier_norm_index_2'], 
                          All_res[key]['success_modifier_norm_index_3'],
                          All_res[key]['success_modifier_norm_index_4'], 
                          All_res[key]['success_modifier_norm_index_5'],
                          success_blurring_index_1, 
                          success_blurring_index_2,
                          success_blurring_index_3,
                          All_res[key]['Eious_oris_T'],
                          All_res[key]['Eious_min_or_GT_maxs_T'],
                          All_res[key]['Target_idxs_T'],
                          All_res[key]['Pred_Topks_T'],
                          All_res[key]['Befores_T'],
                          All_res[key]['Afters_T'],
                          All_res[key]['Label_Fracs_T'],
                          All_res[key]['Bs_befores_T'],
                          All_res[key]['Bs_afters_T'],
                          All_res[key]['Bs_fracs_T'],
                          All_res[key]['times_costs'])

    torch.cuda.empty_cache()