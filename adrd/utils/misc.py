import numpy as np
import sys, tqdm
import torch
import torch.nn.functional as F
from numpy import interp
from collections.abc import Sequence
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import balanced_accuracy_score, precision_score
import warnings
import inspect

_depth = lambda L: isinstance(L, (Sequence, np.ndarray)) and max(map(_depth, L)) + 1


def get_metrics(y_true, y_pred, scores, mask):
    ''' ... '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        masked_y_true = y_true[np.where(mask == 1)]
        masked_y_pred = y_pred[np.where(mask == 1)]
        masked_scores = scores[np.where(mask == 1)]

        # metrics that are based on predictions
        try:
            cnf = confusion_matrix(masked_y_true, masked_y_pred)
            TN, FP, FN, TP = cnf.ravel()
            TNR = TN / (TN + FP)
            FPR = FP / (FP + TN)
            FNR = FN / (FN + TP)
            TPR = TP / (TP + FN)
            N = TN + TP + FN + FP
            S = (TP + FN) / N
            P = (TP + FP) / N
            acc = (TN + TP) / N
            sen = TP / (TP + FN)
            spc = TN / (TN + FP)
            prc = TP / (TP + FP)
            f1s = 2 * (prc * sen) / (prc + sen)
            mcc = (TP / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P))

            # metrics that are based on scores,
            try:
                auc_roc = roc_auc_score(masked_y_true, masked_scores)
            except:
                auc_roc = 0
            try:
                auc_pr = average_precision_score(masked_y_true, masked_scores)
            except:
                auc_pr = 0
                
            bal_acc = balanced_accuracy_score(masked_y_true, masked_y_pred)
        except:
            cnf, acc, bal_acc, prc, sen, spc, f1s, mcc, auc_roc, auc_pr = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

        # construct the dictionary of all metrics
        met = {}
        met['Confusion Matrix'] = cnf
        met['Accuracy'] = acc
        met['Balanced Accuracy'] = bal_acc
        met['Precision'] = prc
        met['Sensitivity/Recall'] = sen
        met['Specificity'] = spc
        met['F1 score'] = f1s
        met['MCC'] = mcc
        met['AUC (ROC)'] = auc_roc
        met['AUC (PR)'] = auc_pr
        
    return met


def get_metrics_multitask(y_true, y_pred, scores, mask):
    ''' ... '''
    if type(y_true) is dict:
        met: dict[str, dict[str, float]] = dict()
        for k in y_true.keys():
            met[k] = get_metrics(y_true[k], y_pred[k], scores[k], mask[k])
    else:
        met = []
        for i in range(len(y_true[0])):
            met.append(get_metrics(y_true[:, i], y_pred[:, i], scores[:, i], mask[:, i]))
    return met


def print_metrics(met):
    ''' ... '''
    for k, v in met.items():
        if k not in ['Confusion Matrix']:
            print('{}:\t{:.4f}'.format(k, v).expandtabs(20))


def print_metrics_multitask(met):
    ''' ... '''
    if type(met) is dict:
        lbl_ks = list(met.keys())
        met_ks = met[lbl_ks[0]].keys()
        for met_k in met_ks:
            if met_k not in ['Confusion Matrix']:
                msg = '{}:\t' + '{:.4f}    ' * len(met)
                val = [met[lbl_k][met_k] for lbl_k in lbl_ks]
                msg = msg.format(met_k, *val)
                msg = msg.replace('nan', '------')
                print(msg.expandtabs(20))
    else:
        for k in met[0]:
            if k not in ['Confusion Matrix']:
                msg = '{}:\t' + '{:.4f}    ' * len(met)
                val = [met[i][k] for i in range(len(met))]
                msg = msg.format(k, *val)
                msg = msg.replace('nan', '------')
                print(msg.expandtabs(20))


def pr_interp(rc_, rc, pr):

    pr_ = np.zeros_like(rc_)
    locs = np.searchsorted(rc, rc_)

    for idx, loc in enumerate(locs):
        l = loc - 1
        r = loc
        r1 = rc[l] if l > -1 else 0
        r2 = rc[r] if r < len(rc) else 1
        p1 = pr[l] if l > -1 else 1
        p2 = pr[r] if r < len(rc) else 0

        t1 = (1 - p2) * r2 / p2 / (r2 - r1) if p2 * (r2 - r1) > 1e-16 else (1 - p2) * r2 / 1e-16
        t2 = (1 - p1) * r1 / p1 / (r2 - r1) if p1 * (r2 - r1) > 1e-16 else (1 - p1) * r1 / 1e-16
        t3 = (1 - p1) * r1 / p1 if p1 > 1e-16 else (1 - p1) * r1 / 1e-16

        a = 1 + t1 - t2
        b = t3 - t1 * r1 + t2 * r1
        pr_[idx] = rc_[idx] / (a * rc_[idx] + b)

    return pr_


def get_roc_info(y_true_all, scores_all):

    fpr_pt = np.linspace(0, 1, 1001)
    tprs, aucs = [], []

    for i in range(len(y_true_all)):
        y_true = y_true_all[i]
        scores = scores_all[i]
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=scores, drop_intermediate=True)
        tprs.append(interp(fpr_pt, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))

    tprs_mean = np.mean(tprs, axis=0)
    tprs_std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tprs_mean + tprs_std, 1)
    tprs_lower = np.maximum(tprs_mean - tprs_std, 0)
    auc_mean = auc(fpr_pt, tprs_mean)
    auc_std = np.std(aucs)
    auc_std = 1 - auc_mean if auc_mean + auc_std > 1 else auc_std

    rslt = {
        'xs': fpr_pt,
        'ys_mean': tprs_mean,
        'ys_upper': tprs_upper,
        'ys_lower': tprs_lower,
        'auc_mean': auc_mean,
        'auc_std': auc_std
    }

    return rslt


def get_pr_info(y_true_all, scores_all):

    rc_pt = np.linspace(0, 1, 1001)
    rc_pt[0] = 1e-16
    prs = []
    aps = []

    for i in range(len(y_true_all)):
        y_true = y_true_all[i]
        scores = scores_all[i]
        pr, rc, _ = precision_recall_curve(y_true=y_true, probas_pred=scores)
        aps.append(average_precision_score(y_true=y_true, y_score=scores))
        pr, rc = pr[::-1], rc[::-1]
        prs.append(pr_interp(rc_pt, rc, pr))

    prs_mean = np.mean(prs, axis=0)
    prs_std = np.std(prs, axis=0)
    prs_upper = np.minimum(prs_mean + prs_std, 1)
    prs_lower = np.maximum(prs_mean - prs_std, 0)
    aps_mean = np.mean(aps)
    aps_std = np.std(aps)
    aps_std = 1 - aps_mean if aps_mean + aps_std > 1 else aps_std

    rslt = {
        'xs': rc_pt,
        'ys_mean': prs_mean,
        'ys_upper': prs_upper,
        'ys_lower': prs_lower,
        'auc_mean': aps_mean,
        'auc_std': aps_std
    }

    return rslt


def get_and_print_metrics(mdl, dat):
    ''' ... '''
    y_pred = mdl.predict(dat.x)
    y_prob = mdl.predict_proba(dat.x)
    met_all = get_metrics(dat.y, y_pred, y_prob)

    for k, v in met_all.items():
        if k not in ['Confusion Matrix']:
            print('{}:\t{:.4f}'.format(k, v).expandtabs(20))


def get_and_print_metrics_multitask(mdl, dat):
    ''' ... '''
    y_pred = mdl.predict(dat.x)
    y_prob = mdl.predict_proba(dat.x)
    met = get_metrics_multitask(dat.y, y_pred, y_prob)
    print_metrics_multitask(met)
    

def split_dataset(dat, ratio=.8, seed=0):

    len_trn = int(np.round(len(dat) * .8))
    len_vld = len(dat) - len_trn
    dat_trn, dat_vld = torch.utils.data.random_split(
        dat, (len_trn, len_vld),
        generator=torch.Generator().manual_seed(0)
    )
    return dat_trn, dat_vld


def l1_regularizer(model, lambda_l1=0.01):
    ''' LASSO '''

    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1


class ProgressBar(tqdm.tqdm):
    
    def __init__(self, total, desc, file=sys.stdout):
        
        super().__init__(total=total, desc=desc, ascii=True, bar_format='{l_bar}{r_bar}', file=file)
    
    def update(self, batch_size, to_disp):
        
        postfix = {}
        
        for k, v in to_disp.items():
            if k == 'cnf':           
                postfix[k] = v.__repr__().replace('\n', '')
            
            else:  
                postfix[k] = '{:.6f}'.format(v.cpu().numpy())
            
        self.set_postfix(postfix)
        super().update(batch_size)


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def convert_args_kwargs_to_kwargs(func, args, kwargs):
    """ ... """
    signature = inspect.signature(func)
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments