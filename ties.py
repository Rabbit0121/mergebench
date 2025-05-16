# TIES
# Trim, Elect Sign and Disjoint Merge
# https://github.com/prateeky2806/ties-merging
# flatten all key-vals and process them
# https://github.com/EnnengYang/AdaMerging/blob/main/src/ties_merging_utils.py#L187
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy
from typing import List, OrderedDict
import torch
from algorithms.task_arithmetic import TaskVector


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # num_batchs_tracked need to be considered
    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    
    kth_values, _ = M.abs().cpu().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    # M.abs() = M.abs().to('cuda')
    kth_values = kth_values.to('cuda')
    mask = M.abs() >= kth_values
    del kth_values 
    torch.cuda.empty_cache()
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        selected_entries_cpu = selected_entries.cpu()

        # 在 CPU 上计算非零元素数量
        non_zero_counts_cpu = (selected_entries_cpu != 0).sum(dim=0).float()

        # 将结果送回 GPU
        non_zero_counts = non_zero_counts_cpu.cuda()
        # non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        # print("aaaaaaaaaaaaaa")
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs

def disjoint_merge_split(Tensor, merge_func, sign_to_mult):
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return selected_entries, disjoint_aggs


def ties_merging_split(
        flat_task_checks,
        reset_thresh=None,
        merge_func="",
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    selected_entries, merged_tv = disjoint_merge_split(updated_checks, merge_func, final_signs)

    return selected_entries, merged_tv


def get_ties_ckpt(task_vector_list_temp, pre):
    task_vector_list = copy.deepcopy(task_vector_list_temp)
    flat_task_vectors = torch.vstack([state_dict_to_vector(task_vector.vector) for task_vector in task_vector_list])
    selected_entries, _ = ties_merging_split(flat_task_vectors, reset_thresh=0.2, merge_func="sum",)

    ties_task_vectors = []
    for vector_ in selected_entries:
        t_state_dict = vector_to_state_dict(vector_, pre, remove_keys=[])
        # ref_model = torch.load(pre)
        # ref_model.load_state_dict(t_state_dict, strict=False)
        ties_task_vectors.append(TaskVector(vector = t_state_dict))
    return ties_task_vectors



def TIES(task_vector_list_temp:List[TaskVector], mask_rate:float, merge_func:str, pre:dict, scaling_coef:float):
    print("PyTorch random tensor:", torch.rand(3))
    task_vector_list = copy.deepcopy(task_vector_list_temp)
    flat_task_vectors = torch.vstack([state_dict_to_vector(task_vector.vector) for task_vector in task_vector_list])
    # flat_ft = torch.vstack([state_dict_to_vector(task_vector) for task_vector in task_vector_list])
    # flat_ptm = state_dict_to_vector(pre)
    # flat_task_vectors = flat_ft - flat_ptm
    del task_vector_list  # 释放不再需要的变量
    torch.cuda.empty_cache()
    all_checks = flat_task_vectors.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=mask_rate, return_mask=False
    )
    del all_checks  # 释放克隆的副本
    torch.cuda.empty_cache()
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    
    assert final_signs is not None
    
    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_task_vectors = disjoint_merge(updated_checks, merge_func, final_signs)
    del updated_checks  # 释放中间结果
    torch.cuda.empty_cache()
    merged_task_vectors = merged_task_vectors.to('cpu')

    merged_new = state_dict_to_vector(pre) + scaling_coef * merged_task_vectors
    del merged_task_vectors  # 释放聚合结果
    torch.cuda.empty_cache()
    # convert the flat merged checkpoint to a state dict
    merged_state_dict = vector_to_state_dict(merged_new, pre)

    return merged_state_dict