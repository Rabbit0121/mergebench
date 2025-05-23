# SLERP
# https://github.com/Digitous/LLM-SLERP-Merge/blob/main/slerpmergelm.py
import copy
import numpy as np
import torch


def lerp(t:float, v0:torch.Tensor, v1:torch.Tensor):
    return (1 - t) * v0 + t * v1

def compute_slerp(t:float, v0:torch.Tensor, v1:torch.Tensor, DOT_THRESHOLD=0.9995):
    epsilon = 1e-10
    # Convert tensors to a common format, float32
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)
    # Convert tensors to numpy arrays
    c = False
    if not isinstance(v0, np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1, np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles    
    norm_v0 = np.linalg.norm(v0)
    norm_v1 = np.linalg.norm(v1)

    if norm_v0 > epsilon:
        v0 = v0 / norm_v0
    else:
        print(f"Warning: Norm of v0 is very small ({norm_v0}). Skipping normalization.")

    if norm_v1 > epsilon:
        v1 = v1 / norm_v1
    else:
        print(f"Warning: Norm of v1 is very small ({norm_v1}). Skipping normalization.")
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        if(type(res) == np.ndarray):
            return torch.from_numpy(res)
        if(type(res) == np.float64):
            return torch.tensor(float(res))
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy

    del v0_copy, v1_copy
    del v1

    if c:
        try:
            res = torch.from_numpy(v2)
        except:
            res = torch.tensor(v2)
    else:
        res = v2
    return res

def SLERP(state_dict_1:dict, state_dict_2:dict, coefficient:float):
    state_dict_3 = copy.deepcopy(state_dict_1)
    for key1, key2 in zip(state_dict_1.keys(), state_dict_2.keys()):
        if(key1 != key2):
            break
        state_dict_3[key1] = compute_slerp(t=coefficient,v0=state_dict_1[key1],v1=state_dict_2[key2])
        # print(key1, type(state_dict_1[key1]))
    return state_dict_3