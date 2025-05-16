# DARE
# https://github.com/yule-BUAA/MergeLM/blob/main/model_merging_methods/mask_weights_utils.py
# drop and rescale
# process each key-val individually.
import torch
from algorithms.task_arithmetic import TaskVector


def DARE(task_vector: TaskVector, mask_rate: float, use_rescale: bool, mask_strategy: str):
    # check the range of mask_rate
    # print("PyTorch random tensor:", torch.rand(3))
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    for key in task_vector.vector:
        input_tensor = task_vector.vector[key]
        # random: randomly select %k data and set them to 0
        if mask_strategy == "random":
            # mask only has 0 or 1
            mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
            masked_input_tensor = input_tensor * (1 - mask)
        # magnitude: set the minimum %k data to 0
        else:
            assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
            original_shape = input_tensor.shape
            
            input_tensor = input_tensor.flatten()
            num_mask_params = int(len(input_tensor) * mask_rate)
            # print(key,num_mask_params)
            if(len(input_tensor)==1 or num_mask_params ==0 ):
                masked_input_tensor = input_tensor
                continue
                # print(len(input_tensor), num_mask_params)
            # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
            kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
            # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
            mask = input_tensor.abs() <= kth_values
            masked_input_tensor = input_tensor * (~mask)
            masked_input_tensor = masked_input_tensor.reshape(original_shape)
        # rescale: /(1-rate)
        if use_rescale and mask_rate != 1.0 and len(input_tensor)!=1:
            masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
        task_vector.vector[key] = masked_input_tensor
    return task_vector