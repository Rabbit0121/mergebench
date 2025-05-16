# Task Arithmetic
# from https://github.com/mlfoundations/task_vectors/blob/main/src/task_vectors.py
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os
CUBLAS_WORKSPACE_CONFIG=':4096:8'
class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        # self.vector:dict
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                # fix: Pass weights directly instead of paths
                # pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                # finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                pretrained_state_dict = pretrained_checkpoint
                finetuned_state_dict = finetuned_checkpoint
                self.vector = {}
                # print(finetuned_state_dict.keys())
                for key in pretrained_state_dict:
                    # fix: put them in the same devices
                    # print(pretrained_state_dict[key].dtype)
                    # import pdb; pdb.set_trace()
                    finetuned_state_dict[key]=finetuned_state_dict[key].to(device)
                    pretrained_state_dict[key]=pretrained_state_dict[key].to(device)
                    # num_batchs_tracked会被过滤
                    # if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        # print(key)
                        # import pdb; pdb.set_trace()
                        # continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
                    # print(key)
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                # print(key)
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    # def __mul__(self, scalar):
    #     """Multiply a taskvector and a scalar(tensor)"""
    #     new_vector = {}
    #     for key in self.vector:
    #         # print(key)
    #         # if key not in other.vector:
    #         #     print(f'Warning, key {key} is not present in both task vectors.')
    #         #     continue
    #         new_vector[key] = self.vector[key] * scalar
    #     return TaskVector(vector=new_vector)


    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            # pretrained_model = torch.(pretrained_checkpoint)
            # fix: Load weights directly
            # pretrained_model.load_state_dict(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_checkpoint
            for key in pretrained_state_dict:
                # fix: put them in the same devices
                # fix: igonre num_batches_tracked
                # print(key,type(key))
                # num_batchs_tracked会被过滤
                # if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                #     continue
                self.vector[key]=self.vector[key].to(device)
                pretrained_state_dict[key]=pretrained_state_dict[key].to(device)
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue                
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        # pretrained_model.load_state_dict(new_state_dict, strict=False)
        return new_state_dict