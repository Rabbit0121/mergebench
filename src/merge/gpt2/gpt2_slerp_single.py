import sys
from merge_class import gpt2_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from utils.files import write_to_csv
from merge_utils import evaluate_nlp, evaluate_upstream_model_nlp
from algorithms.soups import ModelSoups
from ft_class import GPT2SequenceClassifier
from transformers import GPT2Tokenizer
from merge_gpt2_single import gpt2
import logging
from algorithms.soups import ModelSoups
from ft_class import RobertaSequenceClassifier
from transformers import RobertaTokenizer
from merge_roberta_single import roberta_base, roberta_large
import logging
from algorithms.soups import ModelSoups
from algorithms.task_arithmetic import TaskVector
from algorithms.dare import DARE
from algorithms.regmean import RegMean
from algorithms.ties import TIES
from algorithms.slerp import SLERP


logging.getLogger('transformers').setLevel(logging.ERROR)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [gpt2]
type_list = ['gpt2']
tokenizer_list  = ['/data4/xxx/0407/models/hf_gpt2/gpt2']

datasets_list = ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte']
#               mas'core   acc    f1/acc  f1/acc  acc   acc     acc 
for model_path_list, model_type, model_tokenizer_path in zip(path_list, type_list, tokenizer_list):
    tokenizer = GPT2Tokenizer.from_pretrained(model_tokenizer_path + '/tokenizer')
    tokenizer.pad_token = tokenizer.eos_token
    if(model_type == 'bert-base-uncased'):
        continue
    acc_upstream_dict = {}
    for i in range(len(model_path_list)):
        for dataset in model_path_list[i].keys():
            for j in range(len(model_path_list[i][dataset])):
                model = GPT2SequenceClassifier.load_from_checkpoint(model_path_list[i][dataset][j])
                model.model.config.pad_token_id = tokenizer.eos_token_id
                acc_upstream_dict[str(dataset) + '_' + str(j+1)] = evaluate_upstream_model_nlp(model, dataset, 32, 4, tokenizer)

    for i in range(len(model_path_list)):
        for dataset in model_path_list[i].keys():
            model_list = [] 
            ckpt_list = []
            for j in range(len(model_path_list[i][dataset])):
                model_path = model_path_list[i][dataset][j]
                model = GPT2SequenceClassifier.load_from_checkpoint(model_path)
                model_list.append(model)
                ckpt_list.append(model.state_dict())
            for degree in range(1, 10, 1):
                degree = degree / 10
                merged_ckpt = SLERP(ckpt_list[0], ckpt_list[1], degree)
                merged_model = torch.load(f'/data4/xxx/0407/ftnew/hf_gpt2/{dataset}/{model_type}_pretrain.pth')
                soups = ModelSoups(ckpt_list)
                merged_ckpt = soups.run()
                merged_model.load_state_dict(merged_ckpt)
                # evaluate
                acc_merged = evaluate_upstream_model_nlp(merged_model, dataset, batch_size=32, num_workers=4, tokenizer=tokenizer)
                acc_upstream = [ acc_upstream_dict[str(dataset) + '_' + str(1)], acc_upstream_dict[str(dataset) + '_' + str(2)] ]
                # print(acc_upstream_dict, acc_merged)
                headers=["datasets", "acc_upstream", "acc_merged"]
                data={"datasets":model_type + dataset, "acc_upstream":acc_upstream, "acc_merged":acc_merged}
                result_path = '../results/new_merge_2_resetbn/gpt2/single_slerp.csv'
                write_to_csv(result_path, headers, data)