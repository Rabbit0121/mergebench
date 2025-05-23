import sys
from merge_class import roberta_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from utils.files import write_to_csv
from merge_utils import evaluate_nlp, evaluate_upstream_model_nlp
from algorithms.task_arithmetic import TaskVector
from algorithms.ties import TIES
from ft_class import RobertaSequenceClassifier
from transformers import RobertaTokenizer
from merge_roberta import roberta_base, roberta_large
import logging

logging.getLogger('transformers').setLevel(logging.ERROR)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [roberta_base, roberta_large]
type_list = ['robertabase', 'robertalarge']
tokenizer_list  = ['/data4/xxx/0407/models/hf_roberta/roberta-base', 
                   '/data4/xxx/0407/models/hf_roberta/roberta-large']
pre_model_list = [torch.load('/data4/xxx/0407/ftnew/hf_roberta/cola/roberta-base_pretrain.pth'),
                  torch.load('/data4/xxx/0407/ftnew/hf_roberta/cola/roberta-large_pretrain.pth')]
datasets_list = ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte']
#               mas'core   acc    f1/acc  f1/acc  acc   acc     acc 
for model_path_list, model_type, model_tokenizer_path, pre_model in zip(path_list, type_list, tokenizer_list, pre_model_list):
    tokenizer = RobertaTokenizer.from_pretrained(model_tokenizer_path + '/tokenizer')
    acc_upstream_dict = {}
    for i in range(len(model_path_list)):
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model_nlp(RobertaSequenceClassifier.load_from_checkpoint(model_path_list[i]), datasets_list[i], 32, 4, tokenizer)

    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            model_list = []  
            ckpt_list = []
            dataset_list = []
            dataset_list.append(datasets_list[i])
            dataset_list.append(datasets_list[j])
            model1 = RobertaSequenceClassifier.load_from_checkpoint(model_path_list[i])
            model2 = RobertaSequenceClassifier.load_from_checkpoint(model_path_list[j])

            model_list.append(model1)
            model_list.append(model2)
            ckpt_list.append(TaskVector(pre_model.model.roberta.state_dict(), model1.model.roberta.state_dict()))
            ckpt_list.append(TaskVector(pre_model.model.roberta.state_dict(), model2.model.roberta.state_dict()))
            for ties_rate in range(1, 10, 1):
                if(ties_rate != 2):
                    continue
                ties_rate = ties_rate / 10
                for coef in range(1, 30, 1):
                    coef = coef / 10
                    merged_ckpt = TIES(ckpt_list, mask_rate=ties_rate, merge_func="mean", pre = pre_model.model.roberta.state_dict(),  scaling_coef=coef)
                    merged_model = roberta_merged_model(model_list, False)
                    merged_model.load_weight(merged_ckpt)
                    # evaluate
                    acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
                    _, acc_merged = evaluate_nlp(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, tokenizer=tokenizer)
                    print(_, acc_merged)
                    # import pdb; pdb.set_trace()
                    headers=["datasets","ties_rate", "lambda", "acc_f1_mcc_upstream", "acc_f1_mcc_merged"]
                    data={"datasets":model_type + str(dataset_list), "acc_f1_mcc_upstream":acc_upstream, "acc_f1_mcc_merged":acc_merged, "lambda":coef, "ties_rate":ties_rate}
                    result_path = '../results/new_merge_2/roberta/ties.csv'
                    write_to_csv(result_path, headers, data)