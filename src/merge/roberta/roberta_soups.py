import sys
from merge_class import roberta_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from utils.files import write_to_csv
from merge_utils import evaluate_nlp, evaluate_upstream_model_nlp
from algorithms.soups import ModelSoups
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

datasets_list = ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte']
#               mas'core   acc    f1/acc  f1/acc  acc   acc     acc 
for model_path_list, model_type, model_tokenizer_path in zip(path_list, type_list, tokenizer_list):
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
            ckpt_list.append(model1.model.roberta.state_dict())
            ckpt_list.append(model2.model.roberta.state_dict())
            # soups
            soups = ModelSoups(ckpt_list)
            merged_ckpt = soups.run()
            merged_model = roberta_merged_model(model_list, False)
            merged_model.load_weight(merged_ckpt)
            # evaluate
            acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
            _, acc_merged = evaluate_nlp(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, tokenizer=tokenizer)
            print(_, acc_merged)
            # import pdb; pdb.set_trace()
            headers=["datasets", "acc_f1_mcc_upstream", "acc_f1_mcc_merged"]
            data={"datasets":model_type + str(dataset_list), "acc_f1_mcc_upstream":acc_upstream, "acc_f1_mcc_merged":acc_merged}
            result_path = '../results/new_merge_2/roberta/soups.csv'
            write_to_csv(result_path, headers, data)