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
from merge_gpt2 import gpt2
import logging

# 禁用
# Some weights of BertForSequenceClassification were not initialized from the model checkpoint 
# at /data4/xxx/0407/models/hf_bert/bert-base-uncased/model and are newly initialized: ['classifier.bias', 'classifier.weight']
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
    acc_upstream_dict = {}
    for i in range(len(model_path_list)):
        model = GPT2SequenceClassifier.load_from_checkpoint(model_path_list[i])
        model.model.config.pad_token_id = tokenizer.eos_token_id
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model_nlp(model, datasets_list[i], 32, 8, tokenizer)
    print(acc_upstream_dict)
    # import pdb; pdb.set_trace()
    # continue
    # 两两分配
    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            model_list = []  
            ckpt_list = []
            dataset_list = []
            dataset_list.append(datasets_list[i])
            dataset_list.append(datasets_list[j])
            model1 = GPT2SequenceClassifier.load_from_checkpoint(model_path_list[i])
            model2 = GPT2SequenceClassifier.load_from_checkpoint(model_path_list[j])
            # print(model1.state_dict().keys())
            # print(model1.model.transformer.state_dict().keys())
            # print(model1.model.score.state_dict().keys())
            # import pdb; pdb.set_trace()
            model_list.append(model1)
            model_list.append(model2)
            ckpt_list.append(model1.model.transformer.state_dict())
            ckpt_list.append(model2.model.transformer.state_dict())
            # soups
            soups = ModelSoups(ckpt_list)
            merged_ckpt = soups.run()
            merged_model = gpt2_merged_model(model_list, False, tokenizer.eos_token_id)
            merged_model.load_weight(merged_ckpt)
            # evaluate
            acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
            _, acc_merged = evaluate_nlp(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, tokenizer=tokenizer)
            print(acc_upstream, acc_merged)
            # import pdb; pdb.set_trace()
            headers=["datasets", "acc_f1_mcc_upstream", "acc_f1_mcc_merged"]
            data={"datasets":model_type + str(dataset_list), "acc_f1_mcc_upstream":acc_upstream, "acc_f1_mcc_merged":acc_merged}
            result_path = '../results/new_merge_2/gpt2/soups.csv'
            write_to_csv(result_path, headers, data)