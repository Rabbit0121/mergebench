import sys
from merge_class import bert_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from utils.files import write_to_csv
from merge_utils import evaluate_nlp, evaluate_upstream_model_nlp
from algorithms.regmean import RegMean
from ft_class import BertSequenceClassifier
from transformers import BertTokenizer
from merge_bert import bert_base, bert_large
import logging


logging.getLogger('transformers').setLevel(logging.ERROR)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [bert_base, bert_large]
type_list = ['bertbase', 'bertlarge']
tokenizer_list  = ['/data4/xxx/0407/models/hf_bert/bert-base-uncased', 
                   '/data4/xxx/0407/models/hf_bert/bert-large-uncased']
# metric_list = [acc,  mas'core,(mis)matacc,   f1/acc,  acc,   f1/acc, acc,   acc]
# datasets_list = ['sst2', 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'wnli']
datasets_list = ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte']
#               mas'core   acc    f1/acc  f1/acc  acc   acc     acc 
for model_path_list, model_type, model_tokenizer_path in zip(path_list, type_list, tokenizer_list):
    tokenizer = BertTokenizer.from_pretrained(model_tokenizer_path + '/tokenizer')
    acc_upstream_dict = {}
    for i in range(len(model_path_list)):
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model_nlp(BertSequenceClassifier.load_from_checkpoint(model_path_list[i]), datasets_list[i], 32, 4, tokenizer)

    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            model_list = []  
            ckpt_list = []
            dataset_list = []
            dataset_list.append(datasets_list[i])
            dataset_list.append(datasets_list[j])
            model1 = BertSequenceClassifier.load_from_checkpoint(model_path_list[i])
            model2 = BertSequenceClassifier.load_from_checkpoint(model_path_list[j])
            model_list.append(model1)
            model_list.append(model2)
            ckpt_list.append(model1.model.bert.state_dict())
            ckpt_list.append(model2.model.bert.state_dict())
            # regmean
            regmean = RegMean("hf_bert", model_list, dataset_list, 100, processor=None, tokenizer=tokenizer)
            merged_ckpt = regmean.get_merged_checkpoint()
            merged_model = bert_merged_model(model_list, False)
            merged_model.load_weight(merged_ckpt)
            # evaluate
            acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
            _, acc_merged = evaluate_nlp(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, tokenizer=tokenizer)
            print(acc_upstream, acc_merged)
            headers=["datasets", "acc_f1_mcc_upstream", "acc_f1_mcc_merged"]
            data={"datasets":model_type + str(dataset_list), "acc_f1_mcc_upstream":acc_upstream, "acc_f1_mcc_merged":acc_merged}
            result_path = '../results/new_merge_2/bert/regmean.csv'
            write_to_csv(result_path, headers, data)
