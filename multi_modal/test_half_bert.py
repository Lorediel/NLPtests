from NLPtests.only_text.myBert import BertModel
from NLPtests.multi_modal.clipModel import ClipModel
from utils import *
from NLPtests.FakeNewsDataset import collate_fn
from tqdm.auto import tqdm
import torch


class test_half():

    def __init__(self, bert_path, clip_path):
        self.bert = load_model(BertModel(), bert_path)
        self.clip = load_model(ClipModel(), clip_path)

    def eval(self, ds, batch_size=8, tokenization_strategy="first", print_confusion_matrix=False):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.bert.eval()
        self.clip.eval()
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, collate_fn = collate_fn
        )
        total_preds = []
        total_labels = []
        progress_bar = tqdm(range(len(dataloader)))
        with torch.no_grad():
            for batch in dataloader:
                texts = batch["text"]
                images_list = batch["images"]
                labels = batch["label"]
                nums_images = batch["nums_images"]


                t_inputs = self.bert.model.tokenizer(texts, return_tensors="pt", padding = True, truncation=True)

                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                
                logits, probs = self.bert(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"]
                )

                final_preds = [-1] * len(batch)
                bert_preds = torch.argmax(logits, dim=1).tolist()
                to_reclassify = []
                i=0
                for pred in bert_preds:
                    if pred != 0 and pred != 1:
                        to_reclassify.append(i)
                    else:
                        final_preds[i] = pred
                    i+=1

                texts = [texts[i] for i in to_reclassify]
                images_list = [images_list[i] for i in to_reclassify]
                labels = [labels[i] for i in to_reclassify]
                nums_images = [nums_images[i] for i in to_reclassify]

                random_images_list = []
                base = 0
                for i in range(len(nums_images)):
                    if nums_images[i] == 1:
                        random_images_list.append(images_list[base])
                        base += nums_images[i]
                        continue
                    random_index = random.randint(0, nums_images[i]-1)
                    sublist = images_list[base:base+nums_images[i]]
                    random_images_list.append(sublist[random_index])
                    base += nums_images[i]
                
                t_inputs = self.clip.model.tokenizer(texts, return_tensors="pt", padding = True, truncation=True)
                i_inputs = self.clip.model.processor(images = random_images_list, return_tensors="pt", padding=True)
                
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)
                
                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)
                logits, probs = self.model(
                    input_ids=t_inputs["input_ids"],
                    attention_mask=t_inputs["attention_mask"],
                    pixel_values=i_inputs.pixel_values
                )

                clip_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                j = 0
                for index in to_reclassify:
                    final_preds[index] = clip_preds[j]
                    j+=1
                
                total_preds += list(final_preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_preds, total_labels)
        if print_confusion_matrix:
            display_confusion_matrix(total_preds, total_labels)
        return metrics
