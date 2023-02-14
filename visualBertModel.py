from transformers import AutoTokenizer, VisualBertForVisualReasoning, DataCollatorWithPadding
import torch
from utils import build_dataloaders
from NLPtests.imagePreProcessing import ImagePreProcessing as ImgPreProc

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
    device = torch.device(dev) 

class VisualBertModel:
    checkpoint = 'uclanlp/visualbert-nlvr2-coco-pre'

    def __init_(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = VisualBertForVisualReasoning.from_pretrained(self.checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def tokenize_function(self, ds):
        return self.tokenizer(ds['Text'], truncation=True)

    def process_ds(self, datasets):
        # Tokenize the datasets
        tokenized_ds = datasets.map(self.tokenize_function, batched=True)

        # Rename the columns
        tokenized_ds = tokenized_ds.rename_column("Label", "labels")
        #tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        self.tokenized_ds = tokenized_ds

        return tokenized_ds

    

    def test(self):
        train_dataloader, eval_dataloader, test_dataloader = build_dataloaders(self.tokenized_ds, self.data_collator)
        self.model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                print(batch)
                break
                #outputs = self.model(**batch)
                #print(outputs)





def get_visual_embeds(imagesPaths):
    imgPreProc = ImgPreProc()
    images = imgPreProc.openAndConvertBGR(imagesPaths)
    images, batched_inputs = imgPreProc.prepare_image_inputs(images)
    features = imgPreProc.get_features(images)
    proposals = imgPreProc.get_proposals(images, features)
    box_features, features_list = imgPreProc.get_box_features(features, proposals, len(imagesPaths))
    pred_class_logits, pred_proposal_deltas = imgPreProc.get_prediction_logits(features_list, proposals)
    boxes, scores, image_shapes = imgPreProc.get_box_scores(pred_class_logits, pred_proposal_deltas, proposals)
    output_boxes = [imgPreProc.get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
    temp = [imgPreProc.select_boxes(output_boxes[i], scores[i]) for i in range(len(scores))]
    keep_boxes, max_conf = [],[]
    for keep_box, mx_conf in temp:
        keep_boxes.append(keep_box)
        max_conf.append(mx_conf)
    MIN_BOXES=10
    MAX_BOXES=100
    keep_boxes = [imgPreProc.filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
    visual_embeds = [imgPreProc.get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]
    return visual_embeds