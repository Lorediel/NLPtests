from transformers import AutoTokenizer, VisualBertForVisualReasoning, DataCollatorWithPadding
import torch
from NLPtests.utils import build_dataloaders
from NLPtests.imagePreProcessing_2 import ImagePreProcessing as ImgPreProc
from glob import glob

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
    device = torch.device(dev) 

class VisualBertModel:
    checkpoint = 'uclanlp/visualbert-nlvr2-coco-pre'

    def __init__(self):
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
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels", "ID"])
        self.tokenized_ds = tokenized_ds

        return tokenized_ds

    

    def test(self, images_folder):
        if torch.cuda.is_available(): 
          dev = "cuda:0" 
        else: 
          dev = "cpu" 
        device = torch.device(dev) 
        train_dataloader, eval_dataloader, test_dataloader = build_dataloaders(self.tokenized_ds, self.data_collator)
        self.model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                ID_tensor = batch['ID']
                #images_paths = []
                for i, id in enumerate(ID_tensor):
                  single_image_paths = sorted(glob(f"{images_folder}/**/{id}*.jpg", recursive=True))
                  ve = []
                  for path in single_image_paths:
                    single_visual_embeds = get_visual_embeds(path)
                    ve.append(single_visual_embeds)
                  #images_paths.append(single_image_paths)





def get_visual_embeds(imagesPaths):
    total_length = 0
    for l in imagesPaths:
        total_length += len(l)
    imgPreProc = ImgPreProc()
    images = imgPreProc.openAndConvertBGR(imagesPaths)
    images, batched_inputs = imgPreProc.prepare_image_inputs(images)
    features = imgPreProc.get_features(images)
    proposals = imgPreProc.get_proposals(images, features)
    box_features, features_list = imgPreProc.get_box_features(features, proposals, total_length)
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
    total_visual_embeds = []
    i= 0
    for l in imagesPaths:
        current_ve = visual_embeds[i:i+len(l)]
        i+=len(l)
        total_visual_embeds.append(torch.cat(current_ve),0)
    return visual_embeds