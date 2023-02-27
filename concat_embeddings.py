from NLPtests.imagePreProcessing_2 import ImagePreProcessing as ImgPreProc
import torch
from transformers import AdamW, get_scheduler, AutoProcessor, VisionTextDualEncoderModel, AutoTokenizer, BertModel, AdamW, get_scheduler
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from NLPtests.FakeNewsDataset import collate_fn
import torch.nn as nn
from tqdm.auto import tqdm
import os
from NLPtests.utils import compute_metrics

def parts(a, b):
    q, r = divmod(a, b)
    return [q + 1] * r + [q] * (b - r)

def get_visual_embeds(imagesLists, nums_images):
    total_length = len(imagesLists)
    imgPreProc = ImgPreProc()
    converted_images = imgPreProc.convertToBGR(imagesLists)
    images, batched_inputs = imgPreProc.prepare_image_inputs(converted_images)
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
    MIN_BOXES=100
    MAX_BOXES=100
    keep_boxes = [imgPreProc.filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
    visual_embeds = [imgPreProc.get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]
    k= 0
    MAX_BOXES = 100
    final_visual_embeds = []
    i=0
    for l in imagesLists:
        #number of images associated
        b = nums_images[k].item()
        k+=1
        p = parts(MAX_BOXES, b)
        #visual embeds for the images of the same row
        current_ve = visual_embeds[i:i+b]
        j=0
        new_ves = []
        for tensor in current_ve:
            new_ves.append(tensor[:p[j], :])
            j+=1
        new_ves = torch.cat(new_ves, 0)
        final_visual_embeds.append(new_ves)
    
    return final_visual_embeds




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Clip
        self.clip = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        self.processor = AutoProcessor.from_pretrained("clip-italian/clip-italian")
        self.clipTokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")

        # Detectron patch embeddings
        self.patch_embeddings = get_visual_embeds


        # image captioning
        self.vit = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.captionTokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        # bert
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bertTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


        self.relu = nn.ReLU()
        self.linear1 = nn.Sequential(
            nn.Linear(3072, 1536),
            nn.LayerNorm(1536),
            nn.dropout(0.2),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.LayerNorm(1536),
            nn.dropout(0.2),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.dropout(0.1),
            nn.ReLU(),
        )
        self.linear4 = nn.Linear(768, 4)

        self.softmax = nn.Softmax(dim=1)
            

        
    def forward(self, images, input_ids, attention_mask, pixel_values, nums_images):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # clip embeddings
        t_embeddings = self.clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )

        i_embeddings = self.clip.get_image_features(pixel_values = pixel_values) 

        #compute the max of the emnbeddings
        embeddings_images = []
        base = 0
        for i in range(len(nums_images)):
            tensor = i_embeddings[base:base+nums_images[i]]
            max_tensor, _ = torch.max(tensor, dim=0, keepdim=True)
            embeddings_images.append(max_tensor)
            base += nums_images[i]
        
        embeddings_images = torch.cat(embeddings_images, dim=0)
        clip_embeddings = torch.cat((t_embeddings, embeddings_images), dim=1) # clip_embeddings.shape = (batch_size, 1024)
        print(clip_embeddings.shape)


        # detectron
        visual_embeds = self.patch_embeddings(images)
        averages_per_row = []
        for ve in visual_embeds:
            avg_ve = torch.mean(ve, dim=0, keepdim=True)
            averages_per_row.append(avg_ve)
        average_patch_embeddings = torch.cat(averages_per_row, 0) # final_embeddings.shape = (batch_size, 1024)
        print(average_patch_embeddings.shape)
        
        
        # Take the image captions 
        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = self.vit.generate(pixel_values=pixel_values)
        preds = self.captionTokenizer.batch_decode(output_ids, skip_special_tokens=True)
        captions = [pred.strip() for pred in preds]

        t_inputs = self.bertTokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        input_ids = t_inputs.input_ids.to(device)
        attention_mask = t_inputs.attention_mask.to(device)
        caption_embeddings = self.bert(input_ids = input_ids, attention_mask = attention_mask).pooler_output # caption_embeddings.shape = (batch_size, 768)
        print(caption_embeddings.shape)


        # Concatenate the embeddings
        final_embeddings = torch.cat((clip_embeddings, average_patch_embeddings, caption_embeddings), dim=1) # final_embeddings.shape = (batch_size, 3072)
        print(final_embeddings.shape)

        final_embeddings = self.relu(final_embeddings)
        final_embeddings = self.linear1(final_embeddings)
        final_embeddings = self.linear2(final_embeddings)
        final_embeddings = self.linear3(final_embeddings)
        final_embeddings = self.linear4(final_embeddings)
        logits = final_embeddings
        probs = self.softmax(logits)
        return logits, probs


def get_tokens(self, texts, tokenization_strategy):
        if tokenization_strategy == "first":
            return self.model.tokenizer(texts, return_tensors="pt", padding = True, truncation=True)
        elif tokenization_strategy == "last":
            return self.model.tokenizerLast(texts, return_tensors="pt", padding = True, truncation=True)
        else:
            raise ValueError(f"tokenization_strategy {tokenization_strategy} not supported")



class Concatenated_Model():
    def __init__(self):
        self.model = Model()
        
    def eval(self, ds, batch_size = 8, tokenization_strategy="first" ):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval()
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


                #t_inputs = self.model.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                t_inputs = self.model.get_tokens(texts, tokenization_strategy)
                i_inputs = self.model.processor(images = images_list, return_tensors="pt", padding=True)
                
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)

                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)

                outputs = self.model(
                    images = images_list,
                    input_ids = t_inputs.input_ids,
                    attention_mask = t_inputs.attention_mask,
                    pixel_values = i_inputs.pixel_values,
                    nums_images = nums_images
                )

                logits = outputs[0]

                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                total_preds += list(preds)
                total_labels += list(labels)
                progress_bar.update(1)
        metrics = compute_metrics(total_labels, total_preds)
        return metrics
                

    
    def train(self, train_ds, eval_ds, batch_size = 8, num_epochs = 3, lr = 1e-5, warmup_steps = 0, num_eval_steps = 52, tokenization_strategy="first", save_path = "./"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn = collate_fn
        )

        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        # Initialize the optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps=len(dataloader) * num_epochs
        # Initialize the scheduler
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        current_step = 0
        # save the best model
        best_metric = 0
        for epoch in range(num_epochs):
            for batch in dataloader:
                texts = batch["text"]
                images_list = batch["images"]
                labels = batch["label"]
                nums_images = batch["nums_images"]


                #t_inputs = self.model.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                t_inputs = self.model.get_tokens(texts, tokenization_strategy)
                i_inputs = self.model.processor(images = images_list, return_tensors="pt", padding=True)
                
                for k, v in t_inputs.items():
                    t_inputs[k] = v.to(device)
                for k, v in i_inputs.items():
                    i_inputs[k] = v.to(device)
                
                nums_images = torch.tensor(nums_images).to(dtype=torch.long, device=device)

                outputs = self.model(
                    images = images_list,
                    input_ids = t_inputs.input_ids,
                    attention_mask = t_inputs.attention_mask,
                    pixel_values = i_inputs.pixel_values,
                    nums_images = nums_images
                )
                
                
                logits = outputs[0]
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                loss = criterion(logits, labels)
                metrics = compute_metrics(labels, preds)
                print("Train metrics: ", metrics)
                
                """
                if (current_step % num_eval_steps == 0):
                    print("Epoch: ", epoch)
                    print("Loss: ", loss.item())
                    eval_metrics = self.eval(eval_ds, tokenization_strategy, batch_size=batch_size)
                    print("Eval metrics: ", eval_metrics)
                    f1_score = eval_metrics["f1"]
                    if f1_score > best_metric:
                        print("New best model found")
                        best_metric = f1_score
                        torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth"))
                    print("Best metric: ", best_metric)
                    self.model.train()
                """

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                current_step += 1

        return best_metric