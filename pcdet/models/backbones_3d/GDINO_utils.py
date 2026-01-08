import torch
from pcdet.utils.box2d_utils import pairwise_iou
from groundingdino.util.inference import predict

def score_threshold(boxes, logits, phrases, score_thresh_dict):
    
    mask = torch.zeros_like(logits, dtype=bool)
    for j, phrase in enumerate(phrases):
        for cls, thresh in score_thresh_dict.items():
            if cls in phrase and logits[j] > thresh:
                mask[j] = True
                break
    boxes = boxes[mask]
    logits = logits[mask]
    phrases = [item for idx, item in enumerate(phrases) if mask[idx]]
    return boxes, logits, phrases

def gdino_processing(model, image, txt_prompt, box_thresh, txt_thresh, valid_labels, denial_labels=None):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=txt_prompt,
        box_threshold=box_thresh,
        text_threshold=txt_thresh
    )
    valid_flag = torch.zeros(boxes.shape[0], dtype=bool)
    for i in range(boxes.shape[0]):
        phrases[i] = phrases[i].replace('cargo', 'cg')
        for valid_label in valid_labels:            
            if valid_label in phrases[i]:
                if denial_labels is not None:
                    if valid_label in denial_labels:
                        if denial_labels[valid_label] not in phrases[i]:
                            valid_flag[i] = True             
                    else:
                        valid_flag[i] = True
                else:
                    valid_flag[i] = True

    boxes, logits, phrases = boxes[valid_flag], logits[valid_flag], [item for idx, item in enumerate(phrases) if valid_flag[idx]]

    return boxes, logits, phrases

def remove_overlap_boxes(boxes, logits, phrases, boxes_2=None, logits_2=None, phrases_2=None, iou_thresh=0.7):
    if boxes_2 == None:
        use_same_data = True
    else:
        use_same_data = False

    if use_same_data:
        ious = pairwise_iou(boxes, boxes)
        ious = ious * torch.tril(torch.ones(ious.shape), diagonal=-1).to(device=ious.device)

        overleap_pairs = torch.nonzero(ious > iou_thresh)
        if overleap_pairs.shape[0] != 0:
            del_idxs=[]
            for pair in overleap_pairs:
                del_idxs += [pair[1].item()] if logits[pair[0]] > logits[pair[1]] else [pair[0].item()]
            
            unique_del_idxs = list(set(del_idxs))
            mask = torch.ones(boxes.size(0), dtype=bool)
            mask[unique_del_idxs] = False

            boxes = boxes[mask]
            logits = logits[mask] 
            phrases = [item for idx, item in enumerate(phrases) if idx not in unique_del_idxs]
        
        return boxes, logits, phrases
    
    else:
        ious = pairwise_iou(boxes, boxes_2)
        overleap_pairs = torch.nonzero(ious > iou_thresh)
        if overleap_pairs.shape[0] != 0:
            del_idxs=[]
            del_idxs_2=[]
            for pair in overleap_pairs:
                if logits[pair[0]] < logits_2[pair[1]]:
                    del_idxs.append(pair[0].item())
                else:
                    del_idxs_2.append(pair[1].item())
            
            unique_del_idxs = list(set(del_idxs))
            mask = torch.ones(boxes.size(0), dtype=bool)
            mask[unique_del_idxs] = False

            boxes = boxes[mask] 
            logits = logits[mask] 
            phrases = [item for idx, item in enumerate(phrases) if idx not in unique_del_idxs]

            unique_del_idxs_2 = list(set(del_idxs_2))
            mask_2 = torch.ones(boxes_2.size(0), dtype=bool)
            mask_2[unique_del_idxs_2] = False

            boxes_2 = boxes_2[mask_2] 
            logits_2 = logits_2[mask_2] 
            phrases_2 = [item for idx, item in enumerate(phrases_2) if idx not in unique_del_idxs_2]

        return boxes, logits, phrases, boxes_2, logits_2, phrases_2


def score_down_logit(phrases, logits, scores_down):
    for i in range(logits.shape[0]):
        score_de = 0
        for obj in scores_down:
            if obj in phrases[i]:
                score_de = scores_down[obj] if score_de < scores_down[obj] else score_de
        logits[i] -= score_de
    return phrases, logits

def rename_str_in_phrases(phrases, replace_labels):
    for i_obj, phrase in enumerate(phrases):
        for key in replace_labels:
            if key in phrase:
                phrases[i_obj] = phrases[i_obj].replace(key, replace_labels[key])
    return phrases

def map_class_name_to_id(boxes_dino_trans, logits_dino, phrases_dino, det_map_dict):
    boxes, labels, score_rois = [], [], []
    for i_box, box_ in enumerate(boxes_dino_trans):
        for txt_cls in det_map_dict:
            if txt_cls in phrases_dino[i_box]:
                boxes.append(box_)
                labels.append(det_map_dict[txt_cls])
                score_rois.append(logits_dino[i_box])
    
    return boxes, labels, score_rois