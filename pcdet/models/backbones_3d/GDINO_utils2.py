import torch
from pcdet.utils.box2d_utils import pairwise_iou
from groundingdino.util.inference import predict

def score_down_logit(phrases, logits, scores_down):
    logits_new = logits.clone()
    
    for i, phrase in enumerate(phrases):
        if phrase in scores_down:
            logits_new[i] = logits[i] * scores_down[phrase]  
    
    return phrases, logits_new


def remove_overlap_boxes(boxes, logits, phrases, boxes_2=None, logits_2=None, phrases_2=None, iou_thresh=0.7):
    if boxes_2 is None:
        use_same_data = True
    else:
        use_same_data = False

    if use_same_data:
        if len(boxes) == 0:
            return boxes, logits, phrases
        
        try:
            import torchvision
            keep_indices = torchvision.ops.nms(boxes, logits, iou_thresh)
            
            boxes = boxes[keep_indices]
            logits = logits[keep_indices]
            phrases = [phrases[i] for i in keep_indices.tolist()]
            
            return boxes, logits, phrases
        except Exception as e:
            print(f"torchvision NMS failed: {e}, using fallback")
        
        ious = pairwise_iou(boxes, boxes)
        ious = ious * torch.tril(torch.ones(ious.shape), diagonal=-1).to(device=ious.device)

        overleap_pairs = torch.nonzero(ious > iou_thresh)
        if overleap_pairs.shape[0] != 0:
            pairs_0 = overleap_pairs[:, 0]
            pairs_1 = overleap_pairs[:, 1]
            
            del_idxs = torch.where(
                logits[pairs_0] > logits[pairs_1], 
                pairs_1, 
                pairs_0
            ).tolist()
            
            unique_del_idxs = list(set(del_idxs))
            mask = torch.ones(boxes.size(0), dtype=bool, device=boxes.device)
            mask[unique_del_idxs] = False

            boxes = boxes[mask]
            logits = logits[mask]
            phrases = [item for idx, item in enumerate(phrases) if mask[idx]]
        
        return boxes, logits, phrases
    
    else:
        ious = pairwise_iou(boxes, boxes_2)
        overleap_pairs = torch.nonzero(ious > iou_thresh)
        if overleap_pairs.shape[0] != 0:
            del_idxs = []
            del_idxs_2 = []
            for pair in overleap_pairs:
                if logits[pair[0]] < logits_2[pair[1]]:
                    del_idxs.append(pair[0].item())
                else:
                    del_idxs_2.append(pair[1].item())
            
            unique_del_idxs = list(set(del_idxs))
            mask = torch.ones(boxes.size(0), dtype=bool, device=boxes.device)
            mask[unique_del_idxs] = False

            boxes = boxes[mask]
            logits = logits[mask]
            phrases = [item for idx, item in enumerate(phrases) if idx not in unique_del_idxs]

            unique_del_idxs_2 = list(set(del_idxs_2))
            mask_2 = torch.ones(boxes_2.size(0), dtype=bool, device=boxes_2.device)
            mask_2[unique_del_idxs_2] = False

            boxes_2 = boxes_2[mask_2]
            logits_2 = logits_2[mask_2]
            phrases_2 = [item for idx, item in enumerate(phrases_2) if idx not in unique_del_idxs_2]

        return boxes, logits, phrases, boxes_2, logits_2, phrases_2


def map_class_name_to_id(boxes_dino_trans, logits_dino, phrases_dino, det_map_dict):
    boxes, labels, score_rois = [], [], []
    
    for i_box, phrase in enumerate(phrases_dino):
        if phrase in det_map_dict:
            boxes.append(boxes_dino_trans[i_box])
            labels.append(det_map_dict[phrase])
            score_rois.append(logits_dino[i_box])
        else:
            for txt_cls in det_map_dict:
                if txt_cls in phrase:
                    boxes.append(boxes_dino_trans[i_box].clone())
                    labels.append(det_map_dict[txt_cls])
                    score_rois.append(logits_dino[i_box].clone())

    
    return boxes, labels, score_rois


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


def rename_str_in_phrases(phrases, replace_labels):
    new_phrases = []
    for phrase in phrases:
        new_phrase = phrase
        for key, value in replace_labels.items():
            new_phrase = new_phrase.replace(key, value)
        new_phrases.append(new_phrase)
    return new_phrases