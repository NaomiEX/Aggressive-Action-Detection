import numpy as np
import os


class SurveillanceEvaluator():
    def __init__(self, preds, gts, args):
        # preds is an array of dictionaries. Each dictionary contains:
        #   "labels" : [200], first 100 is just 0 (subject category id), next 100 is the top 100 objects
        #   "boxes" : [200, 4], first 100 is the top bbox predictions for the corresponding subjects, 
        #                       next 100 is the top bbox predictions for the corresponding objects.
        #   "verb_scores" : [100, 4], get the corresponding top 100 verbs, 
        #                   4-dimensional because it contains the sigmoid-ed score for each verb category 
        #                   (there are 4 possible verbs)
        #   "sub_ids": [100], indexes the subjects in labels, boxes, it is just a tensor containing [0, 1, 2, 3...,99]
        #   "obj_ids": [100], indexes the objects in labels, boxes, [100, 101, 102, 103, ..., 199]
        
        # gts is an array of dictionaries. Each dictionary represents the gt annotation for an image
        #   "orig_size": [2]
        #   "size": [2]
        #   "boxes": [N_o, 4] (unnormalized)
        #   "labels": [N_o] object labels (0-indexed)
        #   "id": int
        #   "file_name": string
        #   "hois": [N_hoi, 3], each consists of triplets: (subject_id, object_id, category_id) where subject and object id
        #           refer to the index w.r.t "boxes", category_id is 0-indexed
        
        
        
        # self.correct_mat = np.load(os.path.join(args.hoi_path, "annotations", "corre_hoia.npy"))
        # for row i, the columns which are 1 represent the possible objects that can have that interaction
        self.correct_mat = np.array([[0,1,0,0], [0,0,1,1], [0,1,1,1], [0,1,1,1]])
        self.overlap_iou = 0.5
        self.max_hois = 100
        # self.verb_name_dict = {1: 'smoke', 2: 'call', 3: 'play(cellphone)', 4: 'eat', 5: 'drink',
        #                        6: 'ride', 7: 'hold', 8: 'kick', 9: 'read', 10: 'play (computer)'}
        # self.verb_name_dict = {0: 'smoke', 1: 'call', 2: 'play(cellphone)', 3: 'eat', 4: 'drink',
        #                        5: 'ride', 6: 'hold', 7: 'kick', 8: 'read', 9: 'play (computer)'}
        self.verb_name_dict = {0: 'point', 1: 'swing', 2: 'hold', 3: 'no_interaction'}
        self.fp = {}
        self.tp = {}
        self.score = {}
        self.sum_gt = {} # keeps a total number of instances for each HOI category
        
        # keep track of false positives and true positives, score, and the number of instances for each verb
        for i in list(self.verb_name_dict.keys()):
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
            self.sum_gt[i] = 0
            
        self.file_name = []
        self.nms_thresh = args.nms_thresh
        ####################################################
        
        # stores the gts as dictionaries, each dictionary represents the gt for a single image.
        # Dictionaries are of the form:
        #   {
                # "annotations": list of size N_o, of the form: [
        #             {
        #                 "bbox": [4],
        #                 "category_id": int
        #             },
        #             ... 
        #         ],
        #         "hoi_annotation": list of size N_hoi, of the form: [
        #             {
        #                 "subject_id": int,
        #                 "object_id": int,
        #                 "category_id": int
        #             }
        #         ]
        #   }
        self.gts = []
        for img_gts in gts: # img_gts is the ground truth dictionary for an image
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k not in ['id', 'file_name']}
            # print(img_gts['labels'])
            self.gts.append({
                # pair up the gt boxes and object labels
                'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in
                                zip(img_gts['boxes'], img_gts['labels'])],
                # split up the triplet into a dictionary
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in
                                   img_gts['hois']]
            })

        for gt_i in self.gts:
            gt_hoi = gt_i['hoi_annotation']
            for gt_hoi_i in gt_hoi:
                if isinstance(gt_hoi_i['category_id'], str):
                    gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n', ''))
                assert gt_hoi_i['category_id'] in list(self.verb_name_dict.keys())
                self.sum_gt[gt_hoi_i['category_id']] += 1 # keep track of number of instances for that category
                
        self.num_class = len(list(self.verb_name_dict.keys()))

        ####################################################
        # list of length T, where T is the size of the test set,
        # each element consists of a dictionary with:
        #   'predictions' : list of size 200 (the first 100 correspond to the subject, the latter correspond to the object) 
        #                   where each element is a dict with the key-value pairs:
        #       'bbox' : numpy array of size (4,)
        #       'category_id' : category label of the object
        #   'hoi_prediction' : list of size 100 where each element is a dict with the keys:
        #       'subject_id', 'object_id', 'category_id', 'score'
        self.preds = [] 
        for img_preds in preds:
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            # img_preds['labels'][:100] -= 1
            # print(img_preds['labels'])
            
            # pair up the object bboxes and labels
            bboxes = [{'bbox': list(bbox), 'category_id': label} for bbox, label in
                      zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            
            # tensor of shape: [100, 4] of the form: [[0, 1, 2, 3],
            #                                         [0, 1, 2, 3],
            #                                         ...]
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            # tensor of shape: [100, 4] of the form:
            # [[0, 0, 0, 0],
            #  [1, 1, 1, 1],
            #  [2, 2, 2, 2],
            #  ...]
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            # tensor of shape: [100, 4] of the form:
            # [[100, 100, 100, 100],
            #  [101, 101, 101, 101],
            #  [102, 102, 102, 102],
            #  ...]
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T
            
            ## flattens the tensors
            hoi_scores = hoi_scores.ravel() # [400]
            verb_labels = verb_labels.ravel() # [400]
            subject_ids = subject_ids.ravel() # [400]
            object_ids = object_ids.ravel() # [400]
            # Note: now we have the subject-object-verb-verb score pair for every possible verb for each of the top 100 chosen predictions

            if len(subject_ids) > 0:
                # get the object labels
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids]) # [400]
                # 0 if the object-interaction does not exist in the dataset, 1 if it exists
                masks = self.correct_mat[verb_labels, object_labels] # [400]
                # if the object-interaction pair cannot possibly exist, 0 out the hoi score
                hoi_scores *= masks 

                # an array of size 400
                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score}
                        for
                        subject_id, object_id, category_id, score in
                        zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                
                # sort by descending order of hoi score
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois] # get the top 100 hois
            else:
                hois = []

            self.preds.append({
                'predictions': bboxes, # [200], elements are dictionaries, object bboxes-label predictions
                'hoi_prediction': hois # [100], elements are dictionaries
            })
        if args.use_nms:
            self.preds = self.triplet_nms_filter(self.preds)

    def evaluate(self):
        for pred_i, gt_i in zip(self.preds, self.gts):
            gt_bbox = gt_i['annotations']
            pred_bbox = pred_i['predictions']
            pred_hoi = pred_i['hoi_prediction']
            gt_hoi = gt_i['hoi_annotation']
            bbox_pairs = self.compute_iou_mat(gt_bbox, pred_bbox)
            self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs)
        map = self.compute_map()
        print(map)
        return map

    def compute_map(self):
        ap = np.zeros(self.num_class - 1)
        max_recall = np.zeros(self.num_class -1)
        for i in list(self.verb_name_dict.keys()):
            if i == 3: # no need to calculate mAP for no interaction
                continue
            sum_gt = self.sum_gt[i]

            if sum_gt == 0:
                continue
            tp = np.asarray((self.tp[i]).copy())
            fp = np.asarray((self.fp[i]).copy())
            res_num = len(tp)
            if res_num == 0:
                continue
            score = np.asarray(self.score[i].copy())
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gt
            prec = tp / (fp + tp)
            ap[i] = self.voc_ap(rec, prec)
            max_recall[i] = np.max(rec)
            # print('class {} --- ap: {}   max recall: {}'.format(
            #     i, ap[i-1], max_recall[i-1]))
        mAP = np.mean(ap[:])
        m_rec = np.mean(max_recall[:])
        print('--------------------')
        print('mAP: {}   max recall: {}'.format(mAP, m_rec))
        print('--------------------')
        cate_map = {}
        for i in range(len(ap)):
            cate_map[self.verb_name_dict[i]] = ap[i]
        cate_map.update({'mAP': mAP,
                               'mean max recall': m_rec})
        return cate_map

    def voc_ap(self, rec, prec):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
        # ap = 0.
        # for t in np.arange(0., 1.1, 0.1):
        #     if np.sum(rec >= t) == 0:
        #         p = 0
        #     else:
        #         p = np.max(prec[rec >= t])
        #     ap = ap + p / 11.
        # return ap

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i[
                    'object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi_i['object_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    for gt_id in np.nonzero(1 - vis_tag)[0]:
                        gt_hoi_i = gt_hoi[gt_id]
                        if vis_tag[gt_id] ==0 and (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (
                                pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            vis_tag[gt_id] = 1
                            continue
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                if is_match == 1:
                    self.fp[pred_hoi_i['category_id']].append(0)
                    self.tp[pred_hoi_i['category_id']].append(1)

                else:
                    self.fp[pred_hoi_i['category_id']].append(1)
                    self.tp[pred_hoi_i['category_id']].append(0)
                self.score[pred_hoi_i['category_id']].append(pred_hoi_i['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                if i == 0 and j == 0:
                    pass
                if i == 1 and j ==100:
                    pass
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i
        iou_mat[iou_mat >= self.overlap_iou] = 1
        iou_mat[iou_mat < self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
        return match_pairs_dict

    def compute_IOU(self, bbox1, bbox2):
        
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line) * (bottom_line - top_line)
                return intersect / (sum_area - intersect)
        else:
            return 0
        
    # Refer to CDN: https://github.com/YueLiao/CDN/blob/main/datasets/hico_eval.py
    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in preds:
            pred_bboxes = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            all_triplets = {}
            for index, pred_hoi in enumerate(pred_hois):
                triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                          str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

                if triplet not in all_triplets:
                    all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
                all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
                all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
                all_triplets[triplet]['scores'].append(pred_hoi['score'])
                all_triplets[triplet]['indexes'].append(index)

            all_keep_inds = []
            for triplet, values in all_triplets.items():
                subs, objs, scores = values['subs'], values['objs'], values['scores']
                keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

                keep_inds = list(np.array(values['indexes'])[keep_inds])
                all_keep_inds.extend(keep_inds)

            preds_filtered.append({
                # 'filename': img_preds['filename'],
                'predictions': pred_bboxes,
                'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
                })

        return preds_filtered
    
    # Modified from CDN: https://github.com/YueLiao/CDN/blob/main/datasets/hico_eval.py
    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = sub_inter/sub_union * obj_inter/obj_union
            inds = np.where(ovr <= self.nms_thresh)[0]

            order = order[inds + 1]
        return keep_inds
    
# if __name__ == "__main__":
    # import pickle
    # gts = None
    # preds = None
    # with open("gts", 'rb') as rf:
    #     gts = pickle.load(rf)
        
    # with open('preds', "rb") as rf2:
    #     preds = pickle.load(rf2)
    # # with open("gts_one", 'rb') as rf:
    # #     gts = pickle.load(rf)
    # #     gts = [gts] # FIXME: only for gts_one
        
    # # with open('preds_one', "rb") as rf2:
    # #     preds = pickle.load(rf2)
    # #     preds = [preds] # FIXME: only for preds_one
    
    # # gts_small = gts[:5]
    # # gts_med = gts[:10]
    # # preds_small = preds[:5]
    # # preds_med = preds[:10]
    # # gts_one = gts[0]
    # # preds_one = preds[0]
    
    # # with open("gts_one", "wb") as wf1:
    # #     pickle.dump(gts_one, wf1)
        
    # # with open("preds_one", "wb") as wf1_2:
    # #     pickle.dump(preds_one, wf1_2)
        
    # # with open("gts_small", "wb") as wf2:
    # #     pickle.dump(gts_small, wf2)
        
    # # with open("preds_small", "wb") as wf2_2:
    # #     pickle.dump(preds_small, wf2_2)
        
    # # with open("gts_med", "wb") as wf3:
    # #     pickle.dump(gts_med, wf3)
        
    # # with open("preds_med", "wb") as wf3_2:
    # #     pickle.dump(preds_med, wf3_2)
    
    # class Args:
    #     def __init__(self):
    #         self.use_nms = True
    #         self.nms_thresh = 0.5
    #         self.hoi_path = os.path.join("data", "hoia")
    #         self.dataset_file = "hoia"
    # evaluator = HOIAEvaluator(preds, gts, Args())
    # evaluator.evaluate()