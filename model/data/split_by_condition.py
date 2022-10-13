from pathlib import Path
import os
import json
import shutil

BAD_LIGHTING_DATASET_PATH = Path("surveillance/bad_lighting")
GOOD_LIGHTING_DATASET_PATH = Path("surveillance/good_lighting")
ROOT_DATASET_PATH = Path("surveillance")
TRAIN_DATASET_PATH = ROOT_DATASET_PATH / "train_2021"
TEST_DATASET_PATH = ROOT_DATASET_PATH / "test"

OUTDOORS_DATASET_PATH = Path("surveillance/outdoors")
INDOORS_DATASET_PATH = Path("surveillance/indoors")

def split_by_lighting_cond():
    bad_lighting_img_ranges = [(263, 271), (1117,1191), 
                                (1535,1539), (1542,1543),
                                (3032,3064), (3184, 3203),
                                (3239, 3244), (4058, 4070),
                                (4703,4745), (6024, 6059), 
                                (6060, 6083)]

    bad_lighting = []
    bad_lighting_anno = []
    good_lighting_anno = []

    os.makedirs(BAD_LIGHTING_DATASET_PATH, exist_ok=True)
    os.makedirs(GOOD_LIGHTING_DATASET_PATH, exist_ok=True)

    # populate bad_lighting
    for start, end in bad_lighting_img_ranges:
        bad_lighting += [str(num)+".jpg" for num in range(start, end+1)]
    print(bad_lighting)
    
    
    # counter= 0
    # for img_anno in test_annos:
    #     if img_anno['file_name'] in bad_lighting:
    #         counter += 1
    
    # print(counter)
    
    
    with open(ROOT_DATASET_PATH / "train_2021.json", "r") as rf:
        all_annos=json.load(rf)
    
    with open(ROOT_DATASET_PATH / "test_anno.json", "r") as rft:
        test_annos = json.load(rft)
        
    for img_anno in all_annos:
        filename=img_anno['file_name']
        if filename in bad_lighting:
            shutil.copy(TRAIN_DATASET_PATH / filename, BAD_LIGHTING_DATASET_PATH / filename)
            bad_lighting_anno.append(img_anno)
            
    
    for test_img_anno in test_annos:
        filename=test_img_anno['file_name']
        if filename not in bad_lighting:
            shutil.copy(TEST_DATASET_PATH / filename, GOOD_LIGHTING_DATASET_PATH / filename)
            good_lighting_anno.append(test_img_anno)
            # good_lighting.append(filename)
        # else:
    
    with open(ROOT_DATASET_PATH / "good_lighting_anno.json", "w") as wf:
        json.dump(good_lighting_anno, wf)
    
    with open(ROOT_DATASET_PATH / "bad_lighting_anno.json", "w") as wf:
        json.dump(bad_lighting_anno, wf)
        
def split_by_env():
    outdoors_img_ranges = [(700, 716), (2833, 2950),
                               (2967, 3018), (4228, 4387),
                               (4658, 4686), (5965, 5991),
                               (6170, 6212), (6314, 6320),
                               (6024, 6059), (6369, 6384),
                               (6436, 6772)]

    outdoors = []
    outdoors_anno = []
    indoors_anno = []

    os.makedirs(OUTDOORS_DATASET_PATH, exist_ok=True)
    os.makedirs(INDOORS_DATASET_PATH, exist_ok=True)

    # populate outdoors
    for start, end in outdoors_img_ranges:
        outdoors += [str(num)+".jpg" for num in range(start, end+1)]
    print(outdoors)
    
    
    with open(ROOT_DATASET_PATH / "train_2021.json", "r") as rf:
        all_annos=json.load(rf)
    
    with open(ROOT_DATASET_PATH / "test_anno.json", "r") as rft:
        test_annos = json.load(rft)
        
    for img_anno in all_annos:
        filename=img_anno['file_name']
        if filename in outdoors:
            shutil.copy(TRAIN_DATASET_PATH / filename, OUTDOORS_DATASET_PATH / filename)
            outdoors_anno.append(img_anno)
            
    
    for test_img_anno in test_annos:
        filename=test_img_anno['file_name']
        if filename not in outdoors:
            shutil.copy(TEST_DATASET_PATH / filename, INDOORS_DATASET_PATH / filename)
            indoors_anno.append(test_img_anno)
            # indoors.append(filename)
        # else:
    
    with open(ROOT_DATASET_PATH / "indoors_anno.json", "w") as wf:
        json.dump(indoors_anno, wf)
    
    with open(ROOT_DATASET_PATH / "outdoors_anno.json", "w") as wf:
        json.dump(outdoors_anno, wf)

if __name__ == "__main__":
    split_by_lighting_cond()
    # split_by_env()
    
    
    
