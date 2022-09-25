import json
import os
import shutil
import numpy as np

def split_train_test():
    with open("surveillance/train_2021.json", 'r') as f:
        annos = json.load(f)

    print(f"total: {len(annos)}")
    # guns = []
    # knives = []
    # machetes = []

    filenames = {
        'guns': [],
        'knives': [],
        'machetes': []
    }

    for img_anno in annos:
        filename = img_anno['file_name']
        for obj_anno in img_anno['annotations']:
            if (obj_anno['category_id'] == 2) and filename not in filenames['guns']:
                filenames['guns'].append(filename)
            elif (obj_anno['category_id'] == 3) and filename not in filenames['knives']:
                filenames['knives'].append(filename)
            elif (obj_anno['category_id'] == 4) and filename not in filenames['machetes']:
                filenames['machetes'].append(filename)

    print(f"num guns: {len(filenames['guns'])}, num knives: {len(filenames['knives'])}, num machetes: {len(filenames['machetes'])}")

    train = []

    test = []

    num_train = {
        'guns': int(0.8*len(filenames['guns'])),
        'knives': int(0.8 * len(filenames['knives'])),
        'machetes': int(0.8 * len(filenames['machetes']))
    }


    for category in num_train:
        num = num_train[category]
        for idx in range(num):
            # print(category)
            # print(idx)
            filename = filenames[category][idx]
            train.append(filenames[category][idx])
            for cat in filenames:
                try:
                    if cat != category:
                        filenames[cat].remove(filename)
                        num_train[cat] -= 1
                except ValueError:
                    continue
        for img in train:
            try:
                filenames[category].remove(img)
            except ValueError:
                continue
            
    for category in filenames:
        test += filenames[category]
        

    print("After:")
    print(f"num guns: {len(filenames['guns'])}, num knives: {len(filenames['knives'])}, num machetes: {len(filenames['machetes'])}" )     
    print(f"Length of train: {len(train)}")
    print(f"Length of test: {len(test)}")

    for img_filename in train:
        if img_filename in test:
            raise Exception("duplicate!")
        
    print(train[:5])
    print(test[:5])

    TRAIN_PATH = os.path.join("surveillance", "train")
    TEST_PATH = os.path.join("surveillance","test")
    PREV_PATH = os.path.join("surveillance","FinalData")
    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(TEST_PATH, exist_ok=True)

    train_annos = []
    test_annos = []

    for img_anno in annos:
        filename = img_anno['file_name']
        if filename in train:
            train_annos.append(img_anno)
        elif filename in test:
            test_annos.append(img_anno)
        else:
            raise ValueError(f"filename {filename} not in train or test sets")
        
    for img_filename in train:
        shutil.move(os.path.join(PREV_PATH, img_filename), os.path.join(TRAIN_PATH, img_filename))

    with open(os.path.join(TRAIN_PATH, "train_anno.json"), "w") as wf:
        json.dump(train_annos, wf)
        
    for img_filename in test:
        shutil.move(os.path.join(PREV_PATH, img_filename), os.path.join(TEST_PATH, img_filename))

    with open(os.path.join(TEST_PATH, "test_anno.json"), "w") as wf:
        json.dump(test_annos, wf)
        
def split_train():
    with open("surveillance/train_2021.json", 'r') as f:
        annos = json.load(f)

    print(f"total: {len(annos)}")
    # guns = []
    # knives = []
    # machetes = []

    filenames = {
        'guns': [],
        'knives': [],
        'machetes': []
    }

    for img_anno in annos:
        filename = img_anno['file_name']
        for obj_anno in img_anno['annotations']:
            if (obj_anno['category_id'] == 2) and filename not in filenames['guns']:
                filenames['guns'].append(filename)
            elif (obj_anno['category_id'] == 3) and filename not in filenames['knives']:
                filenames['knives'].append(filename)
            elif (obj_anno['category_id'] == 4) and filename not in filenames['machetes']:
                filenames['machetes'].append(filename)

    print(f"num guns: {len(filenames['guns'])}, num knives: {len(filenames['knives'])}, num machetes: {len(filenames['machetes'])}")

    # rng=np.random.default_rng(seed=42)
    np.random.seed(42)
    train_sub = {
        'guns': [],
        'knives': [],
        'machetes': []
    }
    for cat in filenames:
        train_sub[cat] = np.random.choice(filenames[cat], round(0.2 * len(filenames[cat])), replace=False)
    # print(len(np.random.choice(filenames['guns'], round(0.2 * len(filenames['guns'])), replace=False)))
    print("Train subset:")
    print(f"num guns: {len(train_sub['guns'])}, num knives: {len(train_sub['knives'])}, num machetes: {len(train_sub['machetes'])}")
    
    PREV_PATH = os.path.join("surveillance", "FinalData")
    TEST_PATH = os.path.join("surveillance", "train_test")
    os.makedirs(TEST_PATH, exist_ok=True)
    
    TRAIN_PATH = os.path.join("surveillance", "train_2021")
    os.makedirs(TRAIN_PATH, exist_ok=True)
    
    for cat in train_sub:
        for img_filename in train_sub[cat]:
            shutil.copy(os.path.join(PREV_PATH, img_filename), os.path.join(TEST_PATH, img_filename))
    
    train_sub_anno = []
    for img_anno in annos:
        filename =  img_anno['file_name']
        shutil.copy(os.path.join(PREV_PATH, filename), os.path.join(TRAIN_PATH, filename))
        for cat in train_sub:
            if filename in train_sub[cat] and img_anno not in train_sub_anno:
                train_sub_anno.append(img_anno)
                break
            
    print("Length of train sub annos:", len(train_sub_anno))
    print(train_sub_anno[:5])
    
    with open(os.path.join("surveillance", "train_test_anno.json"), "w") as wf:
        json.dump(train_sub_anno, wf)
    
    
    
# print(len(annos))

if __name__ == '__main__':
    split_train()

