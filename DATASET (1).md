# Dataset: RAG-Phase1

Images and captions from **MS COCO**, cleaned and prepared for our RAG image-captioning project.
Images, vectors and indexes are on Google Drive (`RAG-Phase1/`), not on GitHub (too big).
Code: `notebooks/Phase1_v5.ipynb`. Fixed seeds (42, 43), so everything can be reproduced.

## The sets

- **Knowledge base:** 10,000 images (50,000 captions) from train2017. The system searches here.
- **Validation:** 100 images from train2017, different from the knowledge base. Used to choose settings.
- **Test A:** 100 images from val2017.
- **Test B:** 1,000 images from val2017. It contains Test A. Main test set.

Every image has 5 captions. The test sets are balanced between simple, medium and complex scenes.

## What is on the Drive

- `images/`: the pictures (`train`, `validation`, `test`). File name = COCO id with 12 digits.
- `splits.json`: the image ids of each set. Always use this file, not the folder content.
- `captions.json`: the 5 original captions of each image.
- `metadata/kb_records.json`: for each knowledge-base image: captions, detected objects (label, score, box), human objects, scene type, colours / context words.
- `metadata/query_objects.json`: objects detected in validation and test images. Can be used for retrieval.
- `metadata/eval_objects.json`: human objects of validation and test images. Only for evaluation.
- `embeddings/`: CLIP vectors (512 numbers each).
- `index/`: FAISS indexes of the knowledge-base images and captions.
- `plots/` and other files in `metadata/`: figures, statistics and audit files.

Row k of a vector file is item k of its id list. Never sort or shuffle one without the other.

## How it was made

- Images: rotation fixed, converted to RGB, resized for CLIP.
- Captions: the original is kept for evaluation. A lowercase copy without punctuation is used for search.
- Objects: found in the images with DETR (`facebook/detr-resnet-50`). We keep objects with score 0.9 or more (chosen on the validation images). Human data counts as confidence 1.
- Vectors: CLIP `clip-vit-base-patch32`, length 1. Index: FAISS flat, cosine similarity.

## Rules

1. Search only in the knowledge base.
2. Never use validation or test captions for search or generation. They are for scoring only.
3. A test image must never be retrieved for itself.
4. Use `query_objects.json` for queries, never `eval_objects.json`.
5. Do not change the splits.

## Checks (all passed)

- 0 missing, 0 corrupted, 0 exact duplicate images.
- 0 images shared between any two sets.
- Vectors have the right shape and no NaN.

## Main numbers (knowledge base)

- Objects per image: 7.19 (human labels), 6.15 (detector).
- Vocabulary: 16,255 words before cleaning, 10,080 after.
- Average caption length: 10.45 words.
- Detector against human labels (F1): 0.920 on the knowledge base, 0.850 on Test B.

## Load it

```python
import json, numpy as np, faiss
BASE = "/content/drive/MyDrive/RAG-Phase1"   # change to your path

splits = json.load(open(f"{BASE}/splits.json"))
rows = json.load(open(f"{BASE}/metadata/kb_caption_rows.json"))
index = faiss.read_index(f"{BASE}/index/kb_captions.faiss")
test_vectors = np.load(f"{BASE}/embeddings/test1000_image_vectors.npy")

scores, found = index.search(test_vectors[:1], 5)   # 5 closest captions
for k in found[0]:
    print(rows[k]["caption_raw"])
```

## Limitations

- The detector knows only 80 classes and misses some objects (recall about 0.80 on test).
- Colours and context come from caption words, not from the pixels.
- Duplicates are found by exact file fingerprint, so near-identical pictures could be missed.
- The fingerprint file of the first run was lost, so the audit describes the final sets.
- The detector threshold was chosen on only 100 images.

Source: MS COCO (Lin et al., ECCV 2014). Follow the COCO terms of use.
