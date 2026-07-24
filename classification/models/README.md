# models

Store trained model artifacts or pointers here. Bulk weights are gitignored; document download / training steps in the parent `classification/README.md`.

## PatentBERT (`patentbert/`)

Download once (~1.3GB checkpoint + labels + BERT base vocab/config):

```bash
python scripts/download_patentbert.py
# → classification/models/patentbert/
```

Expected files:

- `model.ckpt-181172.data-00000-of-00001`
- `model.ckpt-181172.index`
- `model.ckpt-181172.meta`
- `labels_group_id.tsv`
- `vocab.txt`
- `bert_config.json`

Optional: `python scripts/download_patentbert.py --with-data-2015` also fetches `data.2015.tsv`.
