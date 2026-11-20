# Results

Copy these files here from the Google Drive folder `liexnet_v3/` written by the notebook:

| File | Produced by | Used in paper |
|---|---|---|
| `runs.csv` | cell 6 (one row per variant x seed) | Table 2, Table 4 |
| `master_table.csv` | cell 11 | Tables 2, 6 |
| `per_class_f1.csv` | cell 9 | Table 5 |
| `paired_test_acc.csv`, `paired_macro_f1.csv` | cell 8 / 11 | Table 4 |
| `confusion_matrices.png` | cell 9 | Fig. 3 |
| `config.json` | cell 11 | hyper-parameters, teacher accuracy |
| `probs/*.npy` | cell 6 (flip-TTA test probabilities, float16) | Table 5, Fig. 3 |

The teacher checkpoint (`teacher_resnet18.pt`, ~43 MB) is not stored in git; it is attached to the GitHub Release.
