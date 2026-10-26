## FedAvg with T local steps vs pooled (50 pooled-equivalent epochs, lr 0.003, seeds 0, 1, 2, 3, 4)

Final test scores on all 254 test patients, mean ± std over seeds. Every row takes the same 6100 SGD steps; FedAvg spreads them over rounds × T.

| run | rounds | accuracy | balanced accuracy | AUC |
|---|---|---|---|---|
| pooled | - | 0.798 ± 0.005 | 0.798 ± 0.005 | 0.881 ± 0.001 |
| FedAvg T=1 | 6100 | 0.783 ± 0.004 | 0.783 ± 0.004 | 0.864 ± 0.001 |
| FedAvg T=2 | 3050 | 0.783 ± 0.004 | 0.783 ± 0.004 | 0.864 ± 0.001 |
| FedAvg T=5 | 1220 | 0.783 ± 0.003 | 0.783 ± 0.003 | 0.864 ± 0.001 |
| FedAvg T=10 | 610 | 0.783 ± 0.003 | 0.783 ± 0.003 | 0.864 ± 0.001 |
| FedAvg T=25 | 244 | 0.783 ± 0.003 | 0.783 ± 0.003 | 0.864 ± 0.001 |
| FedAvg T=50 | 122 | 0.783 ± 0.003 | 0.783 ± 0.003 | 0.864 ± 0.001 |
| FedAvg T=100 | 61 | 0.783 ± 0.002 | 0.783 ± 0.002 | 0.863 ± 0.001 |
| FedAvg T=250 | 25 | 0.781 ± 0.004 | 0.781 ± 0.003 | 0.863 ± 0.001 |
