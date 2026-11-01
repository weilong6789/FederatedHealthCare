## FedAvg with T local steps vs pooled (50 pooled-equivalent epochs, lr 0.003, seeds 0, 1, 2, 3, 4)

Final test scores on all 254 test patients, mean ± std over seeds. Every row takes the same 6100 SGD steps; FedAvg spreads them over rounds × T.

| run | rounds | accuracy | balanced accuracy | AUC |
|---|---|---|---|---|
| pooled | - | 0.735 ± 0.007 | 0.734 ± 0.007 | 0.805 ± 0.000 |
| FedAvg T=1 | 6100 | 0.735 ± 0.005 | 0.735 ± 0.005 | 0.805 ± 0.001 |
| FedAvg T=2 | 3050 | 0.735 ± 0.005 | 0.735 ± 0.005 | 0.805 ± 0.001 |
| FedAvg T=5 | 1220 | 0.735 ± 0.005 | 0.735 ± 0.005 | 0.805 ± 0.001 |
| FedAvg T=10 | 610 | 0.735 ± 0.005 | 0.735 ± 0.005 | 0.805 ± 0.001 |
| FedAvg T=25 | 244 | 0.735 ± 0.005 | 0.735 ± 0.005 | 0.805 ± 0.001 |
| FedAvg T=50 | 122 | 0.736 ± 0.005 | 0.736 ± 0.005 | 0.805 ± 0.000 |
| FedAvg T=100 | 61 | 0.735 ± 0.004 | 0.735 ± 0.004 | 0.805 ± 0.000 |
| FedAvg T=250 | 25 | 0.735 ± 0.004 | 0.735 ± 0.004 | 0.805 ± 0.001 |

Final test accuracy at each hospital, mean over seeds:

| hospital | pooled | T=1 | T=2 | T=5 | T=10 | T=25 | T=50 | T=100 | T=250 |
|---|---|---|---|---|---|---|---|---|---|
| Cleveland | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 |
| Hungary | 0.757 | 0.760 | 0.760 | 0.760 | 0.760 | 0.760 | 0.762 | 0.762 | 0.762 |
| Switzerland | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| Long Beach VA | 0.609 | 0.609 | 0.609 | 0.609 | 0.609 | 0.609 | 0.609 | 0.604 | 0.604 |

Final test balanced accuracy at each hospital, mean over seeds:

| hospital | pooled | T=1 | T=2 | T=5 | T=10 | T=25 | T=50 | T=100 | T=250 |
|---|---|---|---|---|---|---|---|---|---|
| Cleveland | 0.769 | 0.769 | 0.769 | 0.769 | 0.769 | 0.769 | 0.769 | 0.769 | 0.769 |
| Hungary | 0.786 | 0.788 | 0.788 | 0.788 | 0.788 | 0.788 | 0.790 | 0.790 | 0.790 |
| Switzerland | 0.867 | 0.867 | 0.867 | 0.867 | 0.867 | 0.867 | 0.867 | 0.867 | 0.867 |
| Long Beach VA | 0.677 | 0.677 | 0.677 | 0.677 | 0.677 | 0.677 | 0.677 | 0.674 | 0.674 |

Final test AUC at each hospital, mean over seeds:

| hospital | pooled | T=1 | T=2 | T=5 | T=10 | T=25 | T=50 | T=100 | T=250 |
|---|---|---|---|---|---|---|---|---|---|
| Cleveland | 0.853 | 0.854 | 0.854 | 0.854 | 0.854 | 0.854 | 0.854 | 0.854 | 0.855 |
| Hungary | 0.869 | 0.869 | 0.869 | 0.869 | 0.869 | 0.869 | 0.869 | 0.869 | 0.869 |
| Switzerland | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 |
| Long Beach VA | 0.743 | 0.743 | 0.743 | 0.743 | 0.743 | 0.743 | 0.743 | 0.743 | 0.743 |

Switzerland has 16 test patients, one of them label 0, so its accuracy moves in steps of 0.062 and its AUC rests on that one patient. Long Beach VA has 45.
