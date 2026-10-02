| key | protocol | training rows | cv_mae | test_mae | test_minus_cv | optimism_vs_D |
|:---|:---|---:|---:|---:|---:|---:|
| A | DSPRO1: listings, LOO kNN, random folds | 5521 | 249.6 | 270.7 | 21.1 | 25.8 |
| B | listings, LOO kNN, grouped folds | 5521 | 272.0 | 270.7 | -1.3 | 3.4 |
| C | objects, LOO kNN, grouped folds | 4875 | 273.0 | 271.9 | -1.1 | 3.6 |
| D | objects, OOF kNN, grouped folds | 4875 | 277.4 | 272.7 | -4.7 | 0.0 |

