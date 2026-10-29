| key   | protocol                                |   training rows |   cv_mae |   test_mae |   test_minus_cv |   optimism_vs_D |
|:------|:----------------------------------------|----------------:|---------:|-----------:|----------------:|----------------:|
| A     | DSPRO1: listings, LOO kNN, random folds |            5531 |    247.9 |      274.5 |            26.6 |            26.3 |
| B     | listings, LOO kNN, grouped folds        |            5531 |    268.4 |      274.5 |             6.1 |             5.8 |
| C     | objects, LOO kNN, grouped folds         |            4893 |    270.6 |      279.7 |             9.1 |             8.8 |
| D     | objects, OOF kNN, grouped folds         |            4893 |    277.2 |      277.5 |             0.3 |             0.0 |
