| key   | protocol                                |   training rows |   cv_mae |   test_mae |   test_minus_cv |   optimism_vs_D |
|:------|:----------------------------------------|----------------:|---------:|-----------:|----------------:|----------------:|
| A     | DSPRO1: listings, LOO kNN, random folds |           12544 |    309.3 |      320.3 |            11.0 |            26.5 |
| B     | listings, LOO kNN, grouped folds        |           12544 |    330.6 |      320.3 |           -10.3 |             5.2 |
| C     | objects, LOO kNN, grouped folds         |           11208 |    332.5 |      317.4 |           -15.2 |             0.4 |
| D     | objects, OOF kNN, grouped folds         |           11208 |    337.5 |      321.9 |           -15.5 |             0.0 |
