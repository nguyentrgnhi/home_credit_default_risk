# Data

The data is from the Kaggle competition [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).
Kaggle's rules don't allow redistributing it, so it isn't in this repo.

1. Accept the competition rules on Kaggle and download the files (website, or `kaggle competitions download -c home-credit-default-risk`).
2. Unzip them into `data/raw/` so this exists:

```
data/raw/application_train.csv
data/raw/bureau.csv
data/raw/installments_payments.csv
data/raw/credit_card_balance.csv
```

Only these four files are used (only `application_train.csv` has labels). Expensive aggregations are cached as parquet files in `data/cache/`, which is also ignored by git.
