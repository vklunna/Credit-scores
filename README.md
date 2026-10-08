# Credit Risk Scorecard on Lending Club Data

## Overview

A lender has to decide which loan applicants to approve, using only the information available at the moment someone applies. This project builds a bank-style credit scorecard for that decision on Lending Club's public loan data: it predicts each applicant's probability of default (PD), turns it into a points-based credit score, estimates expected loss in dollars, and checks whether the model stays reliable over time.

Everything is built step by step in pandas and scikit-learn, including a hand-written Weight of Evidence (WoE) pipeline, rather than relying on a scorecard library.

## Key results

| Model | Features | Gini (test) |
|---|---|---|
| A: own scorecard | borrower application data only | **0.339** |
| B: with Lending Club's assessment | as A, plus Lending Club's sub-grade | **0.385** |

- **The score separates risk clearly.** On the test set, the riskiest 10% of borrowers defaulted at 32.1% and the safest 10% at 4.3%, with default rates falling in every score band in between.
- **The model ranks well but underestimates the level of risk.** Expected loss was $962 per test loan against an actual loss of $1,209, about 20% too low. The main cause: the model predicts an average PD of 14.0% (the training default rate), while the test period defaulted at 16.0%.
- **The population was stable, but the outcome wasn't.** The Population Stability Index (PSI) between train and test is 0.003, far below the usual 0.1 warning level. Yet borrowers with the same score defaulted 2–3 percentage points more often in the test years. A stable PSI alone would have missed this.

| Score band (train edges) | Train default rate | Test default rate |
|---|---|---|
| lowest | 28.5% | 31.5% |
| middle | 13.6% | 15.9% |
| highest | 3.9% | 4.4% |

## Data

- **Source:** Lending Club accepted loans, 2007–2018 Q4 (about 2.26 million loans, 145 columns), available on Kaggle.
- **Target:** default = 1 for "Charged Off", "Default" and "Does not meet the credit policy. Status: Charged Off"; 0 for "Fully Paid" and "Does not meet the credit policy. Status: Fully Paid".
- **Unfinished loans removed:** "Current", "In Grace Period" and "Late" loans were dropped, since their outcome is unknown.
- **Only fully matured loans kept:** the data ends in December 2018, so 36-month loans issued after 2015 and 60-month loans issued after 2013 had not had time to finish. Among those recent loans, the ones that had finished were a biased sample (early payoffs and early defaults), which distorted default rates. Keeping only loans that could have run their full term left 676,301 loans.

## Approach

### Leakage removal
The model may only use information known at application time. Columns recorded after the loan was issued were removed, including payments received (`total_pymnt`, `total_rec_prncp`, `last_pymnt_d`, ...), post-default recoveries (`recoveries`, `collection_recovery_fee`), hardship and settlement fields, and updated credit data (`last_fico_range_*`). IDs, free text, columns over 50% missing and constant columns were also dropped.

### Cleaning and feature engineering
Text fields were converted to numbers (`term`, `emp_length`), dates were parsed, and a new feature, `credit_history_years`, was created from the gap between the first credit line and the loan's issue date. Rows with impossible debt-to-income values (negative or the placeholder 999) were removed. For highly correlated pairs, only one variable was kept, for example `loan_amnt` over `installment`, which is calculated from it.

### Out-of-time split
To mimic how a lender uses a model on future applicants, the data was split by time rather than at random. Each loan term is tested on its most recent matured year:

| Set | Loans | Default rate |
|---|---|---|
| Train: 36-month up to 2014, 60-month up to 2012 | 358,894 | 13.9% |
| Test: 36-month from 2015, 60-month from 2013 | 317,407 | 16.0% |

### WoE binning and IV selection
Numeric variables were split into 10 equal-sized bins and categorical variables kept their categories. Each bin was replaced by its Weight of Evidence, a log-odds measure of how safe or risky it is. Bin edges and WoE values were learned on the training set only, then applied unchanged to the test set (outer edges opened to ±infinity; unseen categories get a neutral WoE of 0).

Information Value (IV) was used to rank variables. Variables below 0.02 were dropped. No variable exceeded 0.5, which supports that the leakage removal worked. The strongest predictors were Lending Club's own sub-grade, interest rate and grade (IV ≈ 0.30–0.33), followed by FICO score (0.12) and annual income (0.07).

### Logistic regression and sign check
Two models were trained on the WoE features. Since Lending Club's grade, sub-grade and interest rate are its own risk assessment, Model A excludes all three and Model B keeps only the sub-grade.

With WoE inputs, every coefficient should be negative (a safer bin should lower the default probability). Variables with positive coefficients, caused by overlap with other variables, were removed. This changed the Gini by less than 0.001, confirming they added noise rather than information.

### Score scaling and points table
Default probabilities were converted into credit scores using the industry-standard scaling: 600 points at good:bad odds of 50:1, with 20 points to double the odds. The score was then split into points per variable, so each borrower's score can be explained line by line (for example, how many points they gained or lost for their FICO band or loan purpose).

### Expected loss
Expected loss = PD × LGD × EAD.

- **PD:** the model's predicted probability of default.
- **LGD:** estimated from defaulted training loans as the share of the loan amount not recovered (principal repaid plus recoveries, minus collection fees). Average LGD was 53.1%, and almost identical in the test set. These recovery columns were excluded from the PD model as leakage, but are the right data for measuring past losses.
- **EAD:** the original loan amount.

### Stability (PSI)
Train and test scores were placed into the same 10 score bands (built on train), and the PSI compared the share of borrowers in each band: PSI = Σ (test% − train%) × ln(test% / train%).


