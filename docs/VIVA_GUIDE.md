# Viva and demonstration guide

Everything below is checked against the code in this repository and the numbers in `reports/metrics.json`.
**Know your own part well, and know the headline numbers from every part.** Responsibilities follow the marking scheme:

| Student | Responsibility | Time on the day |
|---|---|---|
| **Student 1 · Abdul Rafe** | Dataset description and preprocessing | about 2-3 min |
| **Student 2 · Muntasir Ali Khan** | ML model implementation | about 2-3 min |
| **Student 3 · Shaikh Abdullah Saud** | Model evaluation, results and overall demonstration | about 3-4 min |

*(Students are assigned in the order they appear on the proposal slide. Swap the names if you agreed otherwise.)*

## The 30-second story (everyone should be able to say this)

> We predict the **exact selling price, in rupees, of a used car** from seven listing details. We use 3,577 clean CarDekho listings, a **Linear Regression (Ridge)** baseline and a tuned
> **Random Forest**, both inside one scikit-learn pipeline that learns the log of the price. On 716 unseen cars Linear Regression reaches **R² 0.87** and a typical error of
> **about 15%** (mean error Rs 89,000); Random Forest reaches R² 0.83. Guessing the average price gives R² 0 and a 58% typical error. The app also shows a likely price range.

## Numbers to remember

| | |
|---|---|
| Raw rows / columns | 4,340 / 8 (7 inputs + target) |
| Exact duplicates removed | 763 (17.6%) → **3,577** clean rows |
| Prices | min Rs 20,000 · median Rs 3.5 lakh · mean Rs 4.74 lakh · max Rs 89 lakh · skewness 5.5 (log price: -0.02) |
| Split | 80:20 balanced by price decile, `random_state = 42` → 2,861 train / 716 test |
| Test R² | baseline 0.000 · **LR 0.871** · RF 0.834 |
| Mean absolute error | baseline Rs 2,93,893 · **LR Rs 89,212** · RF Rs 98,562 |
| Median % error | baseline 58.0% · **LR 15.5%** · RF 16.3% |
| Within ±10% / ±20% of real price | LR 33.2% / 62.0% · RF 31.7% / 59.2% |
| Cross-validated R² (train set) | LR 0.796 · RF 0.774 |
| Train R² | LR 0.837 · RF 0.952 (Random Forest overfits) |
| Likely range | about × 0.71 to × 1.37 of the estimate, contains the real price for 80% of test cars |

---

## Student 1 · Dataset and preprocessing

**What to show:** notebook Parts 1-2, `reports/figures/01-06`, `src/data_prep.py`, `src/features.py`.

**What to say (about 2 minutes).** The data is the Kaggle CarDekho file: 4,340 used-car listings, 7 input columns and the selling price, which is our target, so this is a **regression** problem.
Prices are very right-skewed, so the models learn the logarithm of the price and the answer is converted back to rupees. We checked missing values (none), removed 763 exact duplicates
**before** splitting so identical cars cannot appear in both train and test, clipped extreme mileage, engineered car age, km per year, brand, model, brand tier and name keywords,
one-hot encoded the categories, scaled the numbers, and did an 80:20 split balanced by price. All of this sits inside one pipeline so the test set never influences training.

### Likely questions

1. **Is this classification or regression?** Regression: the target is a number (the price in rupees). We first looked at price bands, then changed to exact prices because that is more useful to a buyer or seller.
2. **Why do you predict log(price)?** Prices are right-skewed (skewness 5.5): a few luxury cars cost 20 times the median. On the log scale the distribution is almost symmetric (-0.02), and an error of 10% counts the same for a
   Rs 1.5 lakh hatchback and a Rs 30 lakh SUV. We tested it: without the log, Linear Regression's median error goes from 15.5% to 24.6% and it predicts negative prices for 38 test cars.
3. **What is data leakage and where did you avoid it?**
   Leakage is when information from the test set influences training. We avoided it by (a) removing duplicates before the split, and (b) fitting the imputers, outlier fences, scaler and encoder
   inside a `Pipeline`, so they are re-fitted on the training part only, including in every cross-validation fold. In the notebook (section 2.2) we show that splitting first would put about
   a quarter (25.8%) of the test cars in the training data as exact copies.
4. **How did you handle missing values?** There are none in this file. We still added median and most-frequent imputers inside the pipeline so the app would not crash on an empty field.
5. **Why did you remove duplicates? Could they be real, separate listings?**
   Possibly, but with no listing ID we cannot tell, and keeping them lets a model "memorise" cars it will be tested on. Removing exact duplicates is the safe choice; we state it as an assumption.
6. **How did you handle outliers?** For numeric inputs we clip values beyond Q3 + 3 x IQR to that fence (fences learned on training data). We do not delete rows. We do **not** clip the price: it is the target, and a Rs 89 lakh car is a real car.
7. **Why clip instead of delete?** Deleting would remove real cars and shrink an already small dataset. Clipping keeps the row but stops one wild value (806,599 km) from distorting the scaler and the Linear Regression weights.
8. **How did you encode categories?** One-hot encoding. Levels seen fewer than 10 times are pooled into one "infrequent" column, and unseen levels at prediction time are handled, so a brand-new model name does not crash the app.
9. **Why scale features?** Ridge regression is sensitive to feature scale, so we standardise the numeric columns. Random Forest does not need it; we use the same preprocessing for both so the comparison is fair.
10. **What features did you engineer and why?** `car_age` (price falls with age), `km_per_year` (usage intensity), `owner_count` (ordered), `brand` and `model` parsed from the name, `brand_tier` (Economy / Mid-range / Luxury) and `name_keywords` for variant words such as "vdi".
11. **How did you choose the brand tiers?** By our own domain knowledge; the lists are in `src/config.py`. It is an assumption, and the ablation shows it is a minor feature.
12. **Why is 2020 the reference year?** The newest listing is from 2020, so car age is measured against 2020. This also means the model cannot price newer cars or today's market.
13. **Why is the split "balanced by price decile"?** A plain random split could put most luxury cars in one part by chance. Splitting within each price decile keeps the price spread the same in train and test.
14. **Why `random_state = 42`?** So the split, the cross-validation and the Random Forest give the same numbers every time we run the project.
15. **Which features matter most?** Car identity (brand, model, variant) and year. Removing identity raises the cross-validated median error from about 15% to 28-30%; removing age and usage raises it to about 23-24%.

---

## Student 2 · Model implementation

**What to show:** notebook Part 3, `src/pipeline.py`, `src/train.py`, `reports/figures/13_grid_search.png`.

**What to say (about 2-3 minutes).** We picked two regression models: **Linear Regression with an L2 penalty (Ridge)** as an interpretable baseline and **Random Forest Regressor** as a flexible
non-linear model that most of our reference papers used. Linear Regression adds up weighted features to give the log price; the penalty keeps the weights stable. Random Forest grows many decision trees
on random samples of cars and features and averages their answers. Both are wrapped so they learn log(price) and answer in rupees. We tuned both with Grid Search and 5-fold cross-validation on the
training set only, scored by RMSLE, and compared against a dummy model that always predicts the average price.

### Likely questions

1. **Explain Linear Regression in simple words.** It finds weights so that price (on the log scale) ≈ b0 + b1·age + b2·km + b3·(brand) + ... by minimising the squared errors on the training cars.
2. **What is Ridge / `alpha`?** It adds a penalty on the size of the weights (L2 regularisation). A larger `alpha` shrinks weights more, which helps when many one-hot columns are correlated. We tried 0.01 to 30 and the best was `alpha = 1`.
3. **Explain Random Forest in simple words.** Many decision trees, each trained on a random sample of the cars and a random subset of features at every split. Each tree predicts a price; the forest averages them. Averaging many different trees reduces overfitting compared with one deep tree.
4. **What is bagging?** Training each tree on a bootstrap sample (random sample with replacement) and combining their answers.
5. **Which Random Forest settings did you tune?** `n_estimators` (200, 400), `max_depth` (none, 25), `min_samples_leaf` (1, 2) and `max_features` (0.2, 0.33, 0.5): 24 combinations. Best: 400 trees, no depth limit, `min_samples_leaf = 1`, `max_features = 0.33`.
6. **Why two algorithms?** A simple, explainable baseline against a more flexible model tells us whether the extra complexity is worth it. (Here it was not: Linear Regression is slightly better.)
7. **What is Grid Search?** It tries every combination of the listed settings (8 for Ridge, 24 for RF), scores each with cross-validation and keeps the best.
8. **What is 5-fold cross-validation?** The training set is split into 5 parts. The model trains on 4 parts, is scored on the 5th, and this repeats 5 times. The average score is a more reliable estimate than a single split.
9. **Why did you tune on the training set only?** The test set must stay unseen so the final score is honest. If we tuned on it we would be fitting to it.
10. **Why RMSLE to choose the model?** It measures error on the log price, so a 10% miss counts the same on cheap and expensive cars. Plain RMSE would be dominated by a few luxury cars.
11. **What is the dummy baseline for?** It always predicts the average price and scores R² 0, a 58% typical error. Any real model must beat that, otherwise it has learned nothing.
12. **How does `TransformedTargetRegressor` work?** It applies `log` to the price before training and `exp` to the prediction afterwards, so the user always sees rupees.
13. **Why does Random Forest overfit more?** Trees can memorise training cars: training R² is 0.95 against 0.83 on test. Linear Regression is 0.84 against 0.87 (it does not overfit).
14. **How does the pipeline work?** Raw columns → feature engineering → imputing, clipping, scaling and one-hot encoding → regressor. The same fitted pipeline is saved with `joblib` and reused by the app, so training and prediction are always consistent.
15. **Which tools did you use?** Python, pandas, NumPy, scikit-learn, matplotlib and Streamlit, in Google Colab or locally.

---

## Student 3 · Evaluation, results and demo

**What to show:** notebook Parts 4-5, `reports/figures/07-12`, then the Streamlit app.

**What to say (about 3 minutes).** This is a regression problem, so we use R², mean absolute error in rupees, median percentage error and "share of cars within ±20%", on 716 test cars the models never saw.
Linear Regression gets R² 0.87, a mean error of Rs 89,000 and a typical error of 15.5%; Random Forest gets 0.83, Rs 99,000 and 16.3%; guessing the average gets R² 0 and 58%. About 6 in 10 cars are priced within
±20%. Year and car identity drive the price. Linear Regression is slightly ahead, so we do not claim Random Forest wins. The app shows a likely range because no 7-input model can know a car's condition.
The main limitation is that the data are 2020 listings.

### Likely questions

1. **Define R², MAE, RMSE and MAPE.** R²: share of the price variation the model explains (1 is perfect, 0 is no better than the average). MAE: average size of the error in rupees. RMSE: like MAE but punishes big misses more. MAPE: average error as a % of the real price. We also report the median % error, which is not swayed by a few big misses.
2. **Is R² 0.87 good?** It is far better than the baseline (0.00) and our typical error is 15%. It is not good enough to use as a valuation, so the app shows a range (about -29% / +37%) that held for 80% of test cars.
3. **Which model is better?** Linear Regression, slightly: R² 0.871 vs 0.834 and MAE Rs 89,212 vs Rs 98,562. The 95% bootstrap interval of the R² gap is -0.067 to -0.008 and of the MAE gap Rs 3,100 to Rs 15,800, so the lead is real for those metrics; for median % error the two are statistically tied.
4. **Why did Random Forest not win, when papers say it does?** Those papers use other datasets and often report R² on raw prices, where expensive cars dominate. Here the signal is mostly smooth (age, mileage, brand), which a linear model captures well on the log scale; the forest overfits 3,577 rows. (On the raw price Random Forest does reach R² 0.85, but its median error is still about 16%.)
5. **Why is the median error (15.5%) bigger than you would like?** Cars with the same listed details differ in condition, colour, location and the seller's urgency; our 7 inputs cannot see that.
6. **Where does it do worst?** Cars above Rs 10 lakh (only 61 in the test set; median error 21%, MAE Rs 3.8 lakh) and very cheap cars (median error 21% because a small rupee error is a large percentage).
7. **What does the predicted-vs-actual chart show?** Points close to the diagonal mean a good prediction; the dashed lines are ±20%. The cloud is tight in the middle and wider at the ends.
8. **What does the error histogram show?** Most errors are within ±20%; the long tail is on the **too expensive** side, so the model occasionally overprices a car whose condition was poor.
9. **What does the feature-importance chart tell us?** Car identity (name) and year dominate (Rs 1-1.75 lakh of extra error when shuffled), then fuel and transmission; owner and seller type barely matter. This agrees with the drop-one-group test.
10. **How did you build the likely price range?** From out-of-fold predictions on the training set: the 10th and 90th percentiles of actual price ÷ predicted price give multipliers (0.71 and 1.37). We checked it on the test set: it contained the real price for 80.4% of cars (target 80%).
11. **What are the limitations?** 2020 prices only; just 7 inputs (no condition, accident history, location); listing prices rather than confirmed sales; small data; brand tiers are our judgement; and the model under-prices a newer, pricier market.
12. **What happened on the v4 file?** v4 is a pricier CarDekho file (median Rs 8.25 lakh vs Rs 3.5 lakh). R² falls to 0.38 (Linear) and 0.30 (RF), and both models under-price by about 10-14%. Typical error is still about 17-20%. It shows prices drift over time, so the model needs fresh data.
13. **How would you improve it?** Add condition and location information, try gradient boosting, use fresh dated listings. Extra specifications (engine, power, mileage, seats) gave a small gain (about 1 point of % error) on the larger v3 file.
14. **Does it overfit?** Random Forest does: train R² 0.95 vs test 0.83. Linear Regression does not: 0.84 vs 0.87. The learning curves show the same.
15. **What if I enter a 2023 car?** The model was trained on listings up to 2020, so the app limits the year to the data range and warns about it. It cannot price newer cars or today's market.

---

## Live demonstration script (about 3 minutes)

Start the app before you are called: `streamlit run app/streamlit_app.py`. If you have no laptop on the day, use the Colab notebook's Part 5 table.
These are **real test-set cars** (the models never saw them), so you can quote the real price next to the prediction. Choose the brand and model in the dropdowns, type the variant as shown.

| Step | Enter in the **Predict** tab | Real price | Linear / Random Forest estimate |
|---|---|---|---|
| 1 | Maruti · Alto · variant `LX BSIII` · 2008 · 90,000 km · Petrol · Manual · Individual · Second Owner | Rs 1,10,000 | Rs 1,09,000 / Rs 1,09,000 |
| 2 | Maruti · Swift · variant `VDI BSIV` · 2015 · 60,000 km · Diesel · Manual · Individual · First Owner | Rs 4,75,000 | Rs 4,84,000 / Rs 4,74,000 |
| 3 | Hyundai · Verna · variant `1.6 VTVT AT S Option` · 2016 · 80,000 km · Petrol · Automatic · Individual · First Owner | Rs 6,90,000 | Rs 6,60,000 / Rs 7,03,000 |
| 4 | Toyota · Innova · variant `Crysta 2.4 ZX MT` · 2017 · 91,195 km · Diesel · Manual · Dealer · First Owner | Rs 13,30,000 | Rs 14,30,000 / Rs 13,25,000 |
| 5 | **A miss, shown on purpose:** BMW · X1 · variant `sDrive20d` · 2012 · 1,20,000 km · Diesel · Automatic · Individual · Second Owner | Rs 7,00,000 | Rs 11.6 lakh / Rs 11.4 lakh |

What to say: for steps 1-4, "the estimate is within a few percent, and the real price is inside the likely range". For step 5: "this seller asked far less than the model expects, even below our likely range (Rs 8.2 to 15.8 lakh); condition, accident history
or urgency are not among our 7 inputs. That is exactly why we show a range and call it an estimate." Then:

| Step | Do | Say |
|---|---|---|
| 6 | Point to the **"Why this price?"** chart | "Each bar swaps one input for a typical car's value and shows how many rupees the estimate changes." |
| 7 | Scroll to **Comparable listings** | "Real listings of the same model with the nearest year, for a sanity check." |
| 8 | Open **Model performance**, then **Data insights** | R², MAE and error charts; the cleaning numbers |

The numbers come from the saved models and can shift by a few thousand rupees only if you retrain with a different scikit-learn version.

## Questions about the project in general

* **Why did you change from price bands to exact prices?** A buyer or seller wants a number, not a category. The regression model is more useful, and we kept an analysis of errors by price range to see where it is weaker.
* **What would you do differently with more time?** Collect fresh dated listings with condition information and try gradient boosting.
* **Who did what?** See the table at the top; each of us can also explain the other two parts at the level of the 30-second story.
* **Is the code reproducible?** Yes: fixed seeds, one-command rebuild (`python -m src.run_all`), the same numbers in the notebook, the saved reports and the tests.
* **Where is the data from and is it allowed?** Kaggle, `nehalbirla/vehicle-dataset-from-cardekho`; included for coursework, and the licence terms are on the Kaggle page.
