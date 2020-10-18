---
title: "Practical Machine Learning course project"
author: "Ray Bem"
date: "10/17/2020"
output: 
  html_document: 
    keep_md: yes
---





## Synopsis

### Exploratory Data Analysis

The data came in the form of a comma separated file with 160 features and 19622 observations.  

#### Character and Time variable reduction
Sample prints and diagnostics identified a small subset of variables that we set aside as they describe elements not included in this effort.  For example, the `user_name` variable adds a user-specific dimension if included in the models, we are trying to build here a classifier independent of user.  Also, the usefulness of the time variables was difficult to assess, the [original paper](http://web.archive.org/web/20161224072740/http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf) goes into more detail on how these `new_window = yes` records are calculated (e.g., sliding 2.5s intervals).  The table below summarizes our first attempt to reduce covariates.


Table: **Character and Time variables removed from consideration (n=7)**

|varname              | varindex|description            |reason_removed                                              |
|:--------------------|--------:|:----------------------|:-----------------------------------------------------------|
|X1                   |        1|a sequential record id |obviously this would distort the model if left in           |
|user_name            |        2|study participant name |we want to predict classe for any person, therefore removed |
|raw_timestamp_part_1 |        3|raw timestamp part 1   |time disregarded in this analysis                           |
|raw_timestamp_part_2 |        4|raw timestamp part 2   |                                                            |
|cvtd_timestamp       |        5|converted timestamp    |                                                            |
|new_window           |        6|new_window             |used to separate 'yes' data, n=406                          |
|num_window           |        7|num_window             |an incremental window counter                               |

With relatively high dimensionality, summary functions were used to highlight characteristics.  The first aspect examined was the `new_window` variable.  This was determined to be a summarization of detail records tagged with "no", and set aside early in the code.  In the "yes" data, most columns had more data for several fields, when compared with the "no" data.  As our goal is the simplest classifier as possible, and the summary data looked so much different than the bulk of the data, these were left out (n=406).  The remainder of this analysis focuses on the "no" data, 19216 records.

#### Highly correlated variable reduction
The `cor` function was used to identify correlations above 80%, the user sets the cutoff value at the top of the code. Currently this is set at 80%, we found 22 highly correlated variables at this cutoff in the "no" new_window data.

As a processing aside, some of the tests were performed using R functions as opposed to simply allowing the `caret` package to perform during the `preProcess`.  This was done to allow more insight into the data, particularly during the exporatory phase.

As we know the nature of the measurements should include things that will correlate (there are only three dimensions being measured on a single body with a limited, intended motion), we leave some correlated covariates in the model (i.e., we don't exclude all correlated variables).  For example, an analysis of a dance movement would have more complicated relationships and a more strict reduction of covarying covariates might be fine, here, the motion is so limited.  Below is a summary drawn from the `varmap` dataset created in the processing, displaying a sample of these exclusions:


Table: **High Correlation variables removed from consideration (n=22, sample of 5 below)**

|varname          | varindex|high_corr_no   |
|:----------------|--------:|:--------------|
|roll_belt        |        8|corr above 0.8 |
|pitch_belt       |        9|corr above 0.8 |
|yaw_belt         |       10|corr above 0.8 |
|total_accel_belt |       11|corr above 0.8 |
|accel_belt_x     |       40|corr above 0.8 |

#### Low or near-zero variance variable reduction
In a similar fashion, the R function `nearzeroVariance` was used to identify variables that would add little additional information to our models.  We again examined the output and concluded these would be left out.  This process identified 100 variables with zero variance, these were removed.


Table: **Near-zero Variance variables removed from consideration (n=100, sample of 5 below)**

|varname              | varindex| freqRatio| percentUnique|zeroVar |
|:--------------------|--------:|---------:|-------------:|:-------|
|kurtosis_roll_belt   |       12|         0|             0|TRUE    |
|kurtosis_picth_belt  |       13|         0|             0|TRUE    |
|kurtosis_yaw_belt    |       14|         0|             0|TRUE    |
|skewness_roll_belt   |       15|         0|             0|TRUE    |
|skewness_roll_belt.1 |       16|         0|             0|TRUE    |

#### Missing data
Finally, missing data were explored.  At this point we have only the detail "no" data, and highly correlated and near-zero variance variables have been removed.  We observe no missing data, a convenience as our model choices do not require any imputation of data.

#### Final data
This yields a final modeling dataset having the following characteristics...at this point the dimensionality has been thoughtfully reduced, addressing covariance, near-zero variance, and missing data realities.  

Before we traded computational speed for accuracy, in an effort to see the behavior of the model tuning features (for GBM).  Some observations:

1. Even with a great deal of dimension reduction, we have some hopeful model results -- we are not way off
2. Adjusting how the GBM models are fit allows some flexibility to react to overfitting (by choosing a less than "best" solution)
3. The same variables are seen as important across models (covered later)

The model building comes next, where the amount of data and repeated cross validation are expanded.


Table: Sample Print of final dataset

|classe | gyros_belt_x| gyros_belt_y| gyros_belt_z| magnet_belt_y| magnet_belt_z|
|:------|------------:|------------:|------------:|-------------:|-------------:|
|A      |         0.00|         0.00|        -0.02|           599|          -313|
|A      |         0.02|         0.00|        -0.02|           608|          -311|
|A      |         0.00|         0.00|        -0.02|           600|          -305|
|A      |         0.02|         0.00|        -0.03|           604|          -310|
|A      |         0.02|         0.02|        -0.02|           600|          -302|



Table: Dimensions

| observations| variables|
|------------:|---------:|
|        19216|        31|



Table: Distribution of classe variable

|classe |    n|
|:------|----:|
|A      | 5471|
|B      | 3718|
|C      | 3352|
|D      | 3147|
|E      | 3528|



## Building the classification model
A *modeling process* was built to more easily explore the data.  The R package `caret` was used to create data partitions separating training data into a set to build models on, and a set to test.  These test data are used afterwards to assess our estimated out of sample error rate (for application to new study subjects).  

Since processing time is of some consideration here, we *explored* a smaller (10/90random sample of the training data for the GBM model grid, and gave the faster treebag and ldabag models a more realistic 70/30 training/validation split.  

The final model comparisons are done with all settings identical (70/30 training/test, 5 times repeated resampling on 10 folds).

The summarized model results were then examined and the model chosen.  In this code an option exists to select what R considers the "best" model, or the user can choose a less precise version of the model, in consideration of overtraining.  The top five "next best" models generated from the grid are available.  For example, the GBM models do well for these data, but if the variation in a new sample were high, due to say a different set of participants, the option to back off the best fitting model and use a more simplified set of parameters exists.

#### Gradient Boosting Model (GBM)



The Gradient Boosting Model was of interest -- particularly in the spirit of model tuning.  A `caret` grid of variations of model tuning features was built to generate estimates using a variety of the `gbm` tuning parameters.  This allowed for the exploration of 270 models in a convenient way (less code, obviously).

Below are the results, where one can see the increase in model performance as we adjust the required minimum in each branch of the tree (these form the columns), as well as how much information is retained for future branch development (shrinkage, forming the plot rows).  The plots themselves are of increasing accuracy, as we subject the model to more boosting iterations.  The model results are gathered and summarized, and we have a set of 270 model objects in the end.  

Using this approach, we have models competing with each other on various tuning features, and we can see the behavior and effects of the tuning on the Accuracy rates.  Again, these are possible given the smaller data for the GBM set, but clearly one can see how the tuning is working.

<img src="index_files/figure-html/plots-gbm-1.png" style="display: block; margin: auto;" />

Each sub-plot has a color for the tree depths, where we see a reference depth of one, and more realistic depths of 5 and 10, with these having much higher Accuracy in the resampling. Another observation would be that the matrix above suggests requiring stricter pathways has noticeably less accuracy -- that is, a minimum of 100 in each node had to be satisfied (column 3), leaving fewer choices for the model to more tightly decide `classe`.  Also noteworthy are the effects of the shrinkage adjustments, where one observes initial Accuracy gains directly related to this tuning feature, which controls the learning rate.  When dialed down to .1, the model "forgets" it's current mapping and has a wider set of choices to solve (ergo higher performance).

#### Bagged, Boosted Trees (treebag)
A second model was built using `treebag`, this model processes at a much faster rate than the GBM grid.  There are no tuning options for `treebag`.  A comparison will follow.



#### Bagged Linear Discriminate Analysis (LDA)
A third `lda` model was built as well, again there are no tuning parameters for this model, and similar to the `treebag` the processing time was reasonable (less than 5 minutes).  Next we will examine the resamples to check for out of sample error expectations.



## Selecting the Classification Model

#### Cross Validation
Cross validation was performed using 10-folds of the training data, repeated five times.  Output below shows not only differences in Accuracy (placement on plot), but the pattern of solutions, including variance (indicated by the width).







<img src="index_files/figure-html/resamples-analysis-1.png" style="display: block; margin: auto;" />

#### Variable Importance
Below are output from two of our models, the GBM model does not have an analog that works in `knitr`, but results are similar.  Noteworthy is the agreement across the two lists, though in a different format, each indicates the same important variables affecting the classifications.  For example, the magnet dimensions play a huge role in these classifiers.  

From the `caret` documentation...For multi-class outcomes, the problem is decomposed into all pair-wise problems and the area under the curve is calculated for each class pair (i.e. class 1 vs. class 2, class 2 vs. class 3 etc.). For a specific class, the maximum area under the curve across the relevant pair-wise AUC’s is used as the variable importance measure.


```
treebag variable importance

  only 20 most important variables shown (out of 30)

                     Overall
magnet_dumbbell_z     100.00
magnet_dumbbell_y      93.27
gyros_belt_z           92.08
magnet_belt_y          75.95
magnet_belt_z          70.09
roll_forearm           67.77
pitch_forearm          61.13
roll_dumbbell          57.50
accel_dumbbell_y       55.81
magnet_dumbbell_x      54.26
roll_arm               45.38
yaw_arm                39.35
magnet_forearm_z       34.78
gyros_dumbbell_y       34.45
total_accel_dumbbell   30.33
accel_forearm_x        30.03
accel_arm_y            27.08
pitch_arm              24.03
yaw_forearm            21.38
magnet_forearm_x       18.73
```

```
ROC curve variable importance

  variables are sorted by maximum importance across the classes
  only 20 most important variables shown (out of 30)

                        A       B      C      D       E
pitch_forearm       61.44 100.000 66.573 61.435 100.000
accel_forearm_x     40.86  81.625 40.860 40.860  81.625
magnet_forearm_x    36.47  69.768 31.478 31.478  69.768
magnet_dumbbell_y   45.51  44.978 44.978 65.243  45.510
magnet_belt_y       13.51  11.557 63.892 11.557  13.507
magnet_dumbbell_x   60.69  60.690 60.690 60.942  48.419
roll_dumbbell       38.12  50.291 27.183 55.828  50.291
magnet_dumbbell_z   54.72  34.685 54.693 23.109  54.718
pitch_arm           27.04  40.823 51.144 27.043  40.823
magnet_belt_z        1.39   3.509 46.430  1.011   3.509
total_accel_arm     28.37  43.804 34.890 16.665  43.804
magnet_forearm_y    22.41  41.311 33.989 23.867  41.311
accel_dumbbell_y    32.14  15.447 15.447 38.539  32.141
roll_arm            37.11  37.114 37.114 37.114  27.583
total_accel_forearm 23.74  27.099 33.543 23.738  27.099
roll_forearm        33.20  13.066  5.105 28.594  33.203
accel_arm_z         25.44  28.154 15.387 18.158  28.154
accel_forearm_y     26.82   7.466  5.605 25.020  26.821
yaw_forearm         12.92  24.323 14.395 20.781  24.323
accel_arm_y         16.65  20.662 20.199 16.953  20.662
```

## Testing the models -- predicting new data
With a final model chosen (in the case of GBM), we have three models to run against the validation subset of our data.  Recall these models were built on 10% of the training data, leaving 90% as a completely fresh sample.  

<div class="figure" style="text-align: center">
<img src="index_files/figure-html/plot-predictions-1.png" alt="**Prediction versus Truth**"  />
<p class="caption">**Prediction versus Truth**</p>
</div>

## Choices made

## Summary

### Citations
The data used in this analysis were graciously provided by the Human Activity Recognition website, which can be accesed [here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). Thank you HAR!

## System Information
This work was developed on the following system, using `R.version.string`:

      Model Name: iMac
      Processor Name: Quad-Core Intel Core i7
      Memory: 32 GB

The following R libraries were utilized:

`library(tidyverse)`
`library(rattle)`
`library(Hmisc)`
`library(corrgram)`
`library(caret)`
`library(gridExtra)`
`library(adabag)`
`library(fastAdaboost)`
`library(rlist)`
`library(stringi)`
