---
title: "Practical Machine Learning course project"
author: "Ray Bem"
date: "10/17/2020"
---





## Synopsis

### Exploratory Data Analysis

The data came in the form of a comma separated file with 160 features and 19622 observations.  

#### Character and Time variable reduction
Sample prints and diagnostics identified a small subset of variables that we set aside as they describe elements not included in this effort.  For example, the `user_name` variable adds a user-specific dimension if included in the models, we are trying to build here a classifier independent of user.  Also, the usefullness of the time variables was difficult to assess, the [original paper](http://web.archive.org/web/20161224072740/http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf) goes into more detail on how these "yes" records are calculated (e.g., sliding 2.5s intervals).  The table below summarizes our first attempt to reduce covariates.


Table: **Character and Time variables removed from consideration (n=7)**

|varname              | varindex|description            |reason_removed                                              |
|:--------------------|--------:|:----------------------|:-----------------------------------------------------------|
|X1                   |        1|a sequential record id |obviously this would distort the model if left in           |
|user_name            |        2|study participant name |we want to predict classe for any person, therefore removed |
|raw_timestamp_part_1 |        3|raw timestamp part 1   |time disregarded in this analysis                           |
|raw_timestamp_part_2 |        4|raw timestamp part 2   |                                                            |
|cvtd_timestamp       |        5|converted timestamp    |                                                            |
|new_window           |        6|new_window             |these are the 'yes' data, n=406                             |
|num_window           |        7|num_window             |an incremental window counter                               |

With relatively high dimensionality, summary functions were used to highlight characteristics.  The first aspect examined was the `new_window` variable.  This was determined to be a summarization of detail records tagged with "no", and set aside early in the code.  In the "yes" data, most columns had more data for several fields, when compared with the "no" data.  As our goal is the simplest classifier as possible, and the summary data looked so much different than the bulk of the data, these were left out (n=406).  The remainder of this analysis focuses on the "no" data, 19216 records.

#### Highly correlated variable reduction
The `cor` function was used to identify correlations above 80 percent, the user sets the cutoff value at the top of the code. Currently this is set at 0.8, we found 22 highly correlated variables in the "no" new_window data.  As we know the nature of the measurements should include things that will correlate (there are only three dimensions being measured on a single body with a limited, intended motion), we leave some correlated covariates in the model (i.e., we don't exclude all correlated varaiables).  Below is a summary drawn from the `varmap` dataset created in the processing:


Table: **High Correlation variables removed from consideration (n=22, sample of 10 below)**

|varname          | varindex|high_corr_no   |
|:----------------|--------:|:--------------|
|roll_belt        |        8|corr above 0.8 |
|pitch_belt       |        9|corr above 0.8 |
|yaw_belt         |       10|corr above 0.8 |
|total_accel_belt |       11|corr above 0.8 |
|accel_belt_x     |       40|corr above 0.8 |
|accel_belt_y     |       41|corr above 0.8 |
|accel_belt_z     |       42|corr above 0.8 |
|magnet_belt_x    |       43|corr above 0.8 |
|gyros_arm_x      |       60|corr above 0.8 |
|gyros_arm_y      |       61|corr above 0.8 |

#### Low or near-zero variance variable reduction
In a similar fashion, the r function `nearzeroVariance` was used to identify variables that would add little additional information to our models.  We again examined the output and concluded these would be left out.  This process identified 100 variables with low variance, these were removed.


Table: **Near-zero Variance variables removed from consideration (n=100, sample of 10 below)**

|varname              | varindex| freqRatio| percentUnique|zeroVar |
|:--------------------|--------:|---------:|-------------:|:-------|
|kurtosis_roll_belt   |       12|         0|             0|TRUE    |
|kurtosis_picth_belt  |       13|         0|             0|TRUE    |
|kurtosis_yaw_belt    |       14|         0|             0|TRUE    |
|skewness_roll_belt   |       15|         0|             0|TRUE    |
|skewness_roll_belt.1 |       16|         0|             0|TRUE    |
|skewness_yaw_belt    |       17|         0|             0|TRUE    |
|max_roll_belt        |       18|         0|             0|TRUE    |
|max_picth_belt       |       19|         0|             0|TRUE    |
|max_yaw_belt         |       20|         0|             0|TRUE    |
|min_roll_belt        |       21|         0|             0|TRUE    |

#### Missing data
Finally, missing data were explored.  At this point we have only the detail "no" data, and highly correlated and near-zero variance variables have been removed.  We observe no missing data, therefore our model choices do not require any imputation of data.

#### Final data
This yields a final modelling dataset having the following characteristics...

## Building the classification model
A *modelling process* was built to more easily explore the data.  The R package `caret` was used to create data partitions separating training data into a set to build models on, and a set to test.  These test data are used afterwards to assess our estimated out of sample error rate (for application to new study subjects).  

Since processing time is of some consideration here, we *explored* a random sample of the training data for the GBM model grid, and gave the faster treebag and ldabag models a more realistic 70/30 training/validation split.  Note that later, final model comparisons are done with all settings identical (70/30, 5 times repeated 10-fold resampling).

The summarized model results were then examined and the model chosen.  In this code an option exists to select what R considers the "best" model, or the user can choose a less precise version of the model, in consideration of overtraining.  The top five "next best" models generated from the grid are available.  For example, the GBM models do well for these data, but if the variation in a new sample were high, due to say a different set of participants, the option to back off the best fitting model and use a more simplified set of parameters exists.

#### Gradient Boosting Model (GBM)





The Gradient Boosting Model was of interest -- particularly in the spirit of model tuning.  A `caret` grid of model tuning features was built to generate estimates using a variety of the `gbm` tuning parameters.  Below are the results, where one can see the increase in model performance as we adjust the required minimum in each branch of the tree (these form the columns), as well as how much information is retained for future branch development (shrinkage, forming the plot rows).  The plots themselves are of increasing accuracy, as we subject the model to more boosting iterations.  The model results are gathered and summarized, and we have a set of 270 model objects in the end.  Using this approach, we have models competing with each other on various terms, and we can home in on patterns observed in the cross validation.

<img src="index_files/figure-html/plots gbm-1.png" style="display: block; margin: auto;" />

Each sub-plot has a color for the tree depths, where we see a reference depth of one, and more realistic depths of 5 and 10, with these having much higher Accuracy in the resampling. Another observation would be that the matrix above suggests requiring stricter pathways has noticeably less accuracy -- that is, a minimum of 100 in each node had to be satisfied, leaving fewer choices for the model to more tightly decide `classe`.  Also noteworthy are the effects of the shrinkage adjustments, where one observes initial Accuracy gains directly related to this tuning feature.

#### Bagged, Boosted Trees (treebag)
A second model was built using `treebag`, this model processes at a much faster rate than the GBM grid.  There are no tuning options for `treebag`.  A comparison follows.



#### Bagged Linear Discriminate Analysis (lda)
A third `lda` model was built as well, again there are no tuning parameters for this model, and similar to the `treebag` the processing time was reasonable (less than 15 minutes).



## Selecting the Classification Model

#### Cross Validation
Cross validation was performed using 10-folds of the training data, repeated five times.  Output below shows not only differences in Accuracy (placement on plot), but the pattern of solutions (indicated by the width), including variance.








```
$Accuracy
             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
gbm     0.9736842 0.9759118 0.9778454 0.9785519 0.9804856 0.9848327    0
ldabag  0.5828996 0.5847303 0.5941572 0.5938220 0.6032489 0.6040737    0
treebag 0.9467737 0.9477773 0.9491488 0.9517139 0.9530855 0.9617844    0
```

```
$Kappa
             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
gbm     0.9666976 0.9695174 0.9719658 0.9728590 0.9753073 0.9808067    0
ldabag  0.4725524 0.4743415 0.4863220 0.4859216 0.4979021 0.4984900    0
treebag 0.9326115 0.9339009 0.9356300 0.9388769 0.9406060 0.9516360    0
```

<img src="index_files/figure-html/resamples analysis-1.png" style="display: block; margin: auto;" />

We can see above that we would have XXXXX 

The variables selected as important are below:

## Testing the models
With a final model chosen (in the case of GBM), we have three models to run against the validation subset of our data.  Recall these models were built on 30% of the training data, leaving 70% as a completely fresh sample.  

From the `caret` documentation...For multi-class outcomes, the problem is decomposed into all pair-wise problems and the area under the curve is calculated for each class pair (i.e. class 1 vs. class 2, class 2 vs. class 3 etc.). For a specific class, the maximum area under the curve across the relevant pair-wise AUC’s is used as the variable importance measure.

<div class="figure" style="text-align: center">
<img src="index_files/figure-html/plot predictions-1.png" alt="**Prediction versus Truth**"  />
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
