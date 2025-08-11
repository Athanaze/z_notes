# Why naive bootstrap CI is bad

```R
quantile(res, c(0.025, 0.975))
```
res contains bootstrap estimates of the correlation coefficient (each from resampling the original data with replacement).

This interval is just the 2.5% and 97.5% percentiles of the bootstrap distribution of the estimator.

Problem:
This assumes the bootstrap distribution is centered at the true parameter. But in reality, it’s centered at the observed estimate, which can be biased. As a result, the naive percentile CI can be shifted away from the true coverage.

# First solution : 

2. The basic bootstrap CI
The proper method they demonstrate is:

```R
q95. <- quantile(res - th.n, pr = c(0.025, 0.975))
ci95 <- th.n - q95.[2:1]
```
res - th.n is the bootstrap distribution of the error "theta hat star" - "theta hat"

q95. gives the quantiles of that error distribution.

th.n - q95.[2:1] inverts those errors to get bounds for the true parameter.

# Better, fancier way

```R
boot.ci(boot_object, type = "bca", conf = 0.95)  # 95% CI
```

# Lambda.1se

Look at the CV error curve: often there’s a wide, flat region around the minimum where many λ values perform similarly.

The 1-standard-error rule says:

Pick the largest λ whose CV error is within 1 standard error of the minimum CV error.


Larger λ → stronger regularization → simpler model (more coefficients shrunk to zero).

# Bandwith

Small Bandwith → lots of wiggles (low bias, high variance)

Large Bandwith → oversmoothed curve (high bias, low variance)

"SJ" : Sheater-Jones : optimal bandwith estimator


# TODO : hard to replicate this, maybe add parts to cheatsheet ?

## Week 6

```R
################################################################
# Cross validation and Smoothing Splines - Hyperparameter tuning
###############################################################

m_fct <- function (x) {
  ifelse(x < 0.5, exp(cos(6 * pi * x)),
         exp(-1) * ((x / 0.5) ^ 3)) / 3
}

set.seed(200)

# covariate
xvals <- seq(0, 1, by = 0.01)

# additive model 
yvals <- m_fct(xvals) + 0.1 * rnorm(length(xvals))

# data plot with true function m(x)
plot(xvals, yvals, type = 'p', xlab = 'x', ylab = 'f(x)')

lines(xvals, m_fct(xvals), type = 'l', xlab = 'x', ylab = 'f(x)', lty = 'dashed')

title('Observed Data + True Function')


# only data

plot(xvals, yvals, type = 'p', xlab = 'x', ylab = 'f(x)')
title('Observed Data Only')

# fit smoothing spline

spline_fit <- smooth.spline(xvals, yvals)

plot(xvals, yvals)

lines(xvals, predict(spline_fit, xvals)$y, type = 'l', xlab = 'x', ylab = 'f(x)', lwd = 2)

lines(xvals, m_fct(xvals), type = 'l', lty = 'dashed', col = "red", lwd = 2)
title('Smoothing Spline and True Function')
legend("top", c("Smoothing Spline", "True Function"), lty = c('solid', 'dashed'), lwd = c(2, 2), col = c("black", "red"))

# The degree of smoothness is controlled by an argument called spar \in (0,1]
# lambda is a monotone function of spar 
# help(smooth.spline) -> see spar parameter
print(paste("R's optimized spar value: ", spline_fit$spar))


##########################################
# Cross validation manually step-by-step:
##########################################

# do k-fold cross-validation with k = 10

# create a grid between 0 and 1
spar_vals <- seq(0, 1, 0.05)
spar_vals

k <- 10 # do 10-fold cross-validation
n <- length(xvals) # number of data points
fold_size <- floor(n / k) # size of each CV fold

# randomized indices to later make CV folds
sample_ids <- sample(n)

# initialize empty vector for loss values 
fold_losses <- rep(0, k)
cv_errors <- rep(0, length(spar_vals))

for (i in 1:length(spar_vals)) {
  # iterate over CV folds
  for (j in 1:k) {
    # compute this CV fold
    if (j < k) {
      hold_out_ids <- sample_ids[ (1 + (j - 1) * fold_size):(j * fold_size) ]
    } else {
      hold_out_ids <- sample_ids[(1 + (j - 1) * fold_size):n]
    }
    
    # train smoothing spline on all data EXCEPT this CV fold
    spline_fit_cv <- smooth.spline(xvals[-hold_out_ids], yvals[-hold_out_ids], spar = spar_vals[i])
    
    # evaluate error on held-out data
    ypred <- predict(spline_fit_cv, xvals[hold_out_ids])$y
    yobs <- yvals[hold_out_ids]
    squared_loss_fold <- mean((yobs - ypred) ^ 2)
    
    # record loss on this fold
    fold_losses[j] <- squared_loss_fold
  }
  
  cv_error <- mean(fold_losses)
  
  cv_errors[i] <- cv_error
}



plot(spar_vals, cv_errors, xlab = 'spar (our hyperparameter)', ylab = '10-fold CV Error', type = 'l')
best_spar_idx <- which.min(cv_errors)
points(spar_vals[best_spar_idx], cv_errors[best_spar_idx], pch = 'x', cex = 1.5)
title("CV Error vs spar for Smoothing Spline")
legend('top', c('Minimum CV Error'), pch = c('x'), cex = 1.5)
print(paste("our optimized spar value: ", spar_vals[best_spar_idx]))

# refit smoothing spline with best spar value
our_spline <- smooth.spline(xvals, yvals, spar = spar_vals[best_spar_idx])
ypred <- predict(our_spline, xvals)$y

plot(xvals, ypred, type = 'l', xlab = 'x', ylab = 'y', lwd = 2)

lines(xvals, m_fct(xvals), type = 'l', lty = 'dashed',col = "red", lwd = 2)

title('Our CV-Optimized Smoothing Spline and True Function')

legend("top", c("CV-Optimized Spline", "True Function"), lty = c('solid', 'dashed'), lwd = c(2, 2), col = c("black", "red"))
```


## Week 4
```R
## help(faithful)# standard  R  dataset
## ^^^^
str(faithful)
oldfaith <- faithful $ eruptions # eruption lengths (in minutes)

## histogram first:
hist(oldfaith)              # "counts"
hi.of <- hist(oldfaith, probab = TRUE)# -> area = 1
## -> now can compare with density:
lines(density(oldfaith), col = 2, lwd = 2)# with default bandwidth
rug(oldfaith)

# Sanity check for histogram:
sum(oldfaith<2.01)
55/(272*0.5)

## For demo:
library(tcltk) # needed by
library(TeachingDemos)
## Take a smaller random sub sample:
set.seed(11)
ofaith <- sample(oldfaith, 120)

run.hist.demo(ofaith)
## first play with 'number of bins',
## then stay with 20  and move 'Minimum':
## --> 2 modes but also 3 or 4 modes!

### static version:
oldpar <- par(mfrow = c(2,2)) ## 2 x 2 plotting
rng <- range(ofaith)
(breaks <- seq(rng[1] - .5,
               rng[2] + .5, by = 0.3))
hist(ofaith,main = "default in R")
hist(ofaith, breaks,      main = "hist(ofaith, breaks)")     ; rug(breaks, col=2)
hist(ofaith, breaks + .1, main = "hist(ofaith, breaks + 0.1)")
hist(ofaith, breaks + .2, main = "hist(ofaith, breaks + 0.2)")
## reset plot par's:
par(oldpar)


## now again :

plot(density(oldfaith), ylim=c(0,0.58))  # with default bandwidth
text(5.7,0.10, "bw=0.3348")
rug(oldfaith)

lines(density(oldfaith, bw="SJ-dpi"), col = 2) # with plug-in estimate for
text(5.12,0.5,"bw=0.165\n(SJ-dpi)",col=2) # bandwidth via estimating f , f''

plot(hi.of, freq= FALSE, add = TRUE, border = "gray")

bw.SJ(oldfaith, method = "dpi")  # plug-in estimate for
# bandwidth via estimating f and f''
## 0.1652728
bw.nrd0(oldfaith) # default, ad-hoc rule for estimating bandwidth
## 0.3347770

## Play around: ---- Density Demo ----
library(tcltk)
library(sfsmisc)
tkdensity(oldfaith)

## Showing *what* the kernel density is : use *small sample:
tkdensity(c(1:3, 5:8,10, 14))

###---- Plot for *lecture* (~= skript  "dens-faithful-5h" ) : -----------
xl.G <- c( .5, 6) ; yl.G <- c(0, 0.75)

op <- mult.fig(mfrow = c(2,3), marP = c(-2,-1,-.8,0))
for(h in .04* 1.5^(1:6)) {
  plot(density(faithful$eruptions, bw = h),
       xlim = xl.G, ylim = yl.G, cex = .75,
       xlab = "", ylab = "", main= "")
  ## xlab="faithful $ duration",ylab = "f(x)",
  mtext(paste("h = ", round(h,3)), cex = 1.2, line = -1.8, adj = 0.95)
  rug(faithful$eruptions, col="gray", lwd = 0.2)
}

###---- Plot for skript  "dens-faithful-5h" : -----------
op <- mult.fig(mfrow = c(1,5), marP = c(-2,-1.6,-1,-.8))
for(h in .04* 1.6^(1:5)) {
  plot(density(faithful$eruptions, bw = h),
       xlim = xl.G, ylim = yl.G, cex = .75, type = "h", col="gray",
       xlab = "", ylab = "", main= "")
  ## xlab="faithful $ duration",ylab = "f(x)",
  mtext(paste("h = ", round(h,3)), cex = 1.2, line = -1.8, adj = 0.95)
  rug(faithful$eruptions, col="gray20", lwd = 0.2)
}
```


# Leave-one-out CV

High variance typically, because the n training set are so similar to each other

-> leave-d-out CV (d>1) has higher bians than leave-one-out because we use training samples of sizes n-d instead of n, this causes some bias

# T-statistic

Measures how far (e.g. the mean) is from a hypothesized value in terms of standard error units.

<img width="784" height="177" alt="2025-08-07-104022_784x177_scrot" src="https://github.com/user-attachments/assets/eba7b708-0f1d-48c0-ac58-64b6ca827b4e" />


# Week 11

## Logisitic regression:

Logit function

<img width="1438" height="876" alt="2025-08-06-222121_1438x876_scrot" src="https://github.com/user-attachments/assets/6ab795f3-380c-4c27-b4fc-4db2e007d081" />


# Week 10

## LDA

# Week 9

# pivotal vs non-pivotal

Pivotal quantity : it's a function of the sample data whose distribution is known and does not depend on unknown parameters. 

Pivotal quantites are used to :

- Derive Confidence Intervals
- Perform hypothesis testing

<img width="753" height="652" alt="2025-08-07-102451_753x652_scrot" src="https://github.com/user-attachments/assets/b3dd71b7-390d-4108-ac6e-b93846db4590" />


## TODO : estimating DF by bootstrapping

## TODO : hypothesis testing using bootstrap

Repeatedly sample from the observed data to estimate the sampling distribution of a test statistic, which is then used to evaluate the null hypothesis.

We use bootstrap for Hypothesis Testing when the data violates parametric assumptions

## TODO : bayes classifier

## TODO : empirical risk minimization




# Studentization

The process of transforming a statistic by diving it by an estimate of it's standard error -> creating a studentized statistic

# One standard error rule

choose the simplest model whose error is no more than one standard error above the error of the best model.


<img width="1029" height="602" alt="2025-08-06-174933_1029x602_scrot" src="https://github.com/user-attachments/assets/cafea24c-1f9d-4745-a795-276008a1df7e" />

<img width="1111" height="581" alt="2025-08-06-175213_1111x581_scrot" src="https://github.com/user-attachments/assets/29d15d14-ef89-4dbe-bd41-9c2b99ebcbb8" />

<img width="1081" height="541" alt="2025-08-06-174828_1081x541_scrot" src="https://github.com/user-attachments/assets/e4c7fdf1-5624-4873-a4fc-a42449fb8e10" />


# Cubic splines

in addition to continuity we impose some derivatives to be continuous

<img width="682" height="567" alt="2025-08-06-172134_682x567_scrot" src="https://github.com/user-attachments/assets/223b7907-7d8b-4e74-9e46-c5b768877092" />

In R, *splines* can be used to fit cubic spline models.

cubic splines are also called regression cubic splines.

the number of knots can be fixed if we want a specific number of df or we can use cross-validation.

To fix the locations, we can use more knots in more wiggly regions or we can spread them out equally over the range of $x_i$

# MSE : mean square error

# MISE : Mean Integrated Squared Error

Used primarly when estimating a continuous function

<img width="340" height="79" alt="2025-08-06-170608_340x79_scrot" src="https://github.com/user-attachments/assets/ca522310-805a-43c0-8494-fa28345bf44f" />


# Degrees of freedom

df of an estimate = effective number of params

linear regression with p features (intercept included) df = p


# Supports in Kernel functions

The support of G(x) is the subset of its domain where G(x) != 0

<img width="1038" height="1235" alt="2025-08-06-140332_1038x1235_scrot" src="https://github.com/user-attachments/assets/d85c0f9d-3c07-42da-8827-1180f1c608b2" />


# Gradient tree boosting for squared loss regression

- The loss function is just the squared distance between ŷ and y, and the error is just the average square distance.
- The negated gradient values is proportional to the residual of OLS (Ordinary Least Square)

- At each iteration, GBM fits a small tree (small to avoid fitting models that are too complex) to the current residuals, and then we add this tree to the current model with a small weight gamma (use shrinkage to learn slowly=reduce gamma a bit at each iteration)
- We improve the model step by step, fitting the residuals.
- The size of trees M controls the degree of interactions -> use small tree in general because estimage high dim interactions hard -> curse of dimensionality
  
<img width="1114" height="619" alt="2025-08-06-105738_1114x619_scrot" src="https://github.com/user-attachments/assets/87ea700d-d55c-4706-9116-2db4dfa21d57" />

# Subsampling

At each iteration:

- use only a subset of the data to fit the gradient
- and / or use a subset of the features

# Adaboost : special case of GBM (Gradient Boosting Machine) with exponential loss function

- Exponential loss is more sensitive to outliers than the logistic loss

# Receiver Operating Characteristic (ROC) curve : True Positive rate vs False Positive rate

<img width="476" height="476" alt="2025-08-05-202131_476x476_scrot" src="https://github.com/user-attachments/assets/44fbcdd1-d1a9-4c11-a671-5420dc557389" />

Area Under the Curve (AUC) is defined as the area under the ROC curve

# Generative models

Estimate the joint P(Y,X) then use P(Y=1 | X=x) and Bayes rule to predict class 0 or 1

# Discriminative models

Estimate P(Y=1 | X=x) and Bayes to predict class 0 or 12

# We can also estimate a discriminant function that predicts the class 0 or 1 from x (does not need distribution models)

# Linear smoother protip : one can get an explicit form for the leave-one-out CV error

# T-distribution

Used to estimate population parameters when the sample size is small and the std dev of the population is unknown.
The shape of the distribution depends on the number of d.f. -> typically n -1 when the sample is of size n

For V (number of d.f.) > 30 the t-distribution is very close to the normal distribution

# Double bootstrapping

Used to improve the accuracy of bootstrap estimates, particualry for CI or hypothesis testing. We refine the bootstrap method by applying it twice

## First bootstrap

- Generate B samples by resampling with replacement from the original dataset
- For each boostrat sample, compute the statistic (e.g. mean)

## Second bootstrap

- For each of the B boostrap sample, perform another bootstrap by resampling (nested sampling)
- Compute the stat on these samples
- Use the second-evel statistics to estimate the variability or distribution of the first level bootstrap statistic


# Boostrap Confidence Intervals

## Percentile bootstrap CI

Generate many bootstrap samples by resampling with replacement, compute the statistic for each sample. Sort these statistics and take percentiles to form the CI

Assumes the bootstrap distribution approximates the true sampling distribution well. Works best when the statistic’s distribution is symmetric and not heavily biased.

## Basic (or Normal) Bootstrap CI

Compute the bootstrap statistics as in the percentile. Then we do the difference between the bootstrap statistic and the original sample statistic, and we use that to estimate the variability of the sampling distribution.

Assumes the bootstrap differences approximate the true sampling error distribution. Better than percentile for biased statistics.

## Studentized Boostrap CI

 Accounts for both bias and variance by standardizing the bootstrap statistics. For each bootstrap sample, compute the statistic and its estimated standard error (e.g., via a second level of bootstrapping or another method)

Assumes the t-statistics follow a distribution that approximates the true standardized sampling distribution. Requires reliable standard error estimates.

TODO : series 7

##################################################
### TASK e, f)
###################################################

##' Checks if a confidence interval contains the true parameter (separately
##' for the lower and the upper end)
##'
##' @param ci: Output of the function boot.ci which contains CIs
##' @param ty: Type of confidence interval
##' @param true.par: True parameter
##'
##' @return Vector with two elements where first one corresponds to the lower
##'         end and the second to the upper end of the confidence interval.
##'         If the CI is [CI_l, CI_u], the first element is 1 if theta < CI_l
##'         and 0 otherwise. The second element is 1 if theta > CI_u and 0
##'         otherwise.
check_ci <- function(ci, ty, true.par) {
  # Get confidence interval of type ty from object ci
  type <- c("basic"= "basic",
            "norm" = "normal",
            "perc" = "percent")[ty]
  ci. <- ci[[type]]
  k <- length(ci.) # need last two entries
  lower <- ci.[k-1]
  upper <- ci.[k  ]
  
  res <- if (true.par < lower) {
    c(1, 0)
  } else if (true.par > upper) {
    c(0, 1)
  } else {
    c(0, 0)
  }
  names(res) <- c("lower", "upper")
  # return result:
  res
}

##' Runs one simulation run, i.e. creates new data set, calculates bootstrap
##' CIs, and checks if true parameter is contained.
##'
##' @param n: Size of sample
##' @param true.par: True parameter
##' @param R: Number of bootstrap replicates
##' @param  : Type of bootstrap CIs, see function boot.ci
##'
##' @return A vector containing the result of the function check_ci for each
##'         of the confidence intervals
do_sim <- function(n, true.par, R = 500,
                   type = c("basic", "norm", "perc")) {
  # Generate the data
  x <- rData(n)
  # Construct the CIs for the IQR
  res.boot <- boot(data = x, statistic =, R = R)
  res.ci <- boot.ci(???, conf = 0.95, type = type)
  
  # Check if CIs contain true.par -> check_ci()
  res <- numeric(0)
  for (ty in type) {
    res <- c(res, check_ci(ci = res.ci, ty = ty, true.par = true.par))
    names(res)[(length(res) - 1):length(res)] <-
      paste(ty, c("lower", "upper"), sep = "_") # add names in the format
    # '<type>_lower' and '<type>_upper'
  }
  res
}

##########################
### Run simulation     ###
##########################
set.seed(22)
require("boot")
sample.size <- c(20, 40, 80, 160, 320)
n.sim <- 200
type <- c("basic", "norm", "perc")

# The object RES stores the results, i.e. each row corresponds
# to the non-coverage rate for the lower and upper ends of the
# confidence intervals, i.e. the percentage of times that theta < CI_l
# and the percentage of times that theta > CI_u, if the CI is
# denoted by (CI_l, CI_u). The last column of RES corresponds to
# the number of observations.
RES <- matrix(NA, nrow = length(sample.size), ncol = length(type) * 2 + 1)
colnames(RES) <- c(paste(rep(type, each = 2),
                         rep(c("lower", "upper"), times = length(type)),
                         sep = "_"),  "n")
for (j in 1:length(sample.size)) {
  n <- ???[j]
  cat("n = ", n, ":\n", sep="")
  # The object res.sim stores the results, i.e. each row corresponds
  # to the output of the function do_sim. This means that each row contains 0
  # and 1 encoding whether the true parameter was inside the CI or outside.
  # Also see the function check_ci.
  res.sim <- matrix(NA, nrow = n.sim, ncol = length(type) * 2)
  for (i in 1:???) { cat(".") ; if(i %% 50 == 0) cat(i,"\n") # <- show progress
    # Compute CIs and check if true.par is contained in them
    res.sim[i, ] <- do_sim(n = n, true.par = true.par, type = type)
  }
  # Compute the upper and lower non-coverage rate
  RES[j, ] <- c(apply(???), n)
}


# CGAL

```
K::Point_2 a;
K::Point_2 b;

CGAL::right_turn(a, b, c) // true if c makes a right turn relative to the directed line segment from a to b 
```

# AML

# Conditional entropy

$H(X \mid Y) = - \sum_{y \in \mathcal{Y}} \sum_{x \in \mathcal{X}} P(Y = y) P(X = x \mid Y = y) \log P(X = x \mid Y = y)$

$= - \sum_{x \in \mathcal{X}, y \in \mathcal{Y}} P(X = x, Y = y) \log \frac{P(X = x, Y = y)}{P(Y = y)},$

# Mutual information

$I(X; Y) := H(X) - H(X \mid Y)$

Measures the amount of information of X left after Y is revealed.

![2025-01-03-184515_818x62_scrot](https://github.com/user-attachments/assets/88b623b4-0a0e-4ad9-a714-ddecd518e070)

# Matrix calculus

We have

![2025-01-03-180910_189x110_scrot](https://github.com/user-attachments/assets/289a2781-670c-48c2-8fff-409102f5cc00)

# Recursion pro tip : if there are some overlap, just substract it !

![2024-12-16-151305_1733x397_scrot](https://github.com/user-attachments/assets/8532118d-b2fc-452b-bc04-ca0a15491991)

