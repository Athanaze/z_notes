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

