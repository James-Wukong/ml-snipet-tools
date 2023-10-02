from scipy.stats import t
from math import sqrt

# get critical value range for 95% confidence
# with a sample size of 25
n = 12
# cdf() function calculates the probability for a given normal distribution value, 
# while the . ppf() function calculates the normal distribution value for which 
# a given probability is the required value, in short, ppf() <=> cdf()
lower = t.ppf(.025, df=n-1)
upper = t.ppf(.975, df=n-1)
# same with above
# lower_cv = t(n-1).ppf(.025)
# upper_cv = t(n-1).ppf(.975)

# if the test value falls outside of the critical range of 95% confidence,
# we accept that our correlation was NOT by chance.
print(lower, upper)
# -2.063898561628021 2.0638985616280205

# sample size
n = 10
lower_cv = t(n-1).ppf(.025)
upper_cv = t(n-1).ppf(.975)
# correlation coefficient
# derived from data https://bit.ly/2KF29Bd
r = 0.957586
# Perform the test
test_value = r / sqrt((1-r**2) / (n-2))
print("TEST VALUE: {}".format(test_value))
print("CRITICAL RANGE: {}, {}".format(lower_cv, upper_cv))
if test_value < lower_cv or test_value > upper_cv:
    print("CORRELATION PROVEN, REJECT H0")
else:
    print("CORRELATION NOT PROVEN, FAILED TO REJECT H0 ")
# Calculate p-value
if test_value > 0:
    p_value = 1.0 - t(n-1).cdf(test_value)
else:
    p_value = t(n-1).cdf(test_value)
# Two-tailed, so multiply by 2
p_value = p_value * 2
print("P-VALUE: {}".format(p_value))

# The test value above is approximately 9.39956, which is definitely outside
# the range of (-2.262, 2.262) so we can reject the null hypothesis and say our
# correlation is real.