Method
Stock return: S&P 500 price change plus 1/12 of the annual dividend. Shiller's dividends stop in June 2023, so the last value is carried forward after that (small effect on monthly returns).
Bond return: buy a 10-year par bond at last month's yield, sell it a month later at this month's yield (semi-annual coupons). Standard way to turn a yield series into returns.
Inflation regime: "high" when year-over-year CPI inflation is above 3%, otherwise "low".
Rolling correlation: 36-month window of monthly stock and bond returns.


Setup once:  pip install pandas numpy matplotlib



# Stock-bond correlation across inflation regimes 

## Key Findings
1. **Bond hedge stocks when inflation is low, but not when it is high.** From 1953 to 2026, the correlation between monthly S&P 500 and 10-year Treasury returns was +0.23 in high-inflation months and −0.19 in low-inflation months. The difference is statistically significant (Fisher z-test: z = 6.3, p < 0.001).

2. **The correlation rises steadily with inflation.** It goes from −0.34 when inflation is under 2% to about +0.25 to +0.29 when inflation is above 4%. The 95% confidence intervals for the lowest and highest buckets do not overlap.

![Correlation by inflation]
(charts/correlation_by_inflation.png)



## Assumptions:
- High Inflation is defined with periods of inflation rate higher than 3%, interested could adjust the standard by changing the predefined HIGH_INFL value