Method
Stock return: S&P 500 price change plus 1/12 of the annual dividend. Shiller's dividends stop in June 2023, so the last value is carried forward after that (small effect on monthly returns).
Bond return: buy a 10-year par bond at last month's yield, sell it a month later at this month's yield (semi-annual coupons). Standard way to turn a yield series into returns.
Inflation regime: "high" when year-over-year CPI inflation is above 3%, otherwise "low".
Rolling correlation: 36-month window of monthly stock and bond returns.


Setup once:  pip install pandas numpy matplotlib

