# Stock_classifier

### Problem Statement

<span style="font-family:Papyrus"> 
Is it possible to find non-random patterns in a given stock with significant accuracy in order to time the market better than pure randomness?
</span>

#### Introduction

<span style="font-family:Papyrus"> 
Since the beginning of the first publicly traded company in the 1600s to present day Wall Street, investors and speculators alike have been enticed by the notion of growing their wealth beyond the returns of the overall market. This notion is considered contentious among many market economist and analysist. Although, it has been demonstrated that a fraction of investors has beat the overall market through timing and/or fundamental analysis, it is widely accepted that these techniques are not sustainable in the long-term or short-term due to the Efficient Market Hypothesis. This hypothesis states that all important information regarding an asset is already priced in to that stock at any given time. This basically means that no form of technical, or fundamental analysis will give you the edge on buying a stock below fair value. Currently, there are many large financial institutions and hedge funds that invest a lot of their resources on the analysis of stock trends using various machine learning techniques. Most notably, James Harris Simons, a mathematician that has been said to have “cracked Wall Street”, has stated he uses many mathematical techniques to find non-random patterns in the noise of the market through various machine learning techniques and signal analysis. James Simons is currently worth around 20 billion dollars. Although his algorithms are proprietary in nature, we have a lead on the types of techniques he and his colleague’s use.

</span>

#### Type of Data Science Learning

<span style="font-family:Papyrus"> 
This project will incorporate several different types of classification models using supervised learning. My project will explore a few models such as random forests, Xgboost, SVMc, KNNs and DNNs to classify optimal days to buy and sell a stock. The techniques used to label my target data will come from looking into the future at various time intervals such as 1 day, 3 day and up to 20 day highs.

</span>

#### Why This Approach?

<span style="font-family:Papyrus"> 
As you can see from the distributions below there is far more opportunity to maximize your returns from using the highs on the various days as opposed to only using the open to close or open to open prices as buy and sell opportunities. The one major issue with using the highs instead of the closes, would be that we don't really know when those prices occurred. It is much easier to pin down that price at the beginnning or end of the trading day when using the open to open trading targets. However, if we split our targets into quartiles then we now have a little more information about those prices. That is to say we know generally the probability which those prices occur. For example, everything above quartile 1 (Q1) will likely occur 75 percent of the time. The major assumption we making here is that our training set distributions are similar to our test set distributions. 
</span>

<p align="center">
  <h3>Compare Epsilon and Convergance </>
  <img src="../images/distribution.png" )
</p>

## The Problem

<span style="font-family:Papyrus"> The problem I am exploring is with three different bandits initialized with normal distributions. Later I will explore a more complicated version of the problem which uses other distributions (e.g. Exponential, Poisson etc.). Below is the order in which I will Explore this problem.
</span>

- [x] Introduction
- [ ] Explore large and small values of Epsilon
- [ ] Explore normal distributions with closer means
- [ ] Conclusion of my findings


### Explore Epsilon Values
- [x] Explore large and small values of Epsilon

```python
c_25 = run_experiment(1.0,2.0,3.0, 0.5, 100000)
c_25 = run_experiment(1.0,2.0,3.0, 0.25, 100000)
c_1 = run_experiment(1.0,2.0,3.0, 0.1, 100000)
c_05 = run_experiment(1.0,2.0,3.0, 0.05, 100000)
c_01 = run_experiment(1.0,2.0,3.0, 0.01, 100000)
c_001 = run_experiment(1.0,2.0,3.0, 0.001, 100000)

```
<p align="center">
  <h3>Compare Epsilon and Convergance </>
  <img src="graphs_bandit_1.png" )
</p>

<div>
  
| Epsilon       | Long Term Payout   | Start of Convergance |
| ------------- |:------------------:| --------------------:|
| 0.001         |   2.98424162       |    ≈ 2000            |
| 0.01          |   2.98545949       |    ≈ 200             |
| 0.05          |   2.94623982       |    ≈ 30              |
| 0.1           |   2.89199996       |    ≈ 40              |
| 0.25          |   2.74934096       |    NA                |
| 0.50          |   2.49796348       |    NA                |

</div>


### Small vs. Big Epsilon
<span style="font-family:Papyrus"> We can see from the graphs above that as you decrease epsilon you increase your long term payout except once you get to eps = .001. Here, the payout decreases slightly. This could be due  to the psuedo-random generation of values. Either way, we can see it won't benifit that much. Additionally, we don't want to be overconfident in the case we have really close means or weird distributions with high variance, which ultumitely means closer means. Another thing to note here is that higher values of epsilon (i.e.eps = .25 and greater) don't actually converge on the correct bandit. This is due to over-exploration. When an agent indefinteley chooses the other two bandit 25 percent of the time or greater than your long term payout be attracted to a lower payout since it isn't always picking the higher mean payout.
</span>

#### Closer Means 
- [x] Explore normal distributions with closer means

<span style="font-family:Papyrus"> As we decreased the differences between the means notice that we still are able to converge on the highest bandit. This seems to be due to the fact that we are iterating 100,000 times to collect enough of a sample size for each of our epsilon values. Let's explore this further with a few calculations and estimates. See table below.
</span>

| Epsilon | Prob 100000 | Est of 1.05 |Prob 10000| Est of 1.05 |
| ------- |:-----------:| -----------:|:--------:|:-----------:|
| 0.001   |      33.33  |     .88     |   3.33   |    .27      |
| 0.005   |      166.67 |    1.21     |  16.6    |   1.13      |
| 0.01    |     333.33  |    1.07     |  33.3    |   .96       |
| 0.05    |    1666.67  |    1.06     |  166.6   |   1.16      |
| 0.1     |    3333.33  |    1.06     |  333.3   |   1.08      |
| 0.15    |      5000   |    1.04     |  500     |   1.01      |
#### Variance
<span style="font-family:Papyrus"> From the above chart we can see larger sample sizes such as 100,000 explore more of the other bandits even with low epsilon values. Although, the optimal epsilon seemed to be around .01 to .1. These looked to be the best estimates of the normal distribution with mean 1.05. Thus, these look to have lowest variance with there estimates. Probability given N (100,000 and 10,000) iterations were calculated by taking epsilon times N divided by three, which is the probability of exploring one of other bandits given its respective epsilon. In thoery the larger the sample size the better approximation to the mean we get. However, the cost to sample larger numbers could be costly, so we must minimize that as much as possible.
</span>


## Conclusion
- [x] Conclusion: What I learned.

<span style="font-family:Papyrus"> After doing many experiments I was able to learn a few things about this problem. First, epsilon in this case is the probability that were explore the bandits that look to have a smaller sample mean. Thus, we call this the epsilon-greedy problem. Alternativly, one minus epsilon equals the probability that the agent will eploit the perceived largest sample mean. Next, I noticed higher eploration of all the bandits produces lower variance of mean estimates. Consequently, there are two ways to collect larger samples of all the bandits. One is by increasing your epsilon values and two is by increasing the number of iterations you play with the bandits. The downside to increasing epsilon too much was it would never converge on the highest bandit. The downside to larger iterations of play is it may be costly in real-life practice. Thus, the optimal range for epsilon was from .01 to .1. Although, it seemed reasonable to use a psuedo-decay epsilon to maximize effectiveness of minimizing variance of sample means and maximizing the highest returns. I am still working on the last problem, but see code for more info related to this problem.  
</span>


