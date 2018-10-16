# Squashing of Gaussian Process
What happens when we squash a Gaussian Process? The beauty of the GP lies in its calibrated confidence intervals. They make intuitive sense. The math work out beautifully in the regression case. Yet, in the classification case we need this ugly squashing function to output values in the [0, 1] interval. Our question for this project follows: what happens to the confidence intervals when we squash a Gaussian process (GP)?

# Were we to expect this behavior?
In the weeks preceding this project, my laptop broke. That left me with lots of time to think hard about the outcome before I could throw Monte Carlo at it. Here we will explore my reasoning. 

For this reasoning, we will use [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality). One will find many forms of Jensen's inequality online. The most consice version, imo, being "the mean of a convex function lies below the convex function of the mean". In math <img src="https://rawgit.com/RobRomijnders/squashing/master/svgs/01bcf9c4aa3a054c7555013715dd285d.svg?invert_in_darkmode" align=middle width=140.905875pt height=24.56552999999997pt/>.

Now let us compare the mean of the squashed function with the squashed function of the mean. Our squashing function is the logistic function, <img src="https://rawgit.com/RobRomijnders/squashing/master/svgs/bccf253d5eabc9c746afee392f973abd.svg?invert_in_darkmode" align=middle width=83.9091pt height=26.70657pt/>. On the domain with positive inputs, this function is concave. On the domain with negative inputs, this function is convex.

Therefore 

  1. For inputs larger than zero (i.o.w. for outputs larger than 0.5), the squashing function is concave. Hence the mean of squashed functions lies below the squashed mean of the functions.
  2. For inputs smaller than zero (i.o.w. for outputs smaller than 0.5), the squashing function is convex. Hence the mean of squashed functions lies above the squashed mean of the functions.
  3. Combining statement 1 and 2, we conclude that the mean of the squashed functions lies closer to the 0.5 value than the squashed mean function.

# Results
The following figure displays our results:
![squashing](www.google.com)

  * In all figures the dark brown indicates the posterior mean, the blue shading indicates the two sigma confidence interval.
  * Upper left: This diagram displays samples from a GP. No squashing yet.
  * Upper right: This diagram displays the squashed confidence intervals. In other means, we literally squash the confidence intervals from the upper left diagram.
  * Lower left: This diagram displays the estimated confidence intervals after squashing. In other words, we squash all samples and calculate the sample mean and variance to shade the confidence interval.
  * Lower right: Comparison figure for the confidence intervals. The blue lines relate to the upper right diagram. The red lines relate to the lower left diagram.  

Interesting observations

  * The lower right diagram confirms our reasoning in the previous section.
  * For the estimated confidence intervals, the possibility exists that the confidence interval reaches outside the [0, 1] interval.

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com