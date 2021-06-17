# Intuition and Mathematical Proof of The Bias-Variance Trade-Off

Assessing the model’s accuracy is one of the most crucial things while determining the model that’s best suited for a particular dataset and a problem. By which I mean we need to determine how close the predictions made are to the actual observed values.

### Terminology

In general, for a particular problem, for an observed quantitative response $$ Y $$ and $$ p $$ different predictors/features, we define the relationship between $$ Y $$ and Vector $$ X $$ :

$$ Y = f(X) + \epsilon $$

Accordingly, regarding estimation, $$ \hat f $$ represents our estimation for $$ f $$ that on a set of Predictors $$ X $$ yields $$ \hat Y $$ that represents the resulting prediction for $$ Y $$.

$$ \hat Y = \hat f(X) $$

$$ f(x) $$ is some fixed but unknown function of the Vector $$ X $$. Generally, $$ Y $$ has a non-deterministic relationship. It is also a function of $$ \epsilon $$ which cannot be predicted using $$ X $$ and, in the term, is the irreducible error. Think about $$ \epsilon $$ as some low impact or unmeasurable feature, which is essentially a random error term. It’s synonymous with Gaussian noise, henceforth has zero mean. That is,

$$ {\mathbb{E}} [\epsilon] = 0, var(\epsilon) = \sigma_\epsilon^2 = {\mathbb{E}} [\epsilon^2] - {\mathbb{E}} [\epsilon]^2 = {\mathbb{E}} [\epsilon^2] $$

In regression settings, we examine the loss using the **Squared Loss function** for one of the instances: the square of the difference between the actual observed value and the Predicted value.

Averaging the Squared Loss per instance over the whole dataset gives us the **Mean Squared Error** (MSE). That is,

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat f(x_i))^2 = {\mathbb{E}} [(y - \hat f(x))^2] $$

**Bias** refers to the error that is inherently introduced by simplifying the assumptions used in the model for better interpretability at the cost of flexibility for the otherwise highly complicated real-life problem. It’s defined as the difference between the Expected value of Prediction $$ \hat f(x) $$ to the Actual Observed value $$ f(x) $$ for a given unseen test data point. That is,

$$ bias[\hat f(x)] = {\mathbb{E}} [\hat f(x)] - f(x) $$

**Variance** represents the quantitative amount by which a random variable differs from its expected value. When we change the training dataset for a highly flexible model, then because of overfitting, the estimated Prediction function $$ \hat f(x) $$ changes drastically. Whereas for learning algorithms like linear regression, logistics regression, the variance is quite low. And also, $$ \hat f(x) $$ is essentially estimated over a particular training dataset. Hence different training data would result in different $$ \hat f(x) $$. Over the range of different training datasets, the Expected squared deviation of $$ \hat f(x) $$ from its expected value $$ {\mathbb{E}} [\hat f(x)] $$ is the Variance. That is,

$$ var(\hat f(x)) = {\mathbb{E}} [(\hat f(x) - {\mathbb{E}} [\hat f(x)])^2] $$

This brings us to the infamous **Bias-Variance trade-off** resulting from two competing properties of Statistical Learning methods. Accordingly, as evident by the equation to the Expected Test MSE for a given data point $$ x $$, comprises of the variance of $$ \hat f(x) $$, the squared bias of $$ \hat f(x) $$, and the variance of the error term $$ \epsilon $$. That is,

$$ {\mathbb{E}} [(y - \hat f(x))^2] = var(\hat f(x)) + [bias[\hat f(x)]]^2 + var(\epsilon) $$

This equation tells that to minimize the Expected Test MSE, and we need a statistical method that simultaneously achieves *low variance* and *low bias*. Also, as the variance is an inherently nonnegative quantity, and squared bias is also nonnegative. Hence Expected Test MSE would always stay above $$ Var(\epsilon) $$.

### Mathematical Proof

Expected Prediction Error, $$ {\mathbb{E}} [(y - \hat f(x))^2] $$ of a regression fit $$ \hat f(x) $$ at input data point $$ X = x $$ using MSE is —

$$ 
    \begin{equation} \label{eq1}
        \begin{split}
            MSE & = {\mathbb{E}} [(y - \hat f(x))^2 | X = x] \\
                & = {\mathbb{E}} [(y - \hat f(x))^2] = {\mathbb{E}} [(f(x) + \epsilon - \hat f(x))^2] \\
                & = {\mathbb{E} [(f(x) - \hat f(x))^2] + {\mathbb{E}} [\epsilon^2]} + 2{\mathbb{E}} [(f(x) - \hat f(x)) \epsilon] \\
                & = {\mathbb{E} [(f(x) - \hat f(x))^2] + {\mathbb{E}} [\epsilon^2]} + 2{\mathbb{E}} [(f(x) - \hat f(x))] {\mathbb{E}} [\epsilon] \\
                & = {\mathbb{E}} [(f(x) - \hat f(x))^2] + \sigma_\epsilon^2 \\
        \end{split}
    \end{equation}
$$

Taken from Alecos Papadopoulos [StackExchange](https://stats.stackexchange.com/questions/366220/predictor-and-error-are-independent)

Recall that $$ \hat f(x) $$ is the predictor function we have constructed based on m data points. $$ (x^{(1)}, y^{(1)}), …, (x^{(m)}, y^{(m)}) $$.

On the other hand, $$ Y $$ is the prediction we are making on a new data point using the model constructed on the m data points above.
$$ (x^{(m+1)}, y^{(m+1)}) $$.

The assumptions that we have for the new data point is that —
- It was **not** used while constructing $$ \hat f(x) $$
- It is independent of all the previous training data points
- And it's independent of $$ \epsilon^{m+1} $$.

Hence we can conclude that $$ \hat f(x) $$ and $$ \epsilon $$ are independent random variables.

Further expanding $$ {\mathbb{E}} [(f(x) - \hat f(x))^2] $$ term.

$$ 
    \begin{equation} \label{eq2}
        \begin{split}
            {\mathbb{E}} [(f(x) - \hat f(x))^2] 
            & = {\mathbb{E}} [(f(x) + {\mathbb{E}} [\hat f(x)] - {\mathbb{E}} [\hat f(x)] - \hat f(x))^2] \\
            & = {\mathbb{E}} [({\mathbb{E}} [\hat f(x)] - f(x))^2] + {\mathbb{E}} [(\hat f(x) - {\mathbb{E}} [\hat f(x)])^2] - 2 {\mathbb{E}} [(f(x) - {\mathbb{E}} [\hat f(x)])] (\hat f(x) - {\mathbb{E}} [\hat f(x)]) \\
            & = \underbrace{ 
                ({\mathbb{E}} [\hat f(x)] - f(x))^2}_\text{bias[f^(x)]} + \underbrace{
                {\mathbb{E}} [(\hat f(x) - {\mathbb{E}} [\hat f(x)])]^2}_\text{variance[f^(x)]} - 2 {\mathbb{E}} [(f(x) - {\mathbb{E}} [\hat f(x)])] ({\mathbb{E}} [\hat f(x)] - {\mathbb{E}} [\hat f(x)]) \\
            & = bias[\hat f(x)] + variance[\hat f(x)] \\
        \end{split}
    \end{equation}
$$

$$ {\mathbb{E}}[\hat f(x)] $$ and $$ f(x) $$ both are constant. Hence there difference that is, $$ Bias {\mathbb{E}} [\hat f(x)] − f(x) $$ is just a constant. Therefore, applying expectation to squared bias, $$ ({\mathbb{E}} [\hat f(x)] − f(x))^2 $$ does not have any effect.

Hence, $$ {\mathbb{E}} [({\mathbb{E}} [\hat f(x)] - f(x))^2] = ({\mathbb{E}} [\hat f(x)] - f(x))^2 $$

Therefore combining the above three equations we get,

$$ {\mathbb{E}}[(y - \hat f(x))^2] = var(\hat f(x)) + [bias(\hat f(x))]^2 + var(\epsilon) $$

This is actually an interesting equation, and to find an appropriate model for our dataset, we’ve to strike a balance between those two competing properties — *Bias* and *Variance*.

### References

- [StackExchange](https://stats.stackexchange.com/questions/204115/understanding-bias-variance-tradeoff-derivation/354284#354284)

- [Towards Data Science](https://towardsdatascience.com/the-bias-variance-tradeoff-8818f41e39e9)

- [StatLearning](https://www.statlearning.com/)

<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>