# Evaluation Tools  
previous points: $X_{prev} = \{[[(x^{prev}_{1, 1}, y^{prev}_{1, 1}), (x^{prev}_{1, 2}, y^{prev}_{1, 2}), ..., (x^{prev}_{1, m}, y^{prev}_{1, m})], ...,\\ [(x^{prev}_{k, 1}, y^{prev}_{k, 1}), (x^{prev}_{k, 2}, y^{prev}_{k, 2}), ..., (x^{prev}_{k, m}, y^{prev}_{k, m})]]\}$   

shape of $X_{prev}$ should be $(k, m, 2)$  .
$k$ means the batch size.
$m$ means the numbers of the points we input into the model.

groundtruth: $X =\{(x_1, y_1), (x_2, y_2),...,(x_n,y_n)\}$    

predicted: $Y =\{(\tilde{x_1}, \tilde{y_1}), (\tilde{x_2},\tilde{y_2}),...,(\tilde{x_n},\tilde{y_n})\}$      

$n$: Numbers of predicted points, namely numbers of $Y$.

The sum means we sum up error over every points.
## ADE(Average Distance Error)

$ADE = \frac{1}{n} \sum_n \sqrt{(X-Y)^2}$

## MSE(Mean Square Error)

$MSE =\sqrt{\frac{\sum_n (X-Y)^2}{n}}$


## AAE(Average Angle Error)
### cal_dist (Calculate Distance between two points)
We set $I, J$ as the input of the function $cal\_dist$, which means we have a function $cal\_dist(I, J)$.
$cal\_dist(I, J) = \sqrt{ (I - J)^2}$

### Calculate Distance a, b, c
$start=X_{prev}[,m], end = X$
$a = cal\_dist(end, start)$
$b = cal\_dist(start, Y)$, as we know $Y$ is the predicted points.
$c = cal\_dist(end, Y)$
$cos_c = \frac{a^2 + b^2 - c^2}{2 \times a \times b}$, notely we should ensure $a \times b \neq 0$.
$AE = \arccos(cos_c)$ arccos represents arc cosine. And $cos_c \in [-1, 1]$ which means we should clamp $cos_c$ into $(-1, 1)$ for prevent the float overflow.

### Calculate AAE
$AAE = \frac{\sum_n AE}{n}$ 




## APT (Average Predicted Time)
The model predicted time. We can use this signal to know whether the model predicted the enough distance we need.