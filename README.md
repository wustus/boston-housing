# FFNN for Boston Housing Dataset

I've been looking through the internet and haven't seen implementations achieving a much better performance than this.
Adding some better feature scaling / -engineering will probably improve the results of this network. As is, I mostly scaled the features through trial and error and guesstimation.

The dataset is split randomly into training- and test-data and consistently achieves a better avg. difference than 2.5 ($2.500 USD), with some runs achieving values `< 2`.
If someone reads this and sees some critical flaw or could point me to a much better implemenation than this naivety, I'd be happy to know about it. 
Until then this would appear to be as far as the apple falls from the tree.

Later nerds.

## Results 

| Seed | Avg. Difference (10 Runs) |
| ---- | ------------------------- |
| 42 | 1.9713 |
| 43 | 2.1372 |
| 44 | 2.1544 |
| 45 | 2.1849 |
| 46 | 2.0652 |
| 47 | 2.1187 |
| 48 | 2.1354 |
| 49 | 2.2705 |                     
| 50 | 2.2288 |                     
| 51 | 2.2098 |
**AVG: 2.1471**
