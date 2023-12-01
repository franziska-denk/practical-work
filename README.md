# Practical Work (WS 2023)

#### Author: Franziska Denk (k11904292)


This work is implementing the experiments from Ilyas et al. [[1]](#1) within the scope of the practical work in the master studies *Artificial Intelligence* at *JKU*.

## How to rerun the experiments?
1. Install the requirements
```conda create --name practical_work --file requirements.txt```

2. Run the experiments using `scripts/experiment.bat`. Hyperparameters can be adjusted in there, for a list of available hyperparameters have a look at `src/train.py`.
More information is in `01-train.ipynb`.

3. Evaluate the models from the experiments in the second step using `02-evaluate.ipynb` or via the command-line using `src/test.py`.

## Findings
The results from Ilyas et al. (2019) [[1]](#1) can be replicated in a way that the relative accuracy of the experiments to each other is similar.
This confirms their main findings.

The three main differences are:
* The PGD attack in this work doesn't achieve such low adversarial accuracy for the standard model, as in the work of Ilyas et al. For a PGD attack with $\epsilon=0.5$, steps $=7$ and $\alpha=0.1$, we achieve adversarial accuracy of ~ $15$ %, while Ilyas et al. report $0$ %.
* For creating $\mathcal{D} _{rand}$ and $\mathcal{D} _{det}$ with PGD, $\epsilon$ needed to be larger in this work than in their setting. We used $\epsilon=5$ while they had $\epsilon=0.5$
* $\mathcal{D} _{rand}$ & $\mathcal{D} _{det}$ that are created with a robust model (as control dataset), have a much higher accuracy in this work than in the results of Ilyas et al.


## References
<a id="1">[1]</a> 
Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran and Aleksander Madry:
“Adversarial Examples Are Not Bugs, They Are Features”.
In: Advances in Neural Information Processing Systems (NeurIPS). vol. 32. Vancouver, Canada: Curran Associates,
Inc., 2019, pp. 125–136.

## Further readings
https://distill.pub/2019/advex-bugs-discussion/response-5/
