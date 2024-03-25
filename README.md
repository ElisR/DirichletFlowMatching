# Dirichlet Flow Matching

> This repository contains some example code for a talk about [Dirichlet Flow Matching](https://arxiv.org/abs/2402.05841), an approach to "discrete diffusion" over categorical sequences.

> [!NOTE]
>
> Unfortunately, GitHub's LaTeX parser is slightly limited, and will aggressively interpret subscript indicators as attempts to italicise text, so I will be using superscript more than I would like.

## Recap: A Simplex

A simplex $S_K$ in $K$ dimensions is defined by
$$S_K = \lbrace \mathbf{x} = (x^1, \ldots, x^K)^T \in \mathbb{R}^K | \mathbf{1}^T \mathbf{x} = 1, \mathbf{x} \geq 0 \rbrace.$$
This naturally emerges when talking about categorical distributions, where the concatenation of the probabilities of each class lies on a simplex.
When one-hot encoding a categorical variable as $\mathbf{x}$, the variable lies at the vertex of a $K$-dimensional simplex.

Many approaches to discrete diffusion start with this to promote a discrete variable to a continuous space, including Dirichlet Flow Matching, which we will review.

## Discrete Diffusion Alternatives

Here is a "TL;DR" the current approaches to making the idea of diffusion models (i.e. noising the data and learning the gradient of the probability distribution) work for discrete data.
We can come back to this at the end of the talk.

### Simplex Approaches like DDSM or BFN

### D3PM

### Continuous Diffusion in Latent Spaces

Cite CDCD approach.


## Flow Matching ([Lipman et al. 2022](https://arxiv.org/abs/2210.02747))

Flow matching provides a training objective similar to those from diffusion models, but applies it to the (continuouos) normalising flows of yesteryear.
At a high level, the neural network again learns the small steps needed to incrementally go from a pure noise distribution $q^0$ to the data distribution $p^{\text{data}}$. 
The way we get to a pure noise distribution differs compared to diffusion models, however.

The starting point is a noisy sample $\mathbf{x}^0 \sim q^0$ and a data sample $\mathbf{x}^1 \sim p^{\text{data}}$, and the aim is to regress a neural network against the vector field that transports $q(\mathbf{x}^0)$ to $p^{\text{data}}(\mathbf{x}^1)$.
At intermediate "times" $t \in [0, 1]$, we have will have a probability density path $p^t(\mathbf{x})$, generated by $u^t(\mathbf{x})$, that satisfies the above boundary conditions.
The _flow matching_ objective aims to minimise
$$\mathcal{L}^{\text{FM}}(\theta) = \mathbb{E}^{t\sim[0, 1], \mathbf{x} \sim p^t(\mathbf{x})} [ \| v^t(\mathbf{x};{\theta}) - u^t(\mathbf{x}) \|^2 ]$$

The first problem to overcome is that we don't have $p^{\text{data}}$, only _samples_ from the data distribution, so it's not clear what we should regress against.
For this reason, one works with a conditional probability path conditioned on an individual data sample $p^t(\mathbf{x} | \mathbf{x}^1)$ that satisfies the boundary conditions $p^0(\mathbf{x} | \mathbf{x}^1) = q(\mathbf{x})$ and $p^1(\mathbf{x} | \mathbf{x}^1) \approx \delta(\mathbf{x} - \mathbf{x}^1)$.

One also assumes knowledge of a conditional vector field $u^t(\mathbf{x} | \mathbf{x}^1)$ that generates $p^t(\mathbf{x} | \mathbf{x}^1)$ through the transport equation[^1]:
$$\frac{\partial p^t}{\partial t} + \nabla \cdot (p^t u^t) = 0.$$

We will construct the target _marginal_ probability path $p^t(\mathbf{x})$ through a mixture of these simpler probability _conditional_ paths:
$$p^t(\mathbf{x}) = \int p^t(\mathbf{x} | \mathbf{x}^1) p^{\text{data}}(\mathbf{x}^1)\, d\mathbf{x}^1,$$
so that at $t=1$, $p^1(\mathbf{x}) \approx p^{\text{data}}(\mathbf{x}^1)$.

The next leap is that the marginal vector field that generates $p^t(\mathbf{x})$ can also be constructed[^2] in a similar way:
$$u^t(\mathbf{x}) = \int u^t(\mathbf{x} | \mathbf{x}^1) \frac{p^t(\mathbf{x} | \mathbf{x}^1) p^{\text{data}}(\mathbf{x}^1)}{p^t(\mathbf{x})}\, d\mathbf{x}^1.$$

Lipman et al. then show that minimising $\mathcal{L}^{\text{FM}}(\theta)$ is exactly the same as minimising $\mathcal{L}^{\text{CFM}}(\theta)$, where we regress against the conditional vector field instead:
$$\mathcal{L}^{\text{CFM}}(\theta) = \mathbb{E}^{t\sim[0, 1], \mathbf{x}^1 \sim p^{\text{data}}(\mathbf{x}), \mathbf{x} \sim p^t(\mathbf{x} | \mathbf{x}^1)} [ \| v^t(\mathbf{x};{\theta}) - u^t(\mathbf{x} | \mathbf{x}^1) \|^2 ]$$

This is great!
It lets us train a model to produce samples from $p^{\text{data}}(\mathbf{x})$ (by integrating $u^t(\mathbf{x})$), without ever needing access to the marginal probability paths or vector fields. 
We have everything we need from small batches of data.

[^1]: If the transport/continuity equation is unfamiliar, it's just differential equation that expresses that a certain quantity must be "conserved".
For example, given a snapshot of a fluid, whatever the density distribution and velocity field describing the motion of small fluid parcels, we know for certain that the mass comprising the fluid cannot be created or destroyed, which restricts how the density can evolve. 

[^2]: The proof of this is short and just shows the given vector field satisfying the transport equation by taking $\partial^t p^t(\mathbf{x}) = \int [\partial^t p^t(\mathbf{x} | \mathbf{x}^1)] p^{\text{data}}(\mathbf{x}^1)\, d\mathbf{x}^1$ and substituting in the transport equation for the conditional vector field, with some switching of integrals and derivatives.

## _Linear_ Flow Matching vs _Dirichlet_ Flow Matching



## Results
