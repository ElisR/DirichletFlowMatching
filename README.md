# Dirichlet Flow Matching

> This repository contains some example code for a talk about [Dirichlet Flow Matching](https://arxiv.org/abs/2402.05841), an approach to "discrete diffusion" over categorical sequences.

> [!NOTE]
>
> Unfortunately, GitHub's LaTeX parser is slightly limited, and will aggressively interpret subscript indicators as attempts to italicise text, so I will be using superscript more than I would like.


## 🏖️  Destination: Discrete Diffusion

As we all know, generative diffusion models have had considerable success.
In its original form, a diffusion model acts on continuous data, where the noising process is easy to interpret.

However, an extension of diffusion models to discrete data isn't quite so obvious.
The allure of discrete diffusion is obvious, however, even if we just restrict our imagination to (protein) language modelling tasks.
For example, one of the downsides of autoregressive models is that inference takes an amount of time proportional to sequence length.
Producing one word at a time is contrary to how we might perceive our own sentences as forming, _guided_ by intention, in our heads before being spoken, or how we would sketch the outline of a document before filling in the details.
As a final motivator: the already-successful masked manguage modelling (MLM) objective (where ~15% of tokens are masked) looks like a one-step denoising process, so what happens if we go further?

Here, we will review [Dirichlet Flow Matching](https://arxiv.org/abs/2402.05841) (DFM) as a new approach to this problem.

## ⚔️ Discrete Diffusion Alternatives

Here is a "TL;DR" the current approaches to making the idea of diffusion models (i.e. noising the data and learning the gradient of the probability distribution) work for discrete data.
We can come back to this at the end of the talk.

### Simplex Approaches like DDSM or BFN

### D3PM

### Continuous Diffusion in Latent Spaces

Cite CDCD approach.

## 🔀 Flow Matching ([Lipman et al. 2022](https://arxiv.org/abs/2210.02747))

Flow matching provides a training objective similar to those from diffusion models, but applies it to the (continuous) normalising flows of yesteryear.
At a high level, the neural network again learns the small steps needed to incrementally go from a pure noise distribution $q^0$ to the data distribution $p^{\text{data}}$. 
The way we get to a pure noise distribution differs compared to diffusion models, however.

On both ends one will have noisy samples $\mathbf{x}^0 \sim q$ and data samples $\mathbf{x}^1 \sim p^{\text{data}}$, and the aim is to regress a neural network against the vector field that transports $q(\mathbf{x}^0)$ to $p^{\text{data}}(\mathbf{x}^1)$.
At intermediate "times" $t \in [0, 1]$, we have will have a probability density path $p^t(\mathbf{x})$, transported by the vector field $u^t(\mathbf{x})$, that satisfies the above boundary conditions.
The _flow matching_ objective aims to minimise
$$\mathcal{L}^{\text{FM}}(\theta) = \mathbb{E}^{t\sim[0, 1], \mathbf{x} \sim p^t(\mathbf{x})} [ \| v^t(\mathbf{x};{\theta}) - u^t(\mathbf{x}) \|^2 ]$$

<!-- Draw a figure of a probability distribution and a vector field, like one from [FFJORD](https://arxiv.org/abs/1810.01367) -->

The first problem to overcome is that we don't have $p^{\text{data}}$, only _samples_ from the data distribution, so it's not clear what we should regress against.
For this reason, one would like to work with a conditional probability path conditioned on an individual data sample $p^t(\mathbf{x} | \mathbf{x}^1)$ that satisfies the boundary conditions $p^0(\mathbf{x} | \mathbf{x}^1) = q(\mathbf{x})$ and $p^1(\mathbf{x} | \mathbf{x}^1) \approx \delta(\mathbf{x} - \mathbf{x}^1)$ at $t=0$ and $t=1$, respectively.

One also assumes knowledge of a conditional vector field $u^t(\mathbf{x} | \mathbf{x}^1)$ that generates $p^t(\mathbf{x} | \mathbf{x}^1)$ through the transport equation[^1]:
$$\frac{\partial p^t}{\partial t} + \nabla \cdot (p^t u^t) = 0.$$

<!-- Draw a figure of a circle with some amount of stuff leaving it. -->

We will construct the target _marginal_ probability path $p^t(\mathbf{x})$ through a mixture of these simpler probability _conditional_ paths:
$$p^t(\mathbf{x}) = \int p^t(\mathbf{x} | \mathbf{x}^1) p^{\text{data}}(\mathbf{x}^1)\, d\mathbf{x}^1,$$
so that at $t=1$, $p^1(\mathbf{x}) \approx p^{\text{data}}(\mathbf{x}^1)$.

The next leap is that the marginal vector field that generates $p^t(\mathbf{x})$ can also be constructed[^2] in a similar way:
$$u^t(\mathbf{x}) = \int u^t(\mathbf{x} | \mathbf{x}^1) \frac{p^t(\mathbf{x} | \mathbf{x}^1) p^{\text{data}}(\mathbf{x}^1)}{p^t(\mathbf{x})} d\mathbf{x}^1.$$

Lipman et al. then show that minimising $\mathcal{L}^{\text{FM}}(\theta)$ is exactly the same as minimising $\mathcal{L}^{\text{CFM}}(\theta)$, where we regress against the conditional vector field instead:
$$\mathcal{L}^{\text{CFM}}(\theta) = \mathbb{E}^{t\sim[0, 1], \mathbf{x}^1 \sim p^{\text{data}}(\mathbf{x}), \mathbf{x} \sim p^t(\mathbf{x} | \mathbf{x}^1)} [ \| v^t(\mathbf{x};{\theta}) - u^t(\mathbf{x} | \mathbf{x}^1) \|^2 ]$$

<!-- Take a moment to look at this and spot how the structure of the expectation is similar to diffusion models. -->

This is great!
It lets us train a model to produce samples from $p^{\text{data}}(\mathbf{x})$ (by integrating $u^t(\mathbf{x})$ ), without ever needing access to the marginal probability paths or vector fields. 
We have everything we need from small batches of data.


[^1]: If the transport/continuity equation is unfamiliar, it's just differential equation that expresses that a certain quantity must be "conserved".
For example, given a snapshot of a fluid, whatever the density distribution and velocity field describing the motion of small fluid parcels, we know for certain that the mass comprising the fluid cannot be created or destroyed, which restricts how the density can evolve. 

[^2]: The proof of this is short and just shows the given vector field satisfying the transport equation by taking $\partial^t p^t(\mathbf{x}) = \int [\partial^t p^t(\mathbf{x} | \mathbf{x}^1)] p^{\text{data}}(\mathbf{x}^1) d\mathbf{x}^1$ and substituting in the transport equation for the conditional vector field, with some switching of integrals and derivatives.

## ⏪ Recap: A Simplex

A simplex $S_K$ in $K$ dimensions is defined by
$$S_K = \lbrace \mathbf{x} = (x^1, \ldots, x^K)^T \in \mathbb{R}^K | \mathbf{1}^T \mathbf{x} = 1, \mathbf{x} \geq 0 \rbrace.$$
This naturally emerges when talking about categorical distributions, where the concatenation of the probabilities of each class lies on a simplex.
When one-hot encoding a categorical variable as $\mathbf{x}$, the variable lies at the vertex of a $K$-dimensional simplex.

<!-- Draw a multinomial distribution with three options. Draw a simplex as a triangle in 3D. -->

## 💱 Modified Training Objective

DFM goes a step further and relax their $K$-class categorical distribution into continuous space by converting it to a mixture of point masses at the vertices of $S_K$, where $\mathbf{e}^i$ is the $i$ th one-hot vector:
$$p^{\text{data}}(\mathbf{x}) = \sum_{i=1}^K p^i \delta(\mathbf{x} - \mathbf{e}^i).$$
A few other approaches to discrete diffusion start with this to promote a discrete variable to a continuous space too.

During flow matching, this means that the transport _destination_ will be samples from the vertices, but at intermediate times the samples can lie anywhere on the simplex, like a superposition of different valid destinations.

One other modification by DFM is that instead of training their neural network to predict a vector field $v^t(\mathbf{x};{\theta})$, they train a denoising classifier $p(\mathbf{x}^1 | \mathbf{x}; \theta)$ by minimising a cross-entropy loss
$$\mathcal{L}^{\text{CFM}}(\theta) = \mathbb{E}^{t\sim[0, 1], \mathbf{x}^1 \sim p^{\text{data}}(\mathbf{x}), \mathbf{x} \sim p^t(\mathbf{x} | \mathbf{x}^1)} [ \log{p(\mathbf{x}^1 | \mathbf{x}; \theta)}],$$
which share the same minimiser.

This way, at any point in time, the model is trying to guess the correct label of a variable.
This may remind you of how diffusion model objectives are often recast to predicting fully denoised samples at all times.

At inference time, the vector field can be parameterised as
$$v^t(\mathbf{x};{\theta}) = \sum_{i=1}^K u^t(\mathbf{x} | \mathbf{x}^1 = \mathbf{e}^i;{\theta}) p(\mathbf{x} = \mathbf{e}^i | \mathbf{x}; \theta),$$
which is naturally restricted to tangent plane of the simplex.

## 🧠 _Linear_ Flow Matching vs _Dirichlet_ Flow Matching

A key ingredient was missing from our introduction to flow matching: how does one construct $p^t(\mathbf{x} | \mathbf{x}^1)$?
There is considerable design freedom here (much more than for standard diffusion models which rely on the special properties of Gaussians), so let's focus on how the Dirichlet Flow Matching paper does it.

Typically, flow matching papers produce the conditional vector field $u^t(\mathbf{x} | \mathbf{x}^1)$ by defining a conditional flow map (also dubbed an "interpolant") $\psi^t(\mathbf{x}^0 | \mathbf{x}^1)$ that directly transports $\mathbf{x}^0 \sim q$ to $\mathbf{x} = \psi^t(\mathbf{x}^0 | \mathbf{x}^1) \sim p^t(\mathbf{x} | \mathbf{x}^1)$ at time $t$, for which the vector field is
$$u^t(\mathbf{x} | \mathbf{x}^1) = \frac{d}{dt} \psi^t(\mathbf{x}^0 | \mathbf{x}^1),$$
with the boundary conditions $\psi^0(\mathbf{x}^0 | \mathbf{x}^1) = \mathbf{x}^0$ and $\psi^1(\mathbf{x}^0 | \mathbf{x}^1) = \mathbf{x}^1$.

The simplest interpolant is just the linear flow map
$$\psi^t(\mathbf{x}^0 | \mathbf{x}^1) = (1 - t) \mathbf{x}^0 + t \mathbf{x}^1 \implies u^t(\mathbf{x} | \mathbf{x}^1)= \mathbf{x}^1 - \mathbf{x}^0$$
Note that all points remain on the simplex, and that points move in straight paths.

## 📊 Results
