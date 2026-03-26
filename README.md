
# anneal

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

Functional optimizers for the [anvil](https://github.com/r-xla/anvil) ecosystem.
Optimizers are pure functions -- all state flows in and out explicitly, with no mutation.
This makes them compatible with JIT compilation and `nv_while()` loops.

## Installation

{anneal} can be installed from GitHub:

```r
pak::pak("r-xla/anneal")
```

## Quick Start

Create an optimizer, initialize its state, and step through a training loop:

```r
library(anneal)

# Parameters and gradients (named lists)
params <- list(w = matrix(rnorm(6), 2, 3), b = rnorm(3))

opt   <- optim_sgd(lr = 0.01, momentum = 0.9)
state <- opt_init(opt, params)

for (i in seq_len(100)) {
  grads  <- list(w = ..., b = ...)         # from anvil::value_and_gradient()
  result <- opt_step(opt, params, grads, state)
  params <- result$params
  state  <- result$state
}
```

## Design

Optimizers follow a functional, Optax-inspired design:

- **`optim_sgd(lr, momentum, ...)`** -- creates an optimizer (just hyperparameters, no state).
- **`opt_init(opt, params)`** -- initializes optimizer state (momentum buffers, step counts, etc.).
- **`opt_step(opt, params, grads, state)`** -- returns updated `params` and `state`. No mutation.

Because there is no mutable state, the optimizer can be used inside compiled training loops:

```r
library(anvil)

train <- jit(function(params, state, x, y) {
  nv_while(
    list(params = params, state = state, i = nv_scalar(0L)),
    \(params, state, i) i < 100L,
    \(params, state, i) {
      result <- loss_and_grad(params$w, params$b, x, y)
      out    <- opt_step(opt, params, result$gradients, state)
      list(params = out$params, state = out$state, i = i + 1L)
    }
  )
})
```

## Available Optimizers

| Function | Description |
|---|---|
| `optim_sgd()` | SGD with momentum, dampening, weight decay, and Nesterov acceleration |
