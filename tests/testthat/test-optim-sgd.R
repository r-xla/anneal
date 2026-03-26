test_that("basic sgd updates params", {
  params <- list(w = c(1, 2, 3))
  grads  <- list(w = c(0.1, 0.2, 0.3))

  opt   <- optim_sgd(lr = 0.1)
  state <- opt_init(opt, params)
  result <- opt_step(opt, params, grads, state)

  expect_equal(result$params$w, c(1, 2, 3) - 0.1 * c(0.1, 0.2, 0.3))
})

test_that("sgd with weight decay", {
  params <- list(w = c(1, 2, 3))
  grads  <- list(w = c(0.1, 0.2, 0.3))

  opt   <- optim_sgd(lr = 0.1, weight_decay = 0.01)
  state <- opt_init(opt, params)
  result <- opt_step(opt, params, grads, state)

  g <- c(0.1, 0.2, 0.3) + 0.01 * c(1, 2, 3)
  expect_equal(result$params$w, c(1, 2, 3) - 0.1 * g)
})

test_that("sgd with momentum", {
  params <- list(w = c(1, 2))
  grads  <- list(w = c(0.5, 1.0))
  lr <- 0.1
  mu <- 0.9

  opt   <- optim_sgd(lr = lr, momentum = mu)
  state <- opt_init(opt, params)

  # step 1: buffer initialized to gradient
  r1 <- opt_step(opt, params, grads, state)
  buf1 <- c(0.5, 1.0)
  expect_equal(r1$params$w, c(1, 2) - lr * buf1)

  # step 2: buffer = mu * buf1 + g
  grads2 <- list(w = c(0.3, 0.6))
  r2 <- opt_step(opt, r1$params, grads2, r1$state)
  buf2 <- mu * buf1 + grads2$w
  expect_equal(r2$params$w, r1$params$w - lr * buf2)
})

test_that("sgd with dampening", {
  params <- list(w = c(1, 2))
  grads  <- list(w = c(0.5, 1.0))
  lr <- 0.1
  mu <- 0.9
  tau <- 0.5

  opt   <- optim_sgd(lr = lr, momentum = mu, dampening = tau)
  state <- opt_init(opt, params)

  # step 1: buffer = g (no dampening on first step)
  r1 <- opt_step(opt, params, grads, state)
  buf1 <- c(0.5, 1.0)
  expect_equal(r1$params$w, c(1, 2) - lr * buf1)

  # step 2: buffer = mu * buf1 + (1 - tau) * g2
  grads2 <- list(w = c(0.3, 0.6))
  r2 <- opt_step(opt, r1$params, grads2, r1$state)
  buf2 <- mu * buf1 + (1 - tau) * grads2$w
  expect_equal(r2$params$w, r1$params$w - lr * buf2)
})

test_that("sgd with nesterov momentum", {
  params <- list(w = c(1, 2))
  grads  <- list(w = c(0.5, 1.0))
  lr <- 0.1
  mu <- 0.9

  opt   <- optim_sgd(lr = lr, momentum = mu, nesterov = TRUE)
  state <- opt_init(opt, params)

  # step 1: buf = g, then g = g + mu * buf
  r1 <- opt_step(opt, params, grads, state)
  buf1 <- c(0.5, 1.0)
  g1 <- grads$w + mu * buf1
  expect_equal(r1$params$w, c(1, 2) - lr * g1)

  # step 2
  grads2 <- list(w = c(0.3, 0.6))
  r2 <- opt_step(opt, r1$params, grads2, r1$state)
  buf2 <- mu * buf1 + grads2$w
  g2 <- grads2$w + mu * buf2
  expect_equal(r2$params$w, r1$params$w - lr * g2)
})

test_that("sgd with multiple parameters", {
  params <- list(w = matrix(1:4, 2, 2), b = c(0.1, 0.2))
  grads  <- list(w = matrix(0.1, 2, 2), b = c(0.01, 0.02))

  opt   <- optim_sgd(lr = 0.5)
  state <- opt_init(opt, params)
  result <- opt_step(opt, params, grads, state)

  expect_equal(result$params$w, params$w - 0.5 * grads$w)
  expect_equal(result$params$b, params$b - 0.5 * grads$b)
})

test_that("multiple steps accumulate momentum correctly", {
  params <- list(w = 1.0)
  lr <- 0.01
  mu <- 0.9
  opt <- optim_sgd(lr = lr, momentum = mu)
  state <- opt_init(opt, params)

  p <- 1.0
  buf <- NULL
  for (i in 1:5) {
    g <- 2 * p  # gradient of p^2
    if (is.null(buf)) {
      buf <- g
    } else {
      buf <- mu * buf + g
    }
    p <- p - lr * buf

    result <- opt_step(opt, params, list(w = 2 * params$w), state)
    params <- result$params
    state <- result$state
  }

  expect_equal(params$w, p)
})

test_that("state is not mutated", {
  params <- list(w = c(1, 2, 3))
  grads  <- list(w = c(0.1, 0.2, 0.3))

  opt   <- optim_sgd(lr = 0.1, momentum = 0.9)
  state <- opt_init(opt, params)
  state_copy <- state

  opt_step(opt, params, grads, state)

  expect_identical(state, state_copy)
})

test_that("optim_sgd validates parameters", {
  expect_error(optim_sgd(lr = -1))
  expect_error(optim_sgd(lr = 0.1, momentum = -1))
  expect_error(optim_sgd(lr = 0.1, weight_decay = -1))
  expect_error(optim_sgd(lr = 0.1, nesterov = TRUE, momentum = 0), "Nesterov")
  expect_error(optim_sgd(lr = 0.1, nesterov = TRUE, momentum = 0.9, dampening = 0.1), "Nesterov")
})

test_that("opt_init validates params", {
  opt <- optim_sgd(lr = 0.1)
  expect_error(opt_init(opt, list(1, 2)))
  expect_error(opt_init(opt, list()))
})

test_that("print method works", {
  opt <- optim_sgd(lr = 0.01, momentum = 0.9, weight_decay = 1e-4)
  expect_snapshot(print(opt))
})
