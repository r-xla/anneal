#' SGD optimizer
#'
#' Stochastic Gradient Descent with optional momentum, dampening, weight decay,
#' and Nesterov acceleration. Implements the same algorithm as
#' [`torch.optim.SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html).
#'
#' @section Algorithm:
#'
#' For each parameter \eqn{p} with gradient \eqn{g}:
#'
#' \deqn{
#'   g \leftarrow g + \lambda \, p
#' }
#'
#' If \eqn{\mu \neq 0}:
#'
#' \deqn{
#'   b_t \leftarrow
#'     \begin{cases}
#'       g                              & \text{if } t = 0 \\
#'       \mu \, b_{t-1} + (1 - \tau) g  & \text{otherwise}
#'     \end{cases}
#' }
#'
#' If Nesterov: \eqn{g \leftarrow g + \mu \, b_t},
#' otherwise: \eqn{g \leftarrow b_t}.
#'
#' \deqn{
#'   p \leftarrow p - \gamma \, g
#' }
#'
#' where \eqn{\gamma} is the learning rate, \eqn{\lambda} is weight decay,
#' \eqn{\mu} is momentum, and \eqn{\tau} is dampening.
#'
#' @param lr (`numeric(1)`)\cr
#'   Learning rate.
#' @param momentum (`numeric(1)`)\cr
#'   Momentum factor. Default `0`.
#' @param dampening (`numeric(1)`)\cr
#'   Dampening for momentum. Default `0`.
#' @param weight_decay (`numeric(1)`)\cr
#'   Weight decay (L2 penalty). Default `0`.
#' @param nesterov (`logical(1)`)\cr
#'   Whether to use Nesterov momentum. Default `FALSE`.
#'   Requires `momentum > 0` and `dampening == 0`.
#'
#' @returns An optimizer object to pass to [opt_init()] and [opt_step()].
#'
#' @examples
#' params <- list(w = matrix(rnorm(6), 2, 3), b = rnorm(3))
#' grads  <- list(w = matrix(0.1, 2, 3), b = rep(0.1, 3))
#'
#' opt   <- optim_sgd(lr = 0.01, momentum = 0.9)
#' state <- opt_init(opt, params)
#' result <- opt_step(opt, params, grads, state)
#' result$params
#' result$state
#'
#' @export
optim_sgd <- function(lr, momentum = 0, dampening = 0, weight_decay = 0,
                      nesterov = FALSE) {
  assert_sgd_params(lr, momentum, dampening, weight_decay, nesterov)
  init <- function(params) {
    param_state <- lapply(params, function(p) {
      list(momentum_buffer = NULL, step = 0L)
    })
    list(param_state = param_state)
  }

  step <- function(params, gradients, state) {
    param_state <- state$param_state
    nms <- names(params)

    for (nm in nms) {
      p <- params[[nm]]
      g <- gradients[[nm]]

      if (weight_decay != 0) {
        g <- g + weight_decay * p
      }

      if (momentum != 0) {
        ps <- param_state[[nm]]
        if (ps$step == 0L) {
          buf <- g
        } else {
          buf <- momentum * ps$momentum_buffer + (1 - dampening) * g
        }
        param_state[[nm]]$momentum_buffer <- buf
        param_state[[nm]]$step <- ps$step + 1L

        if (nesterov) {
          g <- g + momentum * buf
        } else {
          g <- buf
        }
      }

      params[[nm]] <- p - lr * g
    }

    list(params = params, state = list(param_state = param_state))
  }

  structure(
    list(
      lr = lr,
      momentum = momentum,
      dampening = dampening,
      weight_decay = weight_decay,
      nesterov = nesterov,
      init = init,
      step = step
    ),
    class = c("optim_sgd", "anneal_optimizer")
  )
}

#' @export
print.optim_sgd <- function(x, ...) {
  cli::cli_text("{.cls optim_sgd}")
  cli::cli_ul()
  cli::cli_li("lr: {x$lr}")
  if (x$momentum != 0) cli::cli_li("momentum: {x$momentum}")
  if (x$dampening != 0) cli::cli_li("dampening: {x$dampening}")
  if (x$weight_decay != 0) cli::cli_li("weight_decay: {x$weight_decay}")
  if (x$nesterov) cli::cli_li("nesterov: {x$nesterov}")
  cli::cli_end()
  invisible(x)
}

assert_sgd_params <- function(lr, momentum, dampening, weight_decay, nesterov) {
  checkmate::assert_number(lr, lower = 0)
  checkmate::assert_number(momentum, lower = 0)
  checkmate::assert_number(dampening)
  checkmate::assert_number(weight_decay, lower = 0)
  checkmate::assert_flag(nesterov)
  if (nesterov && (momentum == 0 || dampening != 0)) {
    cli::cli_abort("Nesterov momentum requires {.arg momentum} > 0 and {.arg dampening} == 0.")
  }
}
