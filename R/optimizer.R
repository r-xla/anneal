#' Create optimizer state
#'
#' Initializes the optimizer state for a set of parameters.
#' Each optimizer defines how its state is structured (e.g. momentum buffers).
#'
#' @param optimizer An optimizer object created by a constructor like [optim_sgd()].
#' @param params Named list of parameter arrays.
#'
#' @returns A named list of optimizer state per parameter, plus any global state.
#'
#' @export
opt_init <- function(optimizer, params) {
  checkmate::assert_list(params, names = "unique", min.len = 1L)
  optimizer$init(params)
}

#' Perform one optimization step
#'
#' Computes updated parameters and optimizer state from current parameters,
#' gradients, and optimizer state. This is a pure function -- it does not
#' mutate any of its arguments.
#'
#' @param optimizer An optimizer object created by a constructor like [optim_sgd()].
#' @param params Named list of current parameter arrays.
#' @param gradients Named list of gradient arrays, with names matching `params`.
#' @param state Optimizer state as returned by [opt_init()] or a previous [opt_step()].
#'
#' @returns A list with:
#'   - `params`: Named list of updated parameter arrays.
#'   - `state`: Updated optimizer state.
#'
#' @export
opt_step <- function(optimizer, params, gradients, state) {
  optimizer$step(params, gradients, state)
}
