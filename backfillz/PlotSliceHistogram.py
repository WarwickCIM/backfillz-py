from backfillz.Backfillz import Backfillz

def plot_slice_histogram(
    object: Backfillz,
    slices = NULL,
    save_plot = False,
    verbose = True):

  # check inputs
  if (
    !class(object) == "stanfit" &
     !class(object) == "backfillz" &
     !class(object) == "data.frame") {
    stop("Object is not a stanfit, Backfillz or data frame object")
  }

  # convert stanfit
  if ((class(object) == "stanfit") | (class(object) == "data.frame")) {
    object <- as_backfillz(object, verbose)
  }

  # Preallocate the data frame stored in the backfillz object
  object@df_slice_histogram <- data.frame(
    parameter = character(),
    sample_min = numeric(),
    sample_max = numeric(),
    stringsAsFactors = FALSE
  )

  # Check slices argument
  if (is.null(slices)) { # if no argument for slices
    if (verbose) {
          message("Using default slices of 0 - 0.4, 0.8 - 1.")
          message("Plotting the first two parameters only.")
          message(
            paste0("To plot other parameters please pass ",
            "a slice argument to plot_slice_histogram"))
    }
    parameters <-
     as.array(attributes(object@mcmc_samples)$dimnames$parameters)[1:2]
    lower <- c(0, 0.8)
    upper <- c(0.4, 1)
    slices <- data.frame(
      parameters = character(),
      lower = numeric(),
      upper = numeric(),
      stringsAsFactors = TRUE
    )
    for (parameter in parameters) {
      slices <- rbind(
        slices,
        data.frame(
          parameters = rep(parameter, length(upper)),
          lower = lower,
          upper = upper,
          stringsAsFactors = TRUE
        )
      )
    }
  } else { # if the user has passed a slices  argument
    # Extract the parameters
    parameters <- as.array(unique(slices$parameters))
  }

  parameters <- as.matrix(parameters)

  # Create a plot for each parameter
  apply(X = parameters, FUN = create_single_plot, MARGIN = 1)

  id <- max(object@plot_history$ID + 1)

  # Update log
  object@plot_history <- rbind(
    object@plot_history,
    data.frame(
      ID = id,
      Date = date(),
      Event = "Slice Histogram",
      R_version = R.Version()$version.string,
      Saved = save_plot,
      stringsAsFactors = FALSE
    )
  )

  return(object)

def create_single_plot(parameter):
    pass
