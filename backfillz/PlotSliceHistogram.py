import pandas as pd
from typing import Optional

from backfillz.Backfillz import Backfillz

def plot_slice_histogram(
    backfillz: Backfillz,
    slices: Optional[pd.DataFrame] = None,
    save_plot: bool = False,
    verbose: bool = True):

  # Preallocate the data frame stored in the backfillz object
  backfillz.df_slice_histogram = pd.DataFrame(columns=[
    'parameter'  # character
    'sample_min'  # numeric
    'sample_max'  # numeric
    'stringsAsFactors'  # bool (False)
  ])

  if slices is None:
    if verbose:
      print("Using default slices of 0 - 0.4, 0.8 - 1.")
      print("Plotting the first two parameters only.")
      print("To plot other parameters please pass a slice argument to plot_slice_histogram")

    parameters = as.array(attributes(object@mcmc_samples)$dimnames$parameters)[1:2]
    lower = c(0, 0.8)
    upper = c(0.4, 1)
    slices = pd.DataFrame(
      'parameters'  # character
      'lower'  # numeric
      'upper'  # numeric
      'stringsAsFactors'  # bool (True)
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
