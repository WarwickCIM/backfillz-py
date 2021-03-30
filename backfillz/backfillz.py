from stan.fit import Fit # type: ignore


class Backfillz:
    """Represents a Backfillz user session."""

    def __init__(self) -> None:
        """Initialise a Backfillz session."""
        pass


def as_backfillz(fit: Fit, verbose: bool) -> Backfillz:
    """Create a Backfillz session from a PyStan fit."""
    # create backfillz object
    backfillz_object = Backfillz()

    # populate backfillz object and set theme
    #  if (class(object) == "stanfit") {
    #    backfillz_object@mcmc_samples <-
    #      rstan::extract(object, permuted = FALSE, inc_warmup = TRUE)
    #    backfillz_object@mcmc_model <- object@stanmodel@model_code
    #  } else if (class(object) == "data.frame") {
    #    backfillz_object@mcmc_samples <- df_to_stanarray(object)
    #    backfillz_object@mcmc_model <- "Samples imported from dataframe"
    #  }

    # set default theme
    #  backfillz_object <- set_theme(backfillz_object, verbose)

    # initialise plot history
    #  backfillz_object@plot_history <- data.frame(
    #    ID = 1,
    #    Date = date(),
    #    Event = "Object Creation",
    #    R_version = R.Version()$version.string,
    #    Saved = FALSE,
    #    stringsAsFactors = FALSE
    #  )

    return backfillz_object
