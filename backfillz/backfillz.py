from typing import List

from stan.fit import Fit  # type: ignore


class BackfillzTheme:
    """Backfillz visualisation settings."""

    def __init__(
        self,
        name: str,
        text_family: str,
        text_font: float,
        text_font_colour: str,
        text_cex_title: float,
        text_cex_main: float,
        text_cex_axis: float,
        text_col_title: str,
        text_col_main: str,
        text_col_axis: str,
        bg_colour: str,
        mg_colour: str,
        fg_colour: str,
        alpha: float,
        theme_palette: List[str]
    ):
        self.name = name
        self.text_family = text_family
        self.text_font = text_font
        self.text_font_colour = text_font_colour
        self.text_cex_title = text_cex_title
        self.text_cex_main = text_cex_main
        self.text_cex_axis = text_cex_axis
        self.text_col_title = text_col_title
        self.text_col_main = text_col_main
        self.text_col_axis = text_col_axis
        self.bg_colour = bg_colour
        self.mg_colour = mg_colour
        self.fg_colour = fg_colour
        self.alpha = alpha
        self.theme_palette = theme_palette

default : BackfillzTheme = BackfillzTheme(
    name = "default",
    text_family = "sans",
    text_font = 1,
    text_font_colour = "black",
    text_cex_title = 1.5,
    text_cex_main = 1,
    text_cex_axis = 0.8,
    text_col_title = "grey10",
    text_col_main = "grey60",
    text_col_axis = "grey40",
    bg_colour = "white",
    mg_colour = "grey50",
    fg_colour = "black",
    alpha = 0.7,
    theme_palette = [
        "#FF0000",
        "#0000FF",
        "#FF00FF",
        "#800000",
        "#000080",
        "#FF6347"
    ]
)

class Backfillz:
    """Represents a Backfillz user session."""

    fit: Fit

    def __init__(self, mcmc_samples: Fit) -> None:
        """Initialise a Backfillz session."""
        pass

    def set_theme(self, theme:str="default", verbose: bool=True) -> None:
        # set theme values
        if theme=="default":
            if verbose:
              print("Setting backfillz object theme to default")
        } else if (theme ==  "solarized_dark") {
        if (verbose) {
          message("Setting backfillz object theme to solarized dark")
        }
        backfillz_object@theme_name               <- "solarized_dark"
        backfillz_object@theme_text_family        <- "mono"
        backfillz_object@theme_text_font          <- 1
        backfillz_object@theme_text_font_colour   <- "#2AA198"

        backfillz_object@theme_text_cex_title     <- 2
        backfillz_object@theme_text_cex_main      <- 1
        backfillz_object@theme_text_cex_axis      <- 0.8

        backfillz_object@theme_text_col_title     <- "grey10"
        backfillz_object@theme_text_col_main      <- "grey60"
        backfillz_object@theme_text_col_axis      <- "grey40"

        backfillz_object@theme_bg_colour          <- "#002B36"
        backfillz_object@theme_mg_colour          <- "grey50"
        backfillz_object@theme_fg_colour          <- "#93A1A1"
        backfillz_object@theme_alpha              <- 0.7
        backfillz_object@theme_palette            <- list(
          "#657B83",
          "#D30102",
          "#D33682",
          "#859900",
          "#93A1A1",
          "#268BD2"
        )
        } else if (theme ==  "demo 1") {
        if (verbose) {
          message("Setting backfillz object theme to demo 1")
        }
        backfillz_object@theme_name               <- "demo 1"
        backfillz_object@theme_text_family        <-  "mono"
        backfillz_object@theme_text_font          <-  1
        backfillz_object@theme_text_font_colour   <- "grey20"

        backfillz_object@theme_text_cex_title     <- 1.5
        backfillz_object@theme_text_cex_main      <- 1
        backfillz_object@theme_text_cex_axis      <- 0.6

        backfillz_object@theme_text_col_title     <- "grey10"
        backfillz_object@theme_text_col_main      <- "grey60"
        backfillz_object@theme_text_col_axis      <- "grey40"

        backfillz_object@theme_bg_colour          <- "grey98"
        backfillz_object@theme_mg_colour          <- "grey90"
        backfillz_object@theme_fg_colour          <- "grey40"
        backfillz_object@theme_alpha              <- 0.8
        backfillz_object@theme_palette            <- list(
          "#A3C96D",
          "#DDCF1E",
          "#8E4D91",
          "#003B24",
          "#912B2F",
          "#7C6EAC"
        )
        } else if (theme ==  "demo 2") {
        if (verbose) {
          message("Setting backfillz object theme to demo 2")
        }
        backfillz_object@theme_name               <- "demo 2"
        backfillz_object@theme_text_family        <-  "sans"
        backfillz_object@theme_text_font          <-  1
        backfillz_object@theme_text_font_colour   <- "grey90"

        backfillz_object@theme_text_cex_title     <- 1.5
        backfillz_object@theme_text_cex_main      <- 1
        backfillz_object@theme_text_cex_axis      <- 0.6

        backfillz_object@theme_text_col_title     <- "#F2EEE7"
        backfillz_object@theme_text_col_main      <- "#F2EEE7"
        backfillz_object@theme_text_col_axis      <- "#F2EEE7"

        backfillz_object@theme_bg_colour          <- "#313C3F"
        backfillz_object@theme_mg_colour          <-
         adjustcolor("#313C3F",
          red.f = 1.8,
          green.f = 1.8,
          blue.f = 1.8)
        backfillz_object@theme_fg_colour          <- "#F2EEE7"
        backfillz_object@theme_alpha              <- 0.8
        backfillz_object@theme_palette            <- list(
          "#EEE436",
          "#00AEC7",
          "#C73475",
          "#7FC5D3",
          "#7EB627",
          "#F29530"
        )

        } else {
        if (verbose) {
          message(paste0("Theme not specified so setting backfillz ",
          "object theme to default"))
        }
        backfillz_object@theme_name               <- "default"
        backfillz_object@theme_text_family        <- "sans"
        backfillz_object@theme_text_font          <- 1
        backfillz_object@theme_text_font_colour   <- "black"

        backfillz_object@theme_text_cex_title     <- 2
        backfillz_object@theme_text_cex_main      <- 1
        backfillz_object@theme_text_cex_axis      <- 0.8

        backfillz_object@theme_text_col_title     <- "grey10"
        backfillz_object@theme_text_col_main      <- "grey60"
        backfillz_object@theme_text_col_axis      <- "grey40"

        backfillz_object@theme_bg_colour          <- "white"
        backfillz_object@theme_mg_colour          <- "grey50"
        backfillz_object@theme_fg_colour          <- "black"

        backfillz_object@theme_alpha              <- 0.7
        backfillz_object@theme_palette            <- list(
          "#FF0000",
          "#0000FF",
          "#FF00FF",
          "#800000",
          "#000080",
          "#FF6347"
        )


def as_backfillz(fit: Fit, verbose: bool) -> Backfillz:
    """Create a Backfillz session from a PyStan fit."""
    backfillz = Backfillz(fit)
    backfillz.set_theme(verbose)

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

    return backfillz
