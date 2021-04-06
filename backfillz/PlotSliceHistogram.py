from datetime import datetime
import pandas as pd
import sys
from typing import Optional

from backfillz.Backfillz import Backfillz, HistoryEntry, HistoryEvent

def plot_slice_histogram(
    backfillz: Backfillz,
    save_plot: bool = False,
    verbose: bool = True):

    # Preallocate the data frame stored in the backfillz object
    backfillz.df_slice_histogram = pd.DataFrame(columns=[
        'parameter'  # character
        'sample_min'  # numeric
        'sample_max'  # numeric
        'stringsAsFactors'  # bool (False)
    ])

    parameters = array(attributes(backfillz.fit).dimnames.parameters)[1:2]
    lower = c(0, 0.8)
    upper = c(0.4, 1)
    slices = pd.DataFrame(columns=[
      'parameters'  # character
      'lower'  # numeric
      'upper'  # numeric
      'stringsAsFactors'  # bool (True)
    ])
    for parameter in parameters:
        slices = rbind(
            slices,
            pd.DataFrame(
                parameters = rep(parameter, length(upper)),
                lower = lower,
                upper = upper,
                stringsAsFactors = True
            )
        )

    parameters = matrix(parameters)

    for parameter in parameters:
        create_single_plot(parameter)

    ident = max(map(lambda entry: entry.ident, backfillz.plot_history)) + 1

    # Update log
    backfillz.plot_history.append(HistoryEntry(
        ident=ident,
        date=datetime.now(),
        event=HistoryEvent.SLICE_HISTOGRAM,
        python_version=sys.version,
        saved=save_plot,
        strings_as_factors=False
    ))


def create_single_plot(parameter):
    pass
