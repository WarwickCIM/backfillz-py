<!-- badges: start -->

[![Release build](https://github.com/WarwickCIM/backfillz-py/actions/workflows/build-publish.yml/badge.svg?branch=release)](https://github.com/WarwickCIM/backfillz-py/actions/workflows/build-publish.yml)
[![Develop build](https://github.com/WarwickCIM/backfillz-py/actions/workflows/build-publish.yml/badge.svg?branch=develop)](https://github.com/WarwickCIM/backfillz-py/actions/workflows/build-publish.yml)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

<img src="https://github.com/WarwickCIM/backfillz/raw/master/fig1.png" width=100% alt=""/>

# New View of MCMC

Backfillz-py provides new visual diagnostics for understanding MCMC (Markov Chain Monte Carlo) analyses and outputs. MCMC chains can defy a simple line graph. Unless the chain is very short (which isn’t often the case), plotting tens or hundreds of thousands of data points reveals very little other than a ‘trace plot’ where we only see the outermost points. Common plotting methods may only reveal when an MCMC really hasn’t worked, but not when it has.
BackFillz-py slices and dices MCMC chains so increasingly parameter rich, complex analyses can be visualised meaningfully. What does ‘good mixing’ look like? Is a ‘hair caterpillar’ test verifiable? What does a density plot show and what does it hide?

# Quick Start

Install from [PyPI](https://pypi.org/project/backfillz/) using `pip install backfillz`.

```python
from backfillz import Backfillz

# Let's have an example Stan model.
from backfillz.example.eight_schools import generate_fit

backfillz = Backfillz(generate_fit().fit)

# Plot some of the available plot types.
backfillz.plot_slice_histogram('mu')
backfillz.plot_trace_dial('theta')
backfillz.plot_spiral_stream('mu', [2, 8, 15, 65, 250, 600])
```

See the [example notebook](https://github.com/WarwickCIM/backfillz-py/blob/develop/notebooks/example.ipynb) for running in JupyterLab.

# Current supported plot types

## Pretzel Plot – plot_trace_dial()

This plot shows the chain and summary histograms in a format that can be easily arranged as a grid. The trace plot is stretched, clearly indicating ‘burn-in’, with density plots showing the burn-in and remainder of the chain in context.

<img src="tests/expected_trace_dial.png" width=100% alt=""/>

## Slice plot - plot_slice_histogram()

By partitioning chain slices, in a faceted view, users can assess chain convergence. The slices are currently specified by the user and display density plots for each slice. Have my chains converged? The slice plot offers a clear view of when and how convergence is achieved. Further statistical diagnostics can be embedded in these plots as colour encodings or additional layers and annotations.

<img src="tests/expected_slice_histogram.png" width=100% alt=""/>

## Splash plot - plot_spiral_stream()

Based on a Theodorus spiral, we turn MCMC chains into glyphs and extract properties to answer – What does ‘good mixing’ look like? In these plots variance windows are calculated across chains and parameters. The glyphs have clear diagnostic features and will allow gridded plots to investigate large numbers of parameters.

<img src="tests/expected_spiral_stream.png" width=100% alt=""/>

# Emojis on commit messages

Recent commits use the following `git` aliases (add to `[alias]` section of your `.gitconfig`):

```
doc      = "!f() { git commit -a -m \"📚 : $1\"; }; f"
lint     = "!f() { git commit -a -m \"✨ : $1\"; }; f"
modify   = "!f() { git commit -a -m \"❗ : $1\"; }; f"
refactor = "!f() { git commit -a -m \"♻️ : $1\"; }; f"
```

# Acknowledgements

We are grateful for funding from the Alan Turing Institute within the [Tools, Practices and Systems](https://www.turing.ac.uk/research/research-programmes/tools-practices-and-systems) theme. Initial user research was carried out by GJM on the [2020 Science programme](www.2020science.net/) funded by the EPSRC Cross-Discipline Interface Programme (grant number EP/I017909/1).
