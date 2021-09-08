<!-- badges: start -->

[![Release build](https://github.com/WarwickCIM/backfillz-py/actions/workflows/build-publish.yml/badge.svg?branch=release)](https://github.com/WarwickCIM/backfillz-py/actions/workflows/build-publish.yml)
[![Develop build](https://github.com/WarwickCIM/backfillz-py/actions/workflows/build-publish.yml/badge.svg?branch=develop)](https://github.com/WarwickCIM/backfillz-py/actions/workflows/build-publish.yml)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

<img src="https://github.com/WarwickCIM/backfillz/raw/master/fig1.png" width=100% alt=""/>

# New View of MCMC

Backfillz-py provides new visual diagnostics for understanding MCMC (Markov Chain Monte Carlo) analyses and outputs. MCMC chains can defy a simple line graph. Unless the chain is very short (which isn’t often the case), plotting tens or hundreds of thousands of data points reveals very little other than a ‘trace plot’ where we only see the outermost points. Common plotting methods may only reveal when an MCMC really hasn’t worked, but not when it has.
BackFillz-py slices and dices MCMC chains so increasingly parameter rich, complex analyses can be visualised meaningfully. What does ‘good mixing’ look like? Is a ‘hair caterpillar’ test verifiable? What does a density plot show and what does it hide?

# Quick Start

Install from [PyPI](https://pypi.org/project/backfillz/).

TODO - code example

# Current prototype plots

TODO

# Emojis on commit messages

Recent commits are prefixed with the following emojis:

- ✨ lint/reformat/document
- 💚 refactor
- ❗ change behaviour

# Acknowledgements

We are grateful for funding from the Alan Turing Institute within the ‘Tools, Practices and Systems’ theme. Initial user research was carried out by GJM on the ‘2020 Science’ programme (www.2020science.net/) funded by the EPSRC Cross-Discipline Interface Programme (grant number EP/I017909/1).
