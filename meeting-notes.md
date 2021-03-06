# Meeting notes

# 16 September 2021

## Progress on trace dial plot

- James only (missed meetings since 29 July)
  - radial axis on inside of donut; 'arc' helper
  - running in VS Code
  - tests (when running locally) check against expected image output (PNG)

## Progress on spiral stream plot

- James only (missed meetings since 29 July)
  - archimedian (arithmetic) spiral plotting: r = a + b * theta
  - generalises polar_plot (b = 0)
  - plotting variance
- added spiral stream plot to example notebook

## To do: finalising backfillz

- finalise `README.md`
- publish final version to PyPI
- performance of spiral stream plot
- finalise docs

## To discuss: Backfillz object

- not sure of use case here -- somewhat subsumed by notebooks
- maybe WAYS-related use case would help?
- reproducibility best practices usually want *code* not outputs
- how used in R version?

# 2 September 2021

Present: Greg, Ed, Roly

# 19 August 2021

Meeting cancelled (James ill)

# 12 August 2021

Meeting cancelled (James unable to make it, Greg ill)

# 29 July 2021

Present: Greg, Roly

## Progress on trace dial plot

- Greg: key changes from last time (see 15 July meeting notes)
- radial axis on inside of donut
- running in VS Code

## Progress on spiral stream plot

- archimedian (arithmetic) spiral plotting: r = a + b * theta
- generalises polar_plot (b = 0)
- so far just plot the chains, no variance
- total ~125 LOC

## Questions

- What is "inner burn" vs. "outer burn" in trace dial colours?

# 15 July 2021

Present: James, Roly

## Progress

- example notebooks
- consolidate polar plot infrastructure with existing plot design (including axis handling)
- one segment of "donut" per slice
- slice histograms aggregate all chains
- step plot for each chain
- using go.Histogram, we run into z-ordering problem trying to overlay step plots on top:
  - implement our own histograms using go.Bar for plotting, and numpy for binning
  - need this for consistency with step plots anyway (which also use numpy for binning)

# 24 June May 2021

## Progress on trace dial plot

- histograms overplot rather than group
- switch to Cartesian coordinates to render traces, donut
  - solves z-order problem
  - allows correct positioning of histograms (but have to enforce square figure)

# 10 June May 2021

## Progress on trace dial plot

- Polar trace plot (one per chain)
- "Ring" plot (pie chart with hole)
- Generalise histogram from first plot to one histogram per chain

## Questions (mainly in relation to R version)
- How do we decide number of burn-in iterations?
- What are the "polygons" in the R trace_dial plot?
- What is "inner" vs "outer" burn segment?
- What determines the colours used for burn-in and rest of sample?

# 27 May 2021

## Progress on trace slice histogram plot
- no new progress to report (first pass complete)

## Design topics
- Saving history in Backfillz object, use cases:
 - summary of plots I???ve done so far (as show method)
 - recall a particular plot
 - save/load Backfillz object

## On the horizon
- example notebooks, showing usage of Backfillz library + history object

# 13 May 2021

## Progress on trace slice histogram plot
- add density function plot to histogram (one per chain)
- histograms to aggregate all chains
- performance experiment with 1,000,000 iterations
- labels on "joining segments" to right of y-axis
- disable zoom/drag functionality
- plot Raftery-Lewis diagnostic
- additional x-axis above density plots

## Design topics
- Improve class-oriented design:
  - avoid brittle dependency on magic numbers assigned to subplot axes by Plotly
  - parameterise on height of Raftery-Lewis section and width of other 3 sections

## On the horizon:
- make a pass over R code to check for minor viz details/settings
- record plot information in Backfillz
- further design improvements to make subplot titles and row/column specifications less brittle

## To discuss
- Is Raftery-Lewis the right diagnostic, given no longer supported by PyMC3? (And is R dependency ok?)
- Should each RL plot have its own x-axis? Perhaps should be max of expected/actual iterations for all chains?
- Use cases to drive ledger requirements/design

# 29 April 2021

## Progress on trace slice histogram plot
- histogram per slice on RHS (currently for one chain only)
- single x-axis shared by histograms
- one "joining segment" per slice (shaded for now)
- rectangle drawn around in slice in trace plot

## Design topics
- Class-oriented design to make various things explicit:
  - particular "view" of the MCMC data taken by this visualisation
  - overall organisation of top-plot into subplots
  - allows code to be mostly self-documenting
- Plotly vs. Bokeh w.r.t. "compositionality"
  - neither allow arbitrary nesting of figures but only one level of containment
    - in Bokeh, gridplots and row/column plots (which aren't themselves "plots")
    - in Plotly, single flexible subplot grid with cell merging
  - Bokeh: difficult to precisely place subplots because decorations affect size of core plot region
  - Plotly places subplots relative to parent and then attaches decorations independently

# 13 Apr 2021

## Progress
- Use PyStan to generate sample model from 8 Schools example
- save (pickle) the sample model and test generated model against saved version
- first pass over slice histogram plot:
  - LEFT: line plot of all draws (no per-chain colouring yet)
  - MIDDLE: initial stab at "joining segments"
