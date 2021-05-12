# Meeting notes

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
