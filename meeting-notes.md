# Meeting notes

# 29 April 2021

## Progress on trace slice histogram plot
- 

## Design topics to mention
- Class-oriented design to make various things explicit/self-documenting:
  - particular "view" of the MCMC data taken by this visualisation
  - overall organisation of top-plot into subplots
- Plotly vs. Bokeh w.r.t. "compositionality"
  - neither allow arbitrary nesting of figures but only top-level organisation
    - in Bokeh, gridplots and row/column plots (which aren't themselves "plots")
    - in Plotly, single flexible subplot grid with cell merging
  - Bokeh: difficult to precisely place subplots because decorations affect size of core plot region
  - Plotly places subplots relative to parent and then attaches decorations independently 

## Still to do:
- Raftery-Lewis diagnostic
- make a pass over R code to check for things I've missed

# 13 Apr 2021

## Progress
- Use PyStan to generate sample model from 8 Schools example
- save (pickle) the sample model and test generated model against saved version
- first pass over slice histogram plot:
  - LEFT: line plot of all draws (no per-chain colouring yet)
  - MIDDLE: initial stab at "joining segments"
