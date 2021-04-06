# Backfillz-py Design Notes

Differences to Backfillz-R

- "Themes" are objects of type `BackfillzTheme`
- default theme is only defined once
- Backfillz object is mutable; no need to pass as argument
- given conversion methods from Stan fits and data frames to Backfillz, cleaner for plot methods just to take 
  Backfillz objects 
- `slices` argument to `plot_slice_histogram` is an unused complexity; drop for now
- dropping `stringsAsFactors` for now (not sure what Python equivalent is)
- kill `verbose` for now
