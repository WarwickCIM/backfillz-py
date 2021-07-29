# Refactoring talk

## Points

- test infrastructure to do bitwise comparison of output images
- implementation goals are:
  - further divide 3/4 donut into one segment per slice
- talk goals are to illustrate:
  - behavioural-invariance tests revealing:
    - new bug
    - unintended change in behaviour
    - weak algebraic properties of graphics libraries
  - over-specialisation, need to re-generalise
  - lots of "fine-grained" (micro-)refactorings

## Steps
- see Git commit messages

## To do

- reduce verbosity of terminal window
- remove mypy annotations, add some comments instead?
