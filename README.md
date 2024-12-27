current problems/to do:

[DONE] 1) heat map shows average success of around 6% at best. this is both retarded and unpossible.
   -find source of this error - probably in how X1 and X2 are used
   -May be an alignment error

1.5) Figure out why first column of heat map is so much different from the rest

1.75) Write Code to predict later than just one step for actual price/up down movement
     -make this match a hot spot for m and l values
   
2) Make sure that algorithm actually reflects whats going on in the DMD paper
   -many of the functions were chatgpt'd. Output seems mostly correct (except for heat map) but make sure its correct under the hood
   -datasets may need to be tweated
   -l and m values may need to be tweaked

3) test model on different datasets
   -see if we can get the same data that was used in the paper

4) generalize the model and package it in such a way that we can drop it in a larger algo
   -automatically update with current days price and add to dataset
   -live updates for heat map accuracy
   -run it on a server?

5) If using koopman DMD we may need to find more rigorous way of finding observables. Right now the paper uses the data itself and this should probably be fine
    - PCA
    - autoencoders (?) <-is this gay?
