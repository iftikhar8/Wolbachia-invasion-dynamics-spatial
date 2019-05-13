# Wolbachia-dynamics-spatial
gyp_sim_spat.cpp (version 12/05/2019) was developed by Dr Penelope A. Hancock

gyp_sim_spat.cpp comprises C++ code for implementing the spatial model of mosquito-Wolbachia dynamics developed in Hancock et al. 2019, "Predicting the spatial dynamics of Wolbachia infections in Aedes aegypti arbovirus vector populations in heterogeneous landscapes", Journal of Applied Ecology.

Requires R (version 3.4) and GSL (version 1.15)

To run gyp_sim use the command gyp_sim_spat.exe<gyp_sim_spat_inits.txt

A movie of an example simulated spread is shown in spread_Cairns.mp4

The file gyp_sim_spat_inits.txt contains the following inputs:

Demog_params.txt (a file containing the input values of the density-independent daily adult mortality, the density-independent daily larval mortality, and the daily probability that an adult moves to a neighbouring subpopulation)

Adults_file.txt (a file for storing the number of uninfected adults present in each house on each day)

Adults_wolb_file.txt (a file for storing the number of infected adults present in each house on each day)

lambda_file.txt (a file for storing the per-capita female fecundity in each subpopulation (house) at the time that each cohort is hatched)

release_size.txt (a file for storing the size of each Wolbachia release) 

rel_mod.txt (a file containing a factor by which the release size is multiplied)

patch_probs.txt (a file containing the value of (1-the proportion of houses in the landscape with high quality habitat) and (1-the proportion of houses in the landscape with low quality habitat).

A value equaling the fitness cost s experienced by Wolbachia-infected adults

