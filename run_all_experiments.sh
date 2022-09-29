##!/usr/bin/env bash
# Benchmarking of AL setup

for SIMULATOR in higdon1d motorcycle gramacy2d branin2d ishigami3d hartmann6d gramacy1d
do
  for REPETITION in 1 #{1..10}
  do

    if [ 'motorcycle' = ${SIMULATOR} ]; then AL_STEPS=3; fi #40
    if [ 'gramacy1d' = ${SIMULATOR} ]; then AL_STEPS=3; fi #90
    if [ 'gramacy2d' = ${SIMULATOR} ]; then AL_STEPS=3; fi # 40
    if [ 'higdon1d' = ${SIMULATOR} ]; then AL_STEPS=3; fi #60
    if [ 'branin2d' = ${SIMULATOR} ]; then AL_STEPS=3; fi # 40
    if [ 'ishigami3d' = ${SIMULATOR} ]; then AL_STEPS=3; fi # 200
    if [ 'hartmann6d' = ${SIMULATOR} ]; then AL_STEPS=3; fi # 60

    # FBGP: ALM   
    python al_experiments.py --simulator ${SIMULATOR} --seed ${REPETITION} --active_learning_steps ${AL_STEPS} --metamodel_name "${SIMULATOR}_variance_${REPETITION}" --selection_criteria variance
    # FBGP: B-ALM
    python al_experiments.py --simulator ${SIMULATOR} --seed ${REPETITION} --active_learning_steps ${AL_STEPS} --metamodel_name "${SIMULATOR}_mcmc_mean_variance_${REPETITION}" --selection_criteria mcmc_mean_variance
    # FBGP: B-QBC
    python al_experiments.py --simulator ${SIMULATOR} --seed ${REPETITION} --active_learning_steps ${AL_STEPS} --metamodel_name "${SIMULATOR}_mcmc_qbc_${REPETITION}" --selection_criteria mcmc_qbc
    # FBGP: QB-MGP
    python al_experiments.py --simulator ${SIMULATOR} --seed ${REPETITION} --active_learning_steps ${AL_STEPS} --metamodel_name "${SIMULATOR}_mcmc_gmm_${REPETITION}" --selection_criteria mcmc_gmm
    # FBGP: BALD
    python al_experiments.py --simulator ${SIMULATOR} --seed ${REPETITION} --active_learning_steps ${AL_STEPS} --metamodel_name "${SIMULATOR}_mcmc_bald_${REPETITION}" --selection_criteria mcmc_bald
   
  done
done

