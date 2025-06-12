#!/usr/bin/Rscript
# coding: utf-8

library("tidyverse")
library("rstudioapi")


##################
#      MAIN      #
##################

directory = dirname(getActiveDocumentContext()$path)
setwd(directory)

proteo   = read.table("proteomics_exponential_phase.csv", h=T, sep=";", check.names=F)
proteins = read.csv("../../output/JCVISYN3A_proteins.csv", sep=";")

#-------------------------------------------------------#
# 1) Add protein molecular weights to the proteome data #
#-------------------------------------------------------#
rownames(proteins) = proteins$id
proteo$mass = proteins[proteo$p_id,"mass"]

#-------------------------------------------------------#
# 2) Calculate fg per cell                              #
#-------------------------------------------------------#
proteo$fg_per_cell = proteo$mass*proteo$copy_per_cell/6.022e+23*1.0e+15
fg_sum = sum(proteo$fg_per_cell)

#-------------------------------------------------------#
# 3) Calculate g protein per g total protein            #
#-------------------------------------------------------#
proteo$mg_per_gP = proteo$fg_per_cell/fg_sum*1000.0

#-------------------------------------------------------#
# 4) Save                                               #
#-------------------------------------------------------#
write.table(proteo, file="../manual_curation/MMSYN_proteomics.csv", row.names=F, col.names=T, quote=F, sep=";")

