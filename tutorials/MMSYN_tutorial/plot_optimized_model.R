#!/usr/bin/Rscript
# coding: utf-8

#***********************************************************************
# GBApy (Growth Balance Analysis for Python)
# Copyright © 2024-2025 Charles Rocabert
# Web: https://github.com/charlesrocabert/gbapy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#***********************************************************************

library("tidyverse")
library("rstudioapi")
library("cowplot")
library("ggpmisc")
library("Matrix")
library("ggrepel")

### Load mass fractions data ###
load_mass_fractions <- function()
{
  d = read.table("./data/manual_curation/MMSYN_mass_fractions.csv", sep=";", h=T, check.names=F)
  rownames(d) = d$ID
  return(d)
}

### Build the mass fraction data ###
build_mass_fractions_data <- function( d_b, d_mf, i )
{
  D        = d_b[i,]
  D        = D[,-which(names(D)%in%c("condition", "iter", "t", "dt", "h2o"))]
  D        = D[,which(names(D)%in%d_mf$ID)]
  D        = data.frame(names(D), t(D))
  names(D) = c("id", "sim")
  D$obs    = d_mf[D$id,"Fraction"]
  D$obs    = D$obs/sum(D$obs)
  D$sim    = D$sim/sum(D$sim)
  D$obsp3  = D$obs*3
  D$obsm3  = D$obs/3
  D$obsp10 = D$obs*10
  D$obsm10 = D$obs/10
  return(D)
}

### Evolution of mass fraction prediction ###
mass_fraction_evolution <- function( d_b, d_mf )
{
  cor_vec  = c()
  pval_vec = c()
  for(i in seq(1, dim(d_b)[1]))
  {
    D       = build_mass_fractions_data(d_b, d_mf, i)
    reg     = lm(log10(D$sim)~log10(D$obs))
    cor_vec = c(cor_vec, summary(reg)$adj.r.squared)
    pval_vec = c(pval_vec, summary(reg)$coefficients[,4][[2]])
  }
  D = data.frame(d_b$iter, d_b$t, cor_vec, pval_vec)
  names(D) = c("index", "t", "r2", "pval")
  return(D)
}

### Load proteomics data ###
load_proteomics <- function()
{
  d           = read.table(paste0("./data/manual_curation/MMSYN_proteomics.csv"), sep=";", h=T, check.names=F)
  d$protein   = str_replace(d$locus, "JCVISYN3A", "protein")
  d$mass      = d$mg_per_gP*0.001
  rownames(d) = d$protein
  return(d)
}

### Load protein contributions ###
load_protein_contributions <- function( model_path, model_name )
{
  filename = paste0(model_path,"/",model_name,"/protein_contributions.csv")
  d        = read.table(filename, h=T, sep=";", check.names=F)
  return(d)
}

### Load enzyme composition ###
# Note:
# Some reactions are filtered out as they relate to
# the model size reduction.
load_enzyme_composition <- function()
{
  AMINO_ACID_TRANSPORTERS = c("ALAt2r", "ARGt2r", "ASNt2r", "ASPt2pr", "CYSt2r", "GLNt2r",
                              "GLUt2pr", "GLYt2r", "HISt2r", "ISOt2r", "LEUt2r", "LYSt2r",
                              "METt2r", "PHEt2r", "PROt2r", "SERt2r", "THRt2r", "TRPt2r",
                              "TYRt2r", "VALt2r")
  CHARGING_REACTIONS = c("ALATRS", "ARGTRS", "ASNTRS", "ASPTRS", "CYSTRS", "GLUTRS",
                         "GLYTRS", "HISTRS", "ILETRS", "LEUTRS", "LYSTRS", "METTRS",
                         "PHETRS", "PROTRS", "SERTRS", "THRTRS", "TRPTRS", "TYRTRS",
                         "VALTRS")
  filename = paste0("./data/manual_curation/MMSYN_enzyme_composition.csv")
  d        = read.table(filename, h=T, sep=";", check.names=F)
  d        = filter(d, !info_sample_rid %in% c("Protein_transl", AMINO_ACID_TRANSPORTERS, CHARGING_REACTIONS))
  return(d)
}

### Calculate simulated proteomics ###
calculate_simulated_proteomics <- function( d_p, protein_contributions, i )
{
  X             = d_p[i,-which(names(d_p)%in%c("condition", "iter", "t", "dt"))]
  r_ids         = names(X)
  p_ids         = unique(protein_contributions$protein)
  res           = data.frame(p_ids, rep(0.0, length(p_ids)))
  names(res)    = c("p_id", "value")
  rownames(res) = res$p_id
  for (r_id in r_ids)
  {
    e_conc  = X[r_id][[1]]
    contrib = protein_contributions[protein_contributions$reaction==r_id,]
    if (dim(contrib)[1] > 0)
    {
      for (i in seq(1, dim(contrib)[1]))
      {
        p_id              = contrib$protein[i]
        value             = e_conc*contrib$contribution[i]
        res[p_id,"value"] = res[p_id,"value"]+value
      }
    }
  }
  res$value = res$value/sum(res$value)
  return(res)
}

### Collect proteins excluded from the model ###
collect_excluded_proteins <- function( list_of_reactions, enzyme_composition )
{
  excluded_proteins = c()
  for(i in seq(1, dim(enzyme_composition)[1]))
  {
    r_id = enz_comp$info_sample_rid[i]
    if (!r_id %in% list_of_reactions | r_id %in% c("Ribosome", "AAabc", "AATRS"))
    {
      step1 = enz_comp$composition[i]
      step2 = strsplit(step1,"|",fixed=T)
      for (elmt in step2[[1]])
      {
        step3             = strsplit(elmt,",",fixed=T)
        prot_id           = str_replace(step3[[1]][1], "gene=JCVISYN3A_", "")
        prot_id           = str_replace(prot_id, " ", "")
        excluded_proteins = c(excluded_proteins, paste0("protein_",prot_id))
      }
    }
  }
  return(excluded_proteins)
}

### Build the proteome fraction data ###
build_proteomics_data <- function( d_p )
{
  obs_proteomics = load_proteomics()
  pcontrib       = load_protein_contributions(model_path, model_name)
  enz_comp       = load_enzyme_composition()
  excluded       = collect_excluded_proteins(reaction_ids, enz_comp)
  sim_proteomics = calculate_simulated_proteomics(d_p, pcontrib, dim(d_p)[1])
  D              = data.frame(sim_proteomics$p_id, sim_proteomics$value)
  names(D)       = c("id", "sim")
  D              = filter(D, id%in%obs_proteomics$protein)
  D$obs          = obs_proteomics[D$id, "mass"]
  #D$obs          = D$obs/sum(D$obs)
  D$sim          = D$sim/sum(D$sim)*0.34
  D$obsp3        = D$obs*3
  D$obsm3        = D$obs/3
  D$obsp10       = D$obs*10
  D$obsm10       = D$obs/10
  D              = D[!D$id%in%excluded,]
  return(D)
}


##################
#      MAIN      #
##################

directory = dirname(getActiveDocumentContext()$path)
setwd(directory)

model_path  = "./models"
output_path = "./output"
model_name  = "mmsyn_fcr_v1"
condition   = 1

d_state = read.table(paste0(output_path,"/",model_name,"_",condition,"_state_trajectory.csv"), h=T, sep=";", check.names=F)
d_f     = read.table(paste0(output_path,"/",model_name,"_",condition,"_f_trajectory.csv"), h=T, sep=";", check.names=F)
d_v     = read.table(paste0(output_path,"/",model_name,"_",condition,"_v_trajectory.csv"), h=T, sep=";", check.names=F)
d_p     = read.table(paste0(output_path,"/",model_name,"_",condition,"_p_trajectory.csv"), h=T, sep=";", check.names=F)
d_b     = read.table(paste0(output_path,"/",model_name,"_",condition,"_b_trajectory.csv"), h=T, sep=";", check.names=F)
d_c     = read.table(paste0(output_path,"/",model_name,"_",condition,"_c_trajectory.csv"), h=T, sep=";", check.names=F)

reaction_ids   = names(d_p[,-which(names(d_p)%in%c("condition", "iter", "t", "dt"))])
mass_fractions = load_mass_fractions()

MF      = build_mass_fractions_data(d_b, mass_fractions, dim(d_b)[1])
MF_evol = mass_fraction_evolution(d_b, mass_fractions)
#PR      = build_proteomics_data(d_p)

#plot(PR$obs, PR$sim, log="xy", pch=20)
#abline(a=0, b=1)


p1 = ggplot(MF, aes(obs, sim)) +
  geom_abline(slope=1, intercept=0, color="pink") +
  geom_point() +
  scale_x_log10() + scale_y_log10() +
  geom_text_repel(aes(label=id), size = 3.5) +
  xlab("Observed") +
  ylab("Simulated") +
  ggtitle("Metabolite mass fractions") +
  theme_classic()

p2 = ggplot(MF_evol, aes(index, r2)) +
  geom_line() +
  xlab("Iteration") +
  ylab("Linear regression adj. R-squared") +
  ggtitle("Correlation with observed mass fractions") +
  theme_classic()

p3 = ggplot(MF_evol, aes(index, pval)) +
  geom_line() +
  geom_hline(yintercept=0.05, col="pink") +
  scale_y_log10() +
  xlab("Iteration") +
  ylab("Linear regression p-value") +
  ggtitle("Correlation with observed mass fractions") +
  theme_classic()

plot_grid(p1, p2, p3, nrow=1)

ggplot(d_state, aes(iter, mu)) +
  geom_line() +
  theme_classic()
