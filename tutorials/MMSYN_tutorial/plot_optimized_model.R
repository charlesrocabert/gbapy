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
library("latex2exp")

### Load mass fractions data ###
load_mass_fractions <- function()
{
  d           = read.table("./data/manual_curation/MMSYN_mass_fractions.csv", sep=";", h=T, check.names=F)
  rownames(d) = d$ID
  return(d)
}

### Build the mass fraction data ###
build_mass_fractions_data <- function( d_b, i )
{
  d_mf     = load_mass_fractions()
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

### Correlation of mass fraction prediction ###
mass_fractions_cor <- function( d_b )
{
  D     = build_mass_fractions_data(d_b, dim(d_b)[1])
  reg   = lm(log10(D$sim)~log10(D$obs))
  r2    = summary(reg)$adj.r.squared
  pval  = summary(reg)$coefficients[,4][[2]]
  SSres = sum((log10(D$sim)-log10(D$obs))^2)
  M     = mean(log10(D$obs))
  SStot = sum((log10(D$obs)-M)^2)
  R2    = 1-SSres/SStot
  return(c(r2, pval, R2))
}

### Evolution of mass fraction prediction ###
mass_fraction_evolution <- function( d_b, step )
{
  iter     = c()
  cor_vec  = c()
  pval_vec = c()
  R2_vec   = c()
  for(i in seq(1, dim(d_b)[1], by=step))
  {
    iter     = c(iter, i)
    D        = build_mass_fractions_data(d_b, i)
    reg      = lm(log10(D$sim)~log10(D$obs))
    cor_vec  = c(cor_vec, summary(reg)$adj.r.squared)
    pval_vec = c(pval_vec, summary(reg)$coefficients[,4][[2]])
    SSres    = sum((log10(D$sim)-log10(D$obs))^2)
    M        = mean(log10(D$obs))
    SStot    = sum((log10(D$obs)-M)^2)
    R2_vec   = c(R2_vec, 1-SSres/SStot)
  }
  D = data.frame(iter, cor_vec, pval_vec, R2_vec)
  names(D) = c("index", "r2", "pval", "R2")
  return(D)
}

### Load proteomics data ###
load_proteomics <- function()
{
  d           = read.table(paste0("./data/manual_curation/MMSYN_proteomics.csv"), sep=";", h=T, check.names=F)
  d$protein   = str_replace(d$locus, "JCVISYN3A", "protein")
  d$mass      = d$mg_per_gP*0.001
  rownames(d) = d$protein
  d           = filter(d, mass > 0.0)
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
  X             = d_p[i, -which(names(d_p)%in%c("condition", "iter", "t", "dt"))]
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
      for (j in seq(1, dim(contrib)[1]))
      {
        p_id              = contrib$protein[j]
        value             = e_conc*contrib$contribution[j]
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
    r_id = enzyme_composition$info_sample_rid[i]
    if (!r_id %in% list_of_reactions | r_id %in% c("Ribosome", "AAabc", "AATRS"))
    {
      step1 = enzyme_composition$composition[i]
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
build_proteomics_data <- function( d_p, i )
{
  obs_proteomics = load_proteomics()
  pcontrib       = load_protein_contributions(model_path, model_name)
  enz_comp       = load_enzyme_composition()
  reaction_ids   = names(d_p[,-which(names(d_p)%in%c("condition", "iter", "t", "dt"))])
  excluded       = collect_excluded_proteins(reaction_ids, enz_comp)
  sim_proteomics = calculate_simulated_proteomics(d_p, pcontrib, i)
  D              = data.frame(sim_proteomics$p_id, sim_proteomics$value)
  names(D)       = c("id", "sim")
  obs_sum        = sum(obs_proteomics$mass)
  sim_sum        = sum(D$sim)
  D              = filter(D, id%in%obs_proteomics$protein)
  D$obs          = obs_proteomics[D$id, "mass"]
  D              = D[!D$id%in%excluded,]
  D$obs          = D$obs/sum(D$obs)
  D$sim          = D$sim/sum(D$sim)#*0.23#/sim_sum#*0.23
  D$obsp3        = D$obs*3
  D$obsm3        = D$obs/3
  D$obsp10       = D$obs*10
  D$obsm10       = D$obs/10
  
  return(D)
}

### Correlation of proteomics prediction ###
proteomics_cor <- function( d_p )
{
  D     = build_proteomics_data(d_p, dim(d_p)[1])
  reg   = lm(log10(D$sim)~log10(D$obs))
  r2    = summary(reg)$adj.r.squared
  pval  = summary(reg)$coefficients[,4][[2]]
  SSres = sum((log10(D$sim)-log10(D$obs))^2)
  M     = mean(log10(D$obs))
  SStot = sum((log10(D$obs)-M)^2)
  R2    = 1-SSres/SStot
  return(c(r2, pval, R2))
}

### Evolution of proteomics prediction ###
proteomics_evolution <- function( d_p, step )
{
  iter     = c()
  cor_vec  = c()
  pval_vec = c()
  R2_vec   = c()
  for(i in seq(1, dim(d_p)[1], by=step))
  {
    iter     = c(iter, i)
    D        = build_proteomics_data(d_p, i)
    reg      = lm(log10(D$sim)~log10(D$obs))
    cor_vec  = c(cor_vec, summary(reg)$adj.r.squared)
    pval_vec = c(pval_vec, summary(reg)$coefficients[,4][[2]])
    SSres    = sum((log10(D$sim)-log10(D$obs))^2)
    M        = mean(log10(D$obs))
    SStot    = sum((log10(D$obs)-M)^2)
    R2_vec   = c(R2_vec, 1-SSres/SStot)
  }
  D = data.frame(iter, cor_vec, pval_vec, R2_vec)
  names(D) = c("index", "r2", "pval", "R2")
  return(D)
}

plot_growth_rate <- function( d_state )
{
  last_mu = d_state$mu[dim(d_state)[1]]
  p = ggplot(d_state, aes(iter, mu)) +
    geom_line() +
    xlab("Iterations") +
    ylab("Growth rate") +
    ggtitle(paste0("Growth rate (\u03BC = ",round(last_mu,3),")")) +
    theme_classic()
  return(p)
}

plot_protein_fraction <- function( d_c )
{
  dl                  = d_c[,-which(names(d_c)%in%c("condition","iter","t","dt", "h2o"))]
  dl$sum              = rowSums(dl)
  dl$Protein_fraction = dl$Protein/dl$sum
  dl$iter             = d_c$iter
  last_prot_fraction  = dl$Protein_fraction[dim(dl)[1]]
  p = ggplot(dl, aes(iter, Protein_fraction)) +
    geom_hline(yintercept=0.54727, color="pink") +
    geom_line() +
    xlab("Iterations") +
    ylab("Protein fraction") +
    ggtitle(paste0("Protein fraction (Pf = ",round(last_prot_fraction,3),", obs = ",0.547,")")) +
    ylim(0,1) +
    theme_classic()
  return(p)
}

plot_mass_fractions <- function( mf_data, R2 )
{
  p = ggplot(mf_data, aes(obs, sim)) +
    # geom_abline(slope=1, intercept=0, color="pink") +
    # geom_line(aes(obs, obsp3), color="grey", lty=2) +
    # geom_line(aes(obs, obsm3), color="grey", lty=2) +
    # geom_line(aes(obs, obsp10), color="grey", lty=3) +
    # geom_line(aes(obs, obsm10), color="grey", lty=3) +
    geom_point() +
    geom_smooth(method="lm") +
    scale_x_log10() + scale_y_log10() +
    #geom_text_repel(aes(label=id), size = 3.5) +
    annotate("text", x=1e-5, y=5e-1, label=paste0("italic(R)^2", "==", round(R2,2)), hjust=0, parse=T) +
    xlab("Observed") +
    ylab("Simulated") +
    ggtitle("Metabolite mass fractions") +
    theme_classic()
  return(p)
}

plot_mass_fractions_evolution <- function( mf_evol_data )
{
  p1 = ggplot(mf_evol_data, aes(index, r2)) +
    geom_line() +
    xlab("Iteration") +
    ylab("Linear regression adj. R-squared") +
    ggtitle("Correlation with observed mass fractions") +
    theme_classic()
  p2 = ggplot(mf_evol_data, aes(index, pval)) +
    geom_line() +
    geom_hline(yintercept=0.05, col="pink") +
    scale_y_log10() +
    xlab("Iteration") +
    ylab("Linear regression p-value") +
    ggtitle("Correlation with observed mass fractions") +
    theme_classic()
  p3 = ggplot(mf_evol_data, aes(index, R2)) +
    geom_line() +
    xlab("Iteration") +
    ylab("R2") +
    ggtitle("R2 with observed mass fractions") +
    theme_classic()
  return(list(p1, p2, p3))
}

plot_proteomics <- function( pr_data, R2 )
{
  p = ggplot(pr_data, aes(obs, sim)) +
    # geom_abline(slope=1, intercept=0, color="pink") +
    # geom_line(aes(obs, obsp3), color="grey", lty=2) +
    # geom_line(aes(obs, obsm3), color="grey", lty=2) +
    # geom_line(aes(obs, obsp10), color="grey", lty=3) +
    # geom_line(aes(obs, obsm10), color="grey", lty=3) +
    geom_point() +
    geom_smooth(method="lm") +
    scale_x_log10() + scale_y_log10() +
    annotate("text", x=0.001, y=0.05, label=paste0("italic(R)^2", "==", round(R2,2)), hjust=0, parse=T) +
    #geom_text_repel(aes(label=id), size = 3.5) +
    xlab("Observed") +
    ylab("Simulated") +
    ggtitle("Proteomics") +
    theme_classic()
  return(p)
}

plot_proteomics_evolution <- function( pr_evol_data )
{
  p1 = ggplot(pr_evol_data, aes(index, r2)) +
    geom_line() +
    xlab("Iteration") +
    ylab("Linear regression adj. R-squared") +
    ggtitle("Correlation with observed proteomics") +
    theme_classic()
  p2 = ggplot(pr_evol_data, aes(index, pval)) +
    geom_line() +
    geom_hline(yintercept=0.05, col="pink") +
    scale_y_log10() +
    xlab("Iteration") +
    ylab("Linear regression p-value") +
    ggtitle("Correlation with observed proteomics") +
    theme_classic()
  p3 = ggplot(pr_evol_data, aes(index, R2)) +
    geom_line() +
    xlab("Iteration") +
    ylab("R2") +
    ggtitle("R2 with observed proteomics") +
    theme_classic()
  return(list(p1, p2, p3))
}

##################
#      MAIN      #
##################

directory = dirname(getActiveDocumentContext()$path)
setwd(directory)

model_path  = "./models"
output_path = "./output/old_version"
output_path = "./output"
model_name  = "mmsyn_nfcr"
condition   = 1

d_state = read.table(paste0(output_path,"/",model_name,"_",condition,"_state_trajectory.csv"), h=T, sep=";", check.names=F)
d_f     = read.table(paste0(output_path,"/",model_name,"_",condition,"_f_trajectory.csv"), h=T, sep=";", check.names=F)
d_v     = read.table(paste0(output_path,"/",model_name,"_",condition,"_v_trajectory.csv"), h=T, sep=";", check.names=F)
d_p     = read.table(paste0(output_path,"/",model_name,"_",condition,"_p_trajectory.csv"), h=T, sep=";", check.names=F)
d_b     = read.table(paste0(output_path,"/",model_name,"_",condition,"_b_trajectory.csv"), h=T, sep=";", check.names=F)
d_c     = read.table(paste0(output_path,"/",model_name,"_",condition,"_c_trajectory.csv"), h=T, sep=";", check.names=F)

MF      = build_mass_fractions_data(d_b, dim(d_b)[1])
MF_cor  = mass_fractions_cor(d_b)
MF_evol = mass_fraction_evolution(d_b, 10)
PR      = build_proteomics_data(d_p, dim(d_p)[1])
PR_cor  = proteomics_cor(d_p)

p1 = plot_growth_rate(d_state)
p2 = plot_protein_fraction(d_c)
p3 = plot_mass_fractions(MF, MF_cor[3])
p4 = plot_proteomics(PR, PR_cor[3])
p_mf = plot_mass_fractions_evolution(MF_evol)
plot_grid(p1, p2, p3, p4, p_mf[[1]], ncol=2)

#MF_evol = mass_fraction_evolution(d_b, 100)
#PR_evol = proteomics_evolution(d_p, 100)
#p1   = plot_mass_fractions(MF)

#p2 = plot_proteomics(PR)
#p_pr = plot_proteomics_evolution(PR_evol)
#plot_grid(p1, p_mf[[1]], p_mf[[2]], p_mf[[3]], p2, p_pr[[1]], p_pr[[2]], p_pr[[3]], ncol=4)

X = d_f[dim(d_f)[1],-which(names(d_f)%in%c("condition", "iter", "t", "dt"))]
X = data.frame(names(X), t(X))
X = X[order(X[,2], decreasing=T),]
X
