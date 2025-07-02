#!/usr/bin/Rscript
# coding: utf-8

library("tidyverse")
library("rstudioapi")
library("cowplot")
library("ggpmisc")
library("Matrix")
library("ggrepel")
library("latex2exp")
library("MASS")
library("renz")

#-----------------------------------#
#      DATA ANALYSIS FUNCTIONS      #
#-----------------------------------#

### Load a trajectory ###
load_trajectory <- function( model_name, suffix )
{
  d = read.table(paste0("./output/optimization/",model_name,"_",suffix,"_trajectory.csv"), h=T, sep=";", check.names=F)
}

### Load an optimum ###
load_optimum <- function( model_name, suffix )
{
  d = read.table(paste0("./output/optimization/",model_name,"_",suffix,"_optimum.csv"), h=T, sep=";", check.names=F)
}

### Load external conditions ###
load_conditions <- function( model_name )
{
  conditions           = read.table(paste0("./models/", model_name, "/conditions.csv"), sep=";", h=T, check.names=F)
  rownames(conditions) = conditions[,1]
  conditions           = conditions[,2:ncol(conditions)]
  return(as.integer(colnames(conditions)))
}

### Load external conditions ###
load_external_glucose <- function( model_name )
{
  conditions           = read.table(paste0("./models/", model_name, "/conditions.csv"), sep=";", h=T, check.names=F)
  rownames(conditions) = conditions[,1]
  conditions           = conditions[,2:ncol(conditions)]
  return(as.vector(t(conditions["x_glc__D",])))
}

### Load the observed mass fractions data ###
load_observed_mass_fractions <- function()
{
  d           = read.table("./data/manual_curation/MMSYN_mass_fractions.csv", sep=";", h=T, check.names=F)
  d$Fraction  = d$Fraction*0.01
  rownames(d) = d$ID
  return(d)
}

### Build the mass fraction data ###
build_mass_fractions_data <- function( model_name, condition )
{
  d_b            = load_optimum(model_name, "b")
  i              = which(d_b$condition==condition)
  d_mf           = load_observed_mass_fractions()
  D              = d_b[i,-which(names(d_b)%in%c("condition", "h2o"))]
  D              = data.frame(names(D), t(D))
  names(D)       = c("id", "sim_mass")
  D$sim_fraction = D$sim_mass/sum(D$sim_mass)
  D              = filter(D, id%in%d_mf$ID)
  D$obs_fraction = d_mf[D$id,"Fraction"]
  #############
  #D$sim_fraction = D$sim_fraction/sum(D$sim_fraction)
  #D$obs_fraction = D$obs_fraction/sum(D$obs_fraction)
  #############
  D$obsp3        = D$obs_fraction*3
  D$obsm3        = D$obs_fraction/3
  D$obsp10       = D$obs_fraction*10
  D$obsm10       = D$obs_fraction/10
  D$Category     = d_mf[D$id,"Category"]
  return(D)
}

### Correlation of mass fraction prediction ###
mass_fractions_correlation <- function( model_name, condition )
{
  D     = build_mass_fractions_data(model_name, condition)
  reg   = lm(log10(D$sim_fraction)~log10(D$obs_fraction))
  r2    = summary(reg)$adj.r.squared
  pval  = summary(reg)$coefficients[,4][[2]]
  SSres = sum((log10(D$sim_fraction)-log10(D$obs_fraction))^2)
  M     = mean(log10(D$obs_fraction))
  SStot = sum((log10(D$obs_fraction)-M)^2)
  R2    = 1-SSres/SStot
  return(c(r2, pval, R2))
}

### Mass fraction prediction through conditions ###
mass_fractions_by_condition <- function( model_name )
{
  conditions = load_conditions(model_name)
  cor_vec    = c()
  pval_vec   = c()
  R2_vec     = c()
  for(condition in conditions)
  {
    res      = mass_fractions_correlation(model_name, condition)
    cor_vec  = c(cor_vec, res[1])
    pval_vec = c(pval_vec, res[2])
    R2_vec   = c(R2_vec, res[3])
  }
  D           = data.frame(cor_vec, pval_vec, R2_vec)
  names(D)    = c("r2", "pval", "R2")
  D$condition = conditions
  return(D)
}

### Load observed proteomics data ###
load_observed_proteomics <- function()
{
  d           = read.table("./data/manual_curation/MMSYN_proteomics.csv", sep=";", h=T, check.names=F)
  d$protein   = d$p_id
  d$obs_mass  = d$mg_per_gP*0.001
  rownames(d) = d$protein
  d           = filter(d, obs_mass > 0.0)
  return(d)
}

### Load protein contributions ###
load_protein_contributions <- function( model_name )
{
  #------------------------------------------------#
  # 1) Load original model's protein contributions #
  #------------------------------------------------#
  d            = read.table("./output/JCVISYN3A_protein_contributions_bis.csv", h=T, sep=";", check.names=F)
  nrow         = length(unique(d$protein))
  ncol         = length(unique(d$reaction))
  M1           = matrix(rep(0.0, nrow*ncol), nrow, ncol)
  rownames(M1) = sort(unique(d$protein))
  colnames(M1) = sort(unique(d$reaction))
  for(i in seq(1, dim(d)[1]))
  {
    M1[d$protein[i], d$reaction[i]] = d$contribution[i]
  }
  #------------------------------------------------#
  # 2) Load reduced model's protein contributions  #
  #------------------------------------------------#
  filename     = paste0("./models/",model_name,"/protein_contributions.csv")
  d            = read.table(filename, h=T, sep=";", check.names=F)
  nrow         = length(unique(d$protein))
  ncol         = length(unique(d$reaction))
  M2           = matrix(rep(0.0, nrow*ncol), nrow, ncol)
  rownames(M2) = sort(unique(d$protein))
  colnames(M2) = sort(unique(d$reaction))
  for(i in seq(1, dim(d)[1]))
  {
    M2[d$protein[i], d$reaction[i]] = d$contribution[i]
  }
  #------------------------------------------------#
  # 3) - Remove Protein_transl reaction            #
  #    - Remove spontaneous protein                #
  #    - Merge Ribosome reaction                   #
  #------------------------------------------------#
  M1                     = M1[,-which(colnames(M1)=="Protein_transl")]
  M1                     = M1[-which(rownames(M1)=="spontaneous_protein"),]
  M2                     = M2[-which(rownames(M2)=="spontaneous_protein"),]
  R_prot                 = rownames(M2)[which(M2[,"Ribosome"]>0)]
  R_missing              = R_prot[!R_prot%in%rownames(M1)]
  M1_comp                = matrix(nrow=length(R_missing), ncol=ncol(M1))
  M1_comp[,]             = 0
  rownames(M1_comp)      = R_missing
  M1                     = rbind(M1, M1_comp)
  M1                     = cbind(M1, rep(0.0, nrow(M1)))
  colnames(M1)[ncol(M1)] = "Ribosome"
  for(i in nrow(M1))
  {
    p_id = rownames(M1)[i]
    if (p_id%in%rownames(M2))
    {
      val              = M2[p_id,"Ribosome"]
      M1[i,"Ribosome"] = val
    }
  }
  return(list("original"=M1, "reduced"=M2))
}

### Collect proteins excluded from the model ###
collect_excluded_proteins <- function( pcontributions )
{
  #-------------------------------------------------------#
  # 1) Define AA and tRNA reactions in the original model #
  #-------------------------------------------------------#
  AMINO_ACID_TRANSPORTERS = c("ALAt2r", "ARGt2r", "ASNt2r", "ASPt2pr", "CYSt2r", "GLNt2r",
                              "GLUt2pr", "GLYt2r", "HISt2r", "ISOt2r", "LEUt2r", "LYSt2r",
                              "METt2r", "PHEt2r", "PROt2r", "SERt2r", "THRt2r", "TRPt2r",
                              "TYRt2r", "VALt2r")
  CHARGING_REACTIONS = c("ALATRS", "ARGTRS", "ASNTRS", "ASPTRS", "CYSTRS", "GLUTRS",
                         "GLYTRS", "HISTRS", "ILETRS", "LEUTRS", "LYSTRS", "METTRS",
                         "PHETRS", "PROTRS", "SERTRS", "THRTRS", "TRPTRS", "TYRTRS",
                         "VALTRS")
  #-------------------------------------------------------#
  # 2) Detect reactions to exclude                        #
  #-------------------------------------------------------#
  M1                  = pcontributions[["original"]]
  M2                  = pcontributions[["reduced"]]
  excluded_proteins   = c()
  unmodeled_reactions = colnames(M1)[!colnames(M1)%in%colnames(M2)]
  for(p_id in rownames(M2))
  {
    reactions = M1[p_id,]
    reactions = reactions[reactions>0]
    reactions = reactions[!reactions%in%AMINO_ACID_TRANSPORTERS & !reactions%in%CHARGING_REACTIONS]
    reactions = names(reactions)
    if (p_id!="average_protein" & length(reactions) > 0 & sum(reactions%in%unmodeled_reactions) > 0)
    {
      excluded_proteins = c(excluded_proteins, p_id)
    }
  }
  return(excluded_proteins)
}

### Calculate simulated proteomics ###
calculate_simulated_proteomics <- function( pcontributions, model_name, condition, d_p )
{
  i                = which(d_p$condition==condition)
  M1               = pcontributions[["original"]]
  M2               = pcontributions[["reduced"]]
  M                = M2
  X                = d_p[,-which(names(d_p)%in%c("condition"))]
  X                = data.frame(colnames(d_p), t(d_p[i,]))
  X                = X[colnames(M),]
  Y                = M%*%X[,2]
  r_ids            = colnames(M)
  p_ids            = rownames(M)
  res              = data.frame(p_ids, Y)
  names(res)       = c("p_id", "sim_mass")
  rownames(res)    = res$p_id
  res$sim_fraction = res$sim_mass/sum(res$sim_mass)
  return(res)
}

### Calculate the modeled proteome fraction ###
calculate_modeled_proteome_fraction <- function( model_name, d_p )
{
  obs_proteomics = load_observed_proteomics()
  pcontributions = load_protein_contributions(model_name)
  sim_proteomics = calculate_simulated_proteomics(pcontributions, model_name, 1, d_p)
  Xsum           = sum(obs_proteomics$obs_mass)
  Xsim           = sum(filter(obs_proteomics, protein%in%sim_proteomics$p_id)$obs_mass)
  return(Xsim/Xsum)
}

### Calculate the ribosome fraction in observed proteomics ###
calculate_phi_obs <- function()
{
  obs  = load_observed_proteomics()
  ribp = read.table("./output/JCVISYN3A_ribosomal_proteins.csv", h=T, sep=";")
  return(sum(filter(obs, protein%in%ribp$id)$obs_mass))
}

### Build the proteome fraction data ###
build_proteomics_data <- function( model_name, condition )
{
  d_p            = load_optimum(model_name, "p")
  i              = which(d_p$condition==condition)
  reaction_ids   = names(d_p[,-which(names(d_p)%in%c("condition"))])
  obs_proteomics = load_observed_proteomics()
  pcontributions = load_protein_contributions(model_name)
  excluded       = collect_excluded_proteins(pcontributions)
  sim_proteomics = calculate_simulated_proteomics(pcontributions, model_name, condition, d_p)
  D              = data.frame(sim_proteomics$p_id, sim_proteomics$sim_mass, sim_proteomics$sim_fraction)
  names(D)       = c("id", "sim_mass", "sim_fraction")
  D              = filter(D, id%in%obs_proteomics$protein)
  D$obs_mass     = obs_proteomics[D$id, "obs_mass"]
  D              = filter(D, !id%in%excluded)
  D$obs_fraction = D$obs_mass
  #D$obs_fraction = D$obs_mass/sum(D$obs_mass)
  #D$sim_fraction = D$sim_mass/sum(D$sim_mass)
  D$obsp3        = D$obs_fraction*3
  D$obsm3        = D$obs_fraction/3
  D$obsp10       = D$obs_fraction*10
  D$obsm10       = D$obs_fraction/10
  return(D)
}

### Correlation of proteomics prediction ###
proteomics_correlation <- function( model_name, condition )
{
  D     = build_proteomics_data(model_name, condition)
  reg   = lm(log10(D$sim_fraction)~log10(D$obs_fraction))
  r2    = summary(reg)$adj.r.squared
  pval  = summary(reg)$coefficients[,4][[2]]
  SSres = sum((log10(D$sim_fraction)-log10(D$obs_fraction))^2)
  M     = mean(log10(D$obs_fraction))
  SStot = sum((log10(D$obs_fraction)-M)^2)
  R2    = 1-SSres/SStot
  return(c(r2, pval, R2))
}

### Proteomics prediction through conditions ###
proteomics_by_condition <- function( model_name )
{
  conditions = load_conditions(model_name)
  cor_vec    = c()
  pval_vec   = c()
  R2_vec     = c()
  for(condition in conditions)
  {
    res      = proteomics_correlation(model_name, condition)
    cor_vec  = c(cor_vec, res[1])
    pval_vec = c(pval_vec, res[2])
    R2_vec   = c(R2_vec, res[3])
  }
  D = data.frame(cor_vec, pval_vec, R2_vec)
  names(D) = c("r2", "pval", "R2")
  D$condition = conditions
  return(D)
}

#------------------------------#
#      PLOTTING FUNCTIONS      #
#------------------------------#

### Plot the growth rate depending on external glucose ###
plot_growth_rate <- function( model_name )
{
  d_state           = load_optimum(model_name, "state")
  glc               = load_external_glucose(model_name)
  d_state$glc       = glc
  mu_obs            = read.table("./data/wet_experiments/observed_mu.csv", sep=";", h=T)
  avg_mu_obs        = tapply(log10(mu_obs$mu), mu_obs$glc, mean)
  avg_mu_obs        = data.frame(as.numeric(names(avg_mu_obs)), 10^as.vector(avg_mu_obs))
  names(avg_mu_obs) = c("glc", "mu")
  p = ggplot(d_state, aes(glc, mu)) +
    geom_point(data=mu_obs, aes(glc, mu), color="cornflowerblue") +
    geom_line() +
    scale_x_log10() + scale_y_log10() +
    xlab("External glucose concentration") +
    ylab("Growth rate") +
    ggtitle("Growth rate") +
    theme_classic()
  return(p)
}

### Plot the predicted growth law ###
plot_predicted_growth_law <- function( model_name )
{
  d_p                  = load_optimum(model_name, "p")
  d_state              = load_optimum(model_name, "state")
  phi_obs              = calculate_phi_obs()
  dl                   = d_p[,-which(names(d_p)%in%c("condition"))]
  dl$sum               = rowSums(dl)
  dl$Ribosome_fraction = dl[,"Ribosome"]/dl$sum
  dl$condition         = d_p$condition
  dl$mu                = d_state$mu
  last_ribosome_fraction  = dl$Ribosome_fraction[dim(dl)[1]]
  p = ggplot(dl, aes(mu, Ribosome_fraction)) +
    geom_point(aes(x=0.34, y=phi_obs), col="cornflowerblue", lwd=3) +
    geom_line() +
    xlab("Growth rate") +
    ylab("Ribosome fraction") +
    ggtitle("Growth law") +
    theme_classic()
  return(p)
}

### Plot predicted protein fraction ###
plot_predicted_protein_fraction <- function( model_name )
{
  d_c                 = load_optimum(model_name, "c")
  d_state             = load_optimum(model_name, "state")
  proteins            = c("ACP", "PdhC", "dUTPase", "Protein")
  dl                  = d_c[,-which(names(d_c)%in%c("condition", "h2o"))]
  dl$sum              = rowSums(dl)
  #dl$Protein_fraction = rowSums(dl[,proteins])/dl$sum
  dl$Protein_fraction = (dl[,"Protein"])/dl$sum
  dl$condition        = d_c$condition
  dl$mu               = d_state$mu
  last_prot_fraction  = dl$Protein_fraction[dim(dl)[1]]
  p = ggplot(dl, aes(mu, Protein_fraction)) +
    geom_hline(yintercept=0.54727, color="cornflowerblue", lty=2, lwd=1) +
    geom_line() +
    xlab("Growth rate") +
    ylab("Protein fraction") +
    ggtitle("Protein fraction") +
    ylim(0,1) +
    theme_classic()
  return(p)
}

### Plot the mass fraction of housekeeping proteins ###
plot_predicted_housekeeping_fraction <- function( model_name )
{
  d_p           = load_optimum(model_name, "p")
  d_state       = load_optimum(model_name, "state")
  modeled_pfrac = calculate_modeled_proteome_fraction(model_name, d_p)
  dl            = d_p[,-which(names(d_p)%in%c("condition"))]
  psum          = rowSums(dl)
  atpR          = dl$ATPase/psum
  D             = data.frame(d_p$condition, d_state$mu, atpR)
  print(paste0("> Housekeeping fraction = ", atpR[length(atpR)]))
  names(D) = c("condition", "mu", "fraction")
  p = ggplot(D, aes(mu, fraction)) +
    geom_hline(yintercept=0.5, col="cornflowerblue", lty=2, lwd=1) +
    geom_line() +
    xlab("Growth rate") +
    ylab("Housekeeping proteins fraction") +
    ggtitle("Housekeeping proteins fraction") +
    theme_classic()
  return(p)
}
  
### Plot predicted mass fractions ###
plot_predicted_mass_fractions <- function( MF, MF_by_cond )
{
  R2 = filter(MF_by_cond, condition==67)$R2
  r2 = filter(MF_by_cond, condition==67)$r2
  p = ggplot(MF, aes(obs_fraction, sim_fraction)) +
    geom_abline(slope=1, intercept=0, color="pink") +
    geom_line(aes(obs_fraction, obsp3), color="grey", lty=2) +
    geom_line(aes(obs_fraction, obsm3), color="grey", lty=2) +
    geom_line(aes(obs_fraction, obsp10), color="grey", lty=3) +
    geom_line(aes(obs_fraction, obsm10), color="grey", lty=3) +
    geom_point() +
    #geom_smooth(method="lm") +
    scale_x_log10() + scale_y_log10() +
    #geom_text_repel(aes(label=id), size = 3.5) +
    annotate("text", x=1e-5, y=5, label=paste0("• R2 = ", round(R2,2)), hjust=0) +
    annotate("text", x=1e-5, y=1, label=paste0("• r2 = ", round(r2,2)), hjust=0) +
    xlab("Observed") +
    ylab("Simulated") +
    ggtitle("Metabolite mass fractions (5 g/L glucose)") +
    theme_classic()
  return(p)
}

### Plot metabolite mass fraction correlations per condition ###
plot_mass_fractions_by_condition <- function( MF_by_cond )
{
  p1 = ggplot(MF_by_cond, aes(condition, r2)) +
    geom_line() +
    xlab("Condition") +
    ylab("Linear regression adj. R-squared") +
    ggtitle("Correlation with observed mass fractions") +
    theme_classic()
  p2 = ggplot(MF_by_cond, aes(condition, pval)) +
    geom_line() +
    geom_hline(yintercept=0.05, col="pink") +
    scale_y_log10() +
    xlab("Condition") +
    ylab("Linear regression p-value") +
    ggtitle("Correlation with observed mass fractions") +
    theme_classic()
  p3 = ggplot(MF_by_cond, aes(condition, R2)) +
    geom_line() +
    xlab("Condition") +
    ylab("R2") +
    ggtitle("Simulated vs. observed metabolite fractions") +
    theme_classic()
  return(list(p1, p2, p3))
}

### Plot predicted proteomics ###
plot_predicted_proteomics <- function( PR, PR_by_cond )
{
  p = ggplot(PR, aes(obs_fraction, sim_fraction)) +
    geom_abline(slope=1, intercept=0, color="pink") +
    geom_line(aes(obs_fraction, obsp3), color="grey", lty=2) +
    geom_line(aes(obs_fraction, obsm3), color="grey", lty=2) +
    geom_line(aes(obs_fraction, obsp10), color="grey", lty=3) +
    geom_line(aes(obs_fraction, obsm10), color="grey", lty=3) +
    geom_point() +
    #geom_smooth(method="lm") +
    scale_x_log10() + scale_y_log10() +
    #annotate("text", x=0.001, y=1, label=paste0("italic(R)^2", "==", round(R2,5)), hjust=0, parse=T) +
    #annotate("text", x=0.001, y=0.1, label=paste0("italic(r)^2", "==", round(r2,5)), hjust=0, parse=T) +
    #geom_text_repel(aes(label=id), size = 3.5) +
    xlab("Observed") +
    ylab("Simulated") +
    ggtitle("Proteomics") +
    theme_classic()
  return(p)
}

### Plot proteomics correlations per condition ###
plot_proteomics_by_condition <- function( PR_by_cond )
{
  p1 = ggplot(PR_by_cond, aes(condition, r2)) +
    geom_line() +
    xlab("Condition") +
    ylab("Linear regression adj. R-squared") +
    ggtitle("Correlation with observed proteomics") +
    theme_classic()
  p2 = ggplot(PR_by_cond, aes(condition, pval)) +
    geom_line() +
    geom_hline(yintercept=0.05, col="pink") +
    scale_y_log10() +
    xlab("Condition") +
    ylab("Linear regression p-value") +
    ggtitle("Correlation with observed proteomics") +
    theme_classic()
  p3 = ggplot(PR_by_cond, aes(condition, R2)) +
    geom_line() +
    xlab("Condition") +
    ylab("R2") +
    ggtitle("R2 with observed proteomics") +
    theme_classic()
  return(list(p1, p2, p3))
}

### Plot a given flux depending on growth rate ###
plot_flux <- function( model_name, r_id, title )
{
  d_v      = load_optimum(model_name, "f")
  d_state  = load_optimum(model_name, "state")
  D        = data.frame(d_state$mu, d_v[,r_id])
  names(D) = c("mu", "flux")
  ggplot(D, aes(mu, flux)) +
    geom_line() +
    xlab("Growth rate") +
    ylab("Flux") +
    ggtitle(title) +
    theme_classic()
}

### Plot a given metabolite concentration depending on growth rate ###
plot_metabolite_concentration <- function( model_name, m_id, title, log )
{
  d_c      = load_optimum(model_name, "c")
  d_state  = load_optimum(model_name, "state")
  D        = data.frame(d_state$mu, d_c[,m_id])
  names(D) = c("mu", "concentration")
  p = ggplot(D, aes(mu, -log10(concentration))) +
    geom_line() +
    xlab("Growth rate") +
    ylab("Concentration") +
    ggtitle(title) +
    theme_classic()
  if (grepl("x", log, fixed=TRUE))
  {
    p = p + scale_x_log10()
  }
  if (grepl("y", log, fixed=TRUE))
  {
    p = p + scale_y_log10()
  }
  return(p)
}
plot_metabolite_concentration(model_name, "h", "Proton concentration", log="")

### Plot a given protein concentration depending on growth rate ###

##################
#      MAIN      #
##################

directory = dirname(getActiveDocumentContext()$path)
setwd(directory)

#---------------------------------#
# 1) Define main model properties #
#---------------------------------#
data_path  = "../../gbapy/tutorials/MMSYN_tutorial"
model_name = "mmsyn_fcr_v2"

#---------------------------------#
# 2) Build the different datasets #
#---------------------------------#
MF_67      = build_mass_fractions_data(model_name, 67)
MF_by_cond = mass_fractions_by_condition(model_name)
PR_67      = build_proteomics_data(model_name, 67)
PR_by_cond = proteomics_by_condition(model_name)

#---------------------------------#
# 3) Build the figures            #
#---------------------------------#
p1 = plot_growth_rate(model_name)
p2 = plot_predicted_growth_law(model_name)
p3 = plot_predicted_protein_fraction(model_name)
p4 = plot_predicted_housekeeping_fraction(model_name)
p5 = plot_predicted_mass_fractions(MF_67, MF_by_cond)
p6 = plot_mass_fractions_by_condition(MF_by_cond)[[3]]
p7 = plot_predicted_proteomics(PR_67, PR_by_cond)
p8 = plot_proteomics_by_condition(PR_by_cond)[[3]]

p9  = plot_flux(model_name, "Ht", "Proton import rate")
p10 = plot_metabolite_concentration(model_name, "h", "Proton concentration", log="")
p11 = plot_flux(model_name, "L_LACt2r", "Lactate excretion rate")
p12 = plot_metabolite_concentration(model_name, "lac__L", "Lactate concentration", log="")

p13 = plot_flux(model_name, "ATPase", "ATP maintenance rate")

plot_grid(p1, p2, p3, p4, p5, p6, p7, p8, ncol=4)

plot_grid(p9, p10, p11, p12)

p13
p10
p9
