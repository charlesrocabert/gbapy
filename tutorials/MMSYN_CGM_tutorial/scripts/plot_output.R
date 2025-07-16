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


d_state = read.table(paste0(data_path,"/output/optimization/",model_name,"_state_optimum.csv"), h=T, sep=";", check.names=F)
d_f     = read.table(paste0(data_path,"/output/optimization/",model_name,"_f_optimum.csv"), h=T, sep=";", check.names=F)
d_v     = read.table(paste0(data_path,"/output/optimization/",model_name,"_v_optimum.csv"), h=T, sep=";", check.names=F)
d_p     = read.table(paste0(data_path,"/output/optimization/",model_name,"_p_optimum.csv"), h=T, sep=";", check.names=F)
d_b     = read.table(paste0(data_path,"/output/optimization/",model_name,"_b_optimum.csv"), h=T, sep=";", check.names=F)
d_c     = read.table(paste0(data_path,"/output/optimization/",model_name,"_c_optimum.csv"), h=T, sep=";", check.names=F)

load_trajectory <- function( data_path, model_name, suffix )
{
  d = read.table(paste0(data_path,"/output/optimization/",model_name,"_state_optimum.csv"), h=T, sep=";", check.names=F)
  
}

### Load the observed mass fractions data ###
load_observed_mass_fractions <- function( data_path )
{
  d           = read.table(paste0(data_path, "/data/manual_curation/MMSYN_mass_fractions.csv"), sep=";", h=T, check.names=F)
  d$Fraction  = d$Fraction*0.01
  rownames(d) = d$ID
  return(d)
}

### Build the mass fraction data ###
build_mass_fractions_data <- function( data_path, d_b, cond )
{
  i              = which(d_b$condition==cond)
  d_mf           = load_observed_mass_fractions(data_path)
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
mass_fractions_correlation <- function( data_path, d_b, cond )
{
  D     = build_mass_fractions_data(data_path, d_b, cond)
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
mass_fractions_by_condition <- function( data_path, d_b )
{
  cor_vec  = c()
  pval_vec = c()
  R2_vec   = c()
  for(cond in d_b$condition)
  {
    res      = mass_fractions_correlation(data_path, d_b, cond)
    cor_vec  = c(cor_vec, res[1])
    pval_vec = c(pval_vec, res[2])
    R2_vec   = c(R2_vec, res[3])
  }
  D           = data.frame(cor_vec, pval_vec, R2_vec)
  names(D)    = c("r2", "pval", "R2")
  D$condition = d_b$condition
  return(D)
}

### Load observed proteomics data ###
load_observed_proteomics <- function( data_path )
{
  d           = read.table(paste0(data_path, "/data/manual_curation/MMSYN_proteomics.csv"), sep=";", h=T, check.names=F)
  d$protein   = d$p_id
  d$obs_mass  = d$mg_per_gP*0.001
  rownames(d) = d$protein
  d           = filter(d, obs_mass > 0.0)
  return(d)
}

### Load protein contributions ###
load_protein_contributions <- function( data_path, model_name )
{
  #------------------------------------------------#
  # 1) Load original model's protein contributions #
  #------------------------------------------------#
  filename     = paste0(data_path,"/output/JCVISYN3A_protein_contributions.csv")
  d            = read.table(filename, h=T, sep=";", check.names=F)
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
  filename     = paste0(data_path, "/models/",model_name,"/protein_contributions.csv")
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
calculate_simulated_proteomics <- function( pcontributions, d_p, cond )
{
  i                = which(d_p$condition==cond)
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
calculate_modeled_proteome_fraction <- function( data_path, model_name, d_p )
{
  obs_proteomics = load_observed_proteomics(data_path)
  pcontributions = load_protein_contributions(data_path, model_name)
  sim_proteomics = calculate_simulated_proteomics(pcontributions, d_p, nrow(d_p))
  Xsum           = sum(obs_proteomics$obs_mass)
  Xsim           = sum(filter(obs_proteomics, protein%in%sim_proteomics$p_id)$obs_mass)
  return(Xsim/Xsum)
}

### Calculate the ribosome fraction in observed proteomics ###
calculate_phi_obs <- function( data_path )
{
  obs  = load_observed_proteomics(data_path)
  ribp = read.table(paste0(data_path, "/output/JCVISYN3A_ribosomal_proteins.csv"), h=T, sep=";")
  return(sum(filter(obs, protein%in%ribp$id)$obs_mass))
}

### Build the proteome fraction data ###
build_proteomics_data <- function( data_path, model_name, d_p, cond )
{
  i              = which(d_p$condition==cond)
  reaction_ids   = names(d_p[,-which(names(d_p)%in%c("condition"))])
  obs_proteomics = load_observed_proteomics(data_path)
  pcontributions = load_protein_contributions(data_path, model_name)
  excluded       = collect_excluded_proteins(pcontributions)
  sim_proteomics = calculate_simulated_proteomics(pcontributions, d_p, i)
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
proteomics_correlation <- function( data_path, model_name, d_p, cond )
{
  D     = build_proteomics_data(data_path, model_name, d_p, cond)
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
proteomics_by_condition <- function( data_path, model_name, d_p )
{
  cor_vec  = c()
  pval_vec = c()
  R2_vec   = c()
  for(cond in d_p$condition)
  {
    res      = proteomics_correlation(data_path, model_name, d_p, cond)
    cor_vec  = c(cor_vec, res[1])
    pval_vec = c(pval_vec, res[2])
    R2_vec   = c(R2_vec, res[3])
  }
  D = data.frame(cor_vec, pval_vec, R2_vec)
  names(D) = c("r2", "pval", "R2")
  D$condition = d_p$condition
  return(D)
}

### Plot predicted protein fraction ###
plot_predicted_protein_fraction <- function( d_c )
{
  proteins            = c("ACP", "PdhC", "dUTPase", "Protein")
  dl                  = d_c[,-which(names(d_c)%in%c("condition", "h2o"))]
  dl$sum              = rowSums(dl)
  #dl$Protein_fraction = rowSums(dl[,proteins])/dl$sum
  dl$Protein_fraction = (dl[,"Protein"])/dl$sum
  dl$condition        = d_c$condition
  last_prot_fraction  = dl$Protein_fraction[dim(dl)[1]]
  p = ggplot(dl, aes(condition, Protein_fraction)) +
    geom_hline(yintercept=0.54727, color="pink") +
    geom_line() +
    xlab("Condition") +
    ylab("Protein fraction") +
    ggtitle(paste0("Protein fraction (Pf = ",round(last_prot_fraction,3),", obs = ",0.547,")")) +
    ylim(0,1) +
    theme_classic()
  return(p)
}

### Plot the predicted growth law ###
plot_predicted_growth_law <- function( d_p, d_state, phi_obs )
{
  dl                   = d_p[,-which(names(d_p)%in%c("condition"))]#, "ATPase"))]
  dl$sum               = rowSums(dl)
  dl$Ribosome_fraction = dl[,"Ribosome"]/dl$sum
  dl$condition         = d_p$condition
  dl$mu                = d_state$mu
  last_ribosome_fraction  = dl$Ribosome_fraction[dim(dl)[1]]
  p = ggplot(dl, aes(mu, Ribosome_fraction)) +
    geom_hline(yintercept=phi_obs, col="pink") +
    geom_line() +
    xlab("Growth rate") +
    ylab("Ribosome fraction") +
    ggtitle(paste0("Ribosome fraction (Rf = ",round(last_ribosome_fraction,3),")")) +
    theme_classic()
  return(p)
}

### Plot the mass fraction of housekeeping proteins ###
plot_predicted_housekeeping_fraction <- function( d_p, modeled_pfrac )
{
  dl       = d_p[,-which(names(d_p)%in%c("condition"))]
  psum     = rowSums(dl)
  atpR     = dl$ATPase/psum
  D        = data.frame(d_p$condition, atpR)
  print(paste0("> Housekeeping fraction = ", atpR[length(atpR)]))
  names(D) = c("condition", "fraction")
  p = ggplot(D, aes(condition, fraction)) +
    geom_hline(yintercept=1-modeled_pfrac, col="pink") +
    geom_line() +
    xlab("Condition") +
    ylab("Housekeeping proteins fraction") +
    ggtitle("Housekeeping proteins fraction") +
    theme_classic()
  return(p)
}
  
### Plot predicted mass fractions ###
plot_predicted_mass_fractions <- function( mf_data, R2, r2 )
{
  p = ggplot(mf_data, aes(obs_fraction, sim_fraction)) +
    geom_abline(slope=1, intercept=0, color="pink") +
    geom_line(aes(obs_fraction, obsp3), color="grey", lty=2) +
    geom_line(aes(obs_fraction, obsm3), color="grey", lty=2) +
    geom_line(aes(obs_fraction, obsp10), color="grey", lty=3) +
    geom_line(aes(obs_fraction, obsm10), color="grey", lty=3) +
    geom_point() +
    #geom_smooth(method="lm") +
    scale_x_log10() + scale_y_log10() +
    #geom_text_repel(aes(label=id), size = 3.5) +
    annotate("text", x=1e-5, y=10, label=paste0("italic(R)^2", "==", round(R2,2)), hjust=0, parse=T) +
    annotate("text", x=1e-5, y=1, label=paste0("italic(r)^2", "==", round(r2,2)), hjust=0, parse=T) +
    xlab("Observed") +
    ylab("Simulated") +
    ggtitle("Metabolite mass fractions") +
    theme_classic()
  return(p)
}

### Plot mass fraction correlation during optimization ###
plot_mass_fractions_optimization <- function( mf_evol_data )
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

### Plot predicted proteomics ###
plot_predicted_proteomics <- function( pr_data, R2, r2 )
{
  p = ggplot(pr_data, aes(obs_fraction, sim_fraction)) +
    geom_abline(slope=1, intercept=0, color="pink") +
    geom_line(aes(obs_fraction, obsp3), color="grey", lty=2) +
    geom_line(aes(obs_fraction, obsm3), color="grey", lty=2) +
    geom_line(aes(obs_fraction, obsp10), color="grey", lty=3) +
    geom_line(aes(obs_fraction, obsm10), color="grey", lty=3) +
    geom_point() +
    #geom_smooth(method="lm") +
    scale_x_log10() + scale_y_log10() +
    annotate("text", x=0.001, y=1, label=paste0("italic(R)^2", "==", round(R2,5)), hjust=0, parse=T) +
    annotate("text", x=0.001, y=0.1, label=paste0("italic(r)^2", "==", round(r2,5)), hjust=0, parse=T) +
    #geom_text_repel(aes(label=id), size = 3.5) +
    xlab("Observed") +
    ylab("Simulated") +
    ggtitle("Proteomics") +
    theme_classic()
  return(p)
}

### Plot proteomics correlation during optimization ###
plot_proteomics_optimization <- function( pr_evol_data )
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

#---------------------------------#
# 1) Define main model properties #
#---------------------------------#
data_path  = "../../gbapy/tutorials/MMSYN_tutorial"
model_name = "mmsyn_fcr_v2"

#---------------------------------#
# 2) Load simulation data         #
#---------------------------------#
d_state = read.table(paste0(data_path,"/output/optimization/",model_name,"_state_optimum.csv"), h=T, sep=";", check.names=F)
d_f     = read.table(paste0(data_path,"/output/optimization/",model_name,"_f_optimum.csv"), h=T, sep=";", check.names=F)
d_v     = read.table(paste0(data_path,"/output/optimization/",model_name,"_v_optimum.csv"), h=T, sep=";", check.names=F)
d_p     = read.table(paste0(data_path,"/output/optimization/",model_name,"_p_optimum.csv"), h=T, sep=";", check.names=F)
d_b     = read.table(paste0(data_path,"/output/optimization/",model_name,"_b_optimum.csv"), h=T, sep=";", check.names=F)
d_c     = read.table(paste0(data_path,"/output/optimization/",model_name,"_c_optimum.csv"), h=T, sep=";", check.names=F)

#---------------------------------#
# 3) Build the different datasets #
#---------------------------------#
cond           = read.table(paste0(data_path, "/models/", model_name, "/conditions.csv"), sep=";", h=T, check.names=F)
rownames(cond) = cond[,1]
cond           = cond[,2:ncol(cond)]
glc_vec        = as.vector(t(conditions["x_glc__D",]))
mu_obs         = read.table(paste0(data_path, "/data/wet_experiments/observed_mu.csv"), sep=";", h=T)
MF            = build_mass_fractions_data(data_path, d_b, 67)
MF_by_cond    = mass_fractions_by_condition(data_path, d_b)
PR            = build_proteomics_data(data_path, model_name, d_p, 67)
PR_by_cond    = proteomics_by_condition(data_path, model_name, d_p)
modeled_pfrac = calculate_modeled_proteome_fraction(data_path, model_name, d_p)
phi_obs       = calculate_phi_obs(data_path)



### Load the model results ###
d_state     = read.table(paste0(output_path, "/", model_name, "_state_optimum.csv"), h=T, sep=";", check.names=F)
d_b         = read.table(paste0(output_path, "/", model_name,"_b_optimum.csv"), h=T, sep=";", check.names=F)
d_p         = read.table(paste0(output_path, "/", model_name,"_p_optimum.csv"), h=T, sep=";", check.names=F)
d_c         = read.table(paste0(output_path, "/", model_name,"_c_optimum.csv"), h=T, sep=";", check.names=F)
d_state$glc = glc_vec

### Build final datasets ###
MF_data  = build_mass_fractions_data(d_b, dim(d_b)[1])
PR_data  = build_proteomics_data(model_path, model_name, d_p, dim(d_p)[1])
MF_cor   = mass_fractions_correlation(d_b, dim(d_b)[1])
PR_cor   = proteomics_correlation(model_path, model_name, d_p, dim(d_p)[1])
MF_cor_D = data.frame()
PR_cor_D = data.frame()
for (i in seq(1, dim(d_b)[1]))
{
  res      = mass_fractions_correlation(d_b, i)
  MF_cor_D = rbind(MF_cor_D, res)
  res      = proteomics_correlation(model_path, model_name, d_p, i)
  PR_cor_D = rbind(PR_cor_D, res)
}
MF_cor_D$condition  = d_b$condition
MF_cor_D$glc        = glc_vec
names(MF_cor_D)     = c("r2", "pval", "R2", "condition", "glc")
PR_cor_D$condition  = d_p$condition
PR_cor_D$glc        = glc_vec
names(PR_cor_D)     = c("r2", "pval", "R2", "condition", "glc")
growth_law_data     = build_growth_law_data(d_p)
growth_law_data$glc = glc_vec
growth_law_data$mu  = d_state$mu

### Make plots ###
p1 = ggplot(d_state, aes(glc, mu)) +
  geom_line() +
  geom_point(data=Davg, aes(glc, mu, color="Observed")) +
  scale_x_log10() +
  scale_y_log10() +
  xlab("Glucose concentration (g/L)") +
  ylab("Growth rate") +
  ggtitle(paste0("Growth rate (max=",round(max(d_state$mu),3),")")) +
  theme_classic() +
  theme(legend.position="none")

p2 = plot_predicted_protein_fraction(d_c, d_state)

p3 = plot_predicted_mass_fractions(MF_data, MF_cor[3], MF_cor[1])

p4 = plot_predicted_proteomics(PR_data, PR_cor[3], PR_cor[1])

p5 = ggplot(MF_cor_D, aes(glc, R2)) +
  geom_line() +
  scale_x_log10() +
  xlab("Glucose concentration (g/L)") +
  ylab(TeX("$R^2$")) +
  ggtitle(paste0("Metabolite mass fraction correlation (max=",round(MF_cor[3],3),")")) +
  theme_classic()

rib_prot = c("protein_0025", "protein_0027", "protein_0082", "protein_0137", "protein_0148", "protein_0149", "protein_0198", "protein_0199", "protein_0238", "protein_0294", "protein_0362", "protein_0365", "protein_0422", "protein_0482", "protein_0499", "protein_0500", "protein_0501", "protein_0930", "protein_0526", "protein_0540", "protein_0637", "protein_0638", "protein_0644", "protein_0646", "protein_0647", "protein_0648", "protein_0653", "protein_0654", "protein_0655", "protein_0656", "protein_0657", "protein_0658", "protein_0659", "protein_0660", "protein_0661", "protein_0662", "protein_0663", "protein_0664", "protein_0665", "protein_0666", "protein_0667", "protein_0668", "protein_0669", "protein_0670", "protein_0671", "protein_0672", "protein_0806", "protein_0807", "protein_0809", "protein_0810", "protein_0833", "protein_0932", "protein_0910")
dprot    = load_observed_proteomics()
obs_phi  = sum(filter(dprot, protein%in%rib_prot)$obs_mass)/sum(dprot$obs_mass)
obs_phi
p6 = ggplot(growth_law_data, aes(mu, Phi)) + #*0.3627)) +
  geom_line() +
  geom_point(aes(x=0.34, y=obs_phi, color="Observed Phi")) +
  #geom_point(aes(x=0.4, y=obs_phi+0.16274/0.54727, color="Observed Phi + rRNA")) +
  xlab("Growth rate") +
  ylab(TeX("$\\Phi$ (weighted by 0.36)")) +
  ggtitle("Growth law") +
  #scale_y_log10() +
  theme_classic() +
  theme(legend.position="none")
p6

plot_grid(p1, p2, p3, p4, p5, p6, ncol=3, labels="AUTO")
# ggplot(PR_cor_D, aes(glc, R2)) +
#   scale_x_log10() +
#   geom_point()
