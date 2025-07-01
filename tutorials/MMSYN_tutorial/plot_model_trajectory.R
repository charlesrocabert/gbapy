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

### Load the observed mass fractions data ###
load_observed_mass_fractions <- function( data_path )
{
  d           = read.table(paste0(data_path, "/data/manual_curation/MMSYN_mass_fractions.csv"), sep=";", h=T, check.names=F)
  d$Fraction  = d$Fraction*0.01
  rownames(d) = d$ID
  return(d)
}

### Build the mass fraction data ###
build_mass_fractions_data <- function( data_path, d_b, i )
{
  d_mf           = load_observed_mass_fractions(data_path)
  D              = d_b[i,-which(names(d_b)%in%c("condition", "iter", "t", "dt", "h2o"))]
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
mass_fractions_correlation <- function( data_path, d_b, i )
{
  D     = build_mass_fractions_data(data_path, d_b, i)
  reg   = lm(log10(D$sim_fraction)~log10(D$obs_fraction))
  r2    = summary(reg)$adj.r.squared
  pval  = summary(reg)$coefficients[,4][[2]]
  SSres = sum((log10(D$sim_fraction)-log10(D$obs_fraction))^2)
  M     = mean(log10(D$obs_fraction))
  SStot = sum((log10(D$obs_fraction)-M)^2)
  R2    = 1-SSres/SStot
  return(c(r2, pval, R2))
}

### Evolution of mass fraction prediction ###
mass_fractions_optimization <- function( data_path, d_b, step )
{
  iter     = c()
  cor_vec  = c()
  pval_vec = c()
  R2_vec   = c()
  for(i in seq(1, dim(d_b)[1], by=step))
  {
    iter     = c(iter, i)
    res      = mass_fractions_correlation(data_path, d_b, i)
    cor_vec  = c(cor_vec, res[1])
    pval_vec = c(pval_vec, res[2])
    R2_vec   = c(R2_vec, res[3])
  }
  if (iter[length(iter)] != dim(d_b)[1])
  {
    iter     = c(iter, dim(d_b)[1])
    res      = mass_fractions_correlation(data_path, d_b, dim(d_b)[1])
    cor_vec  = c(cor_vec, res[1])
    pval_vec = c(pval_vec, res[2])
    R2_vec   = c(R2_vec, res[3])
  }
  D = data.frame(iter, cor_vec, pval_vec, R2_vec)
  names(D) = c("index", "r2", "pval", "R2")
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
    reactions = M2[p_id,]
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
calculate_simulated_proteomics <- function( pcontributions, d_p, i )
{
  M1               = pcontributions[["original"]]
  M2               = pcontributions[["reduced"]]
  M                = M2
  X                = d_p[,-which(names(d_p)%in%c("condition", "iter", "t", "dt"))]
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
build_proteomics_data <- function( data_path, model_name, d_p, i )
{
  reaction_ids   = names(d_p[,-which(names(d_p)%in%c("condition", "iter", "t", "dt"))])
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
proteomics_correlation <- function( data_path, model_name, d_p, i )
{
  D     = build_proteomics_data(data_path, model_name, d_p, i)
  reg   = lm(log10(D$sim_fraction)~log10(D$obs_fraction))
  r2    = summary(reg)$adj.r.squared
  pval  = summary(reg)$coefficients[,4][[2]]
  SSres = sum((log10(D$sim_fraction)-log10(D$obs_fraction))^2)
  M     = mean(log10(D$obs_fraction))
  SStot = sum((log10(D$obs_fraction)-M)^2)
  R2    = 1-SSres/SStot
  return(c(r2, pval, R2))
}

### Evolution of proteomics prediction ###
proteomics_optimization <- function( data_path, model_name, d_p, step )
{
  iter     = c()
  cor_vec  = c()
  pval_vec = c()
  R2_vec   = c()
  for(i in seq(1, nrow(d_p), by=step))
  {
    iter     = c(iter, i)
    res      = proteomics_correlation(data_path, model_name, d_p, i)
    cor_vec  = c(cor_vec, res[1])
    pval_vec = c(pval_vec, res[2])
    R2_vec   = c(R2_vec, res[3])
  }
  if (iter[length(iter)] != dim(d_p)[1])
  {
    iter     = c(iter, dim(d_p)[1])
    res      = proteomics_correlation(data_path, model_name, d_p, dim(d_p)[1])
    cor_vec  = c(cor_vec, res[1])
    pval_vec = c(pval_vec, res[2])
    R2_vec   = c(R2_vec, res[3])
  }
  D = data.frame(iter, cor_vec, pval_vec, R2_vec)
  names(D) = c("index", "r2", "pval", "R2")
  return(D)
}

### Plot predicted growth rates ###
plot_predicted_growth_rate <- function( d_state, d_obs )
{
  last_mu = d_state$mu[dim(d_state)[1]]
  p = ggplot(d_state, aes(iter, mu)) +
    geom_line() +
    #geom_hline(yintercept=max(d_obs$mu), color="pink") +
    xlab("Iterations") +
    ylab("Growth rate") +
    ggtitle(paste0("Growth rate (\u03BC = ",round(last_mu,3),")")) +
    theme_classic()
  return(p)
}

### Plot predicted protein fraction ###
plot_predicted_protein_fraction <- function( d_c )
{
  proteins = c("ACP", "PdhC", "dUTPase", "Protein")
  dl                  = d_c[,-which(names(d_c)%in%c("condition","iter","t","dt", "h2o"))]
  dl$sum              = rowSums(dl)
  #dl$Protein_fraction = rowSums(dl[,proteins])/dl$sum
  dl$Protein_fraction = (dl[,"Protein"])/dl$sum
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

### Plot the ribosome fraction ###
plot_predicted_ribosome_fraction <- function( d_p, phi_obs )
{
  dl                  = d_p[,-which(names(d_p)%in%c("condition","iter","t","dt"))]#, "ATPase"))]
  dl$sum              = rowSums(dl)
  dl$Ribosome_fraction = dl[,"Ribosome"]/dl$sum
  dl$iter             = d_p$iter
  last_ribosome_fraction  = dl$Ribosome_fraction[dim(dl)[1]]
  p = ggplot(dl, aes(iter, Ribosome_fraction)) +
    geom_hline(yintercept=phi_obs, col="pink") +
    geom_line() +
    xlab("Iterations") +
    ylab("Ribosome fraction") +
    ggtitle(paste0("Ribosome fraction (Rf = ",round(last_ribosome_fraction,3),")")) +
    ylim(0,1) +
    theme_classic()
  return(p)
}

### Plot the mass fraction of housekeeping proteins ###
plot_predicted_housekeeping_fraction <- function( d_p, modeled_pfrac )
{
  dl       = d_p[,-which(names(d_p)%in%c("condition", "iter", "t", "dt"))]
  psum     = rowSums(dl)
  atpR     = dl$ATPase/psum
  D        = data.frame(d_p$iter, atpR)
  print(paste0("> Housekeeping fraction = ", atpR[length(atpR)]))
  names(D) = c("iter", "fraction")
  p = ggplot(D, aes(iter, fraction)) +
    geom_hline(yintercept=1-modeled_pfrac, col="pink") +
    geom_line() +
    xlab("Iteration") + ylab("Housekeeping proteins fraction") +
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
condition  = 67

#---------------------------------#
# 2) Load simulation data         #
#---------------------------------#
d_state = read.table(paste0(data_path,"/output/optimization/",model_name,"_",condition,"_state_trajectory.csv"), h=T, sep=";", check.names=F)
d_f     = read.table(paste0(data_path,"/output/optimization/",model_name,"_",condition,"_f_trajectory.csv"), h=T, sep=";", check.names=F)
d_v     = read.table(paste0(data_path,"/output/optimization/",model_name,"_",condition,"_v_trajectory.csv"), h=T, sep=";", check.names=F)
d_p     = read.table(paste0(data_path,"/output/optimization/",model_name,"_",condition,"_p_trajectory.csv"), h=T, sep=";", check.names=F)
d_b     = read.table(paste0(data_path,"/output/optimization/",model_name,"_",condition,"_b_trajectory.csv"), h=T, sep=";", check.names=F)
d_c     = read.table(paste0(data_path,"/output/optimization/",model_name,"_",condition,"_c_trajectory.csv"), h=T, sep=";", check.names=F)

#---------------------------------#
# 3) Build the different datasets #
#---------------------------------#
MF            = build_mass_fractions_data(data_path, d_b, dim(d_b)[1])
MF_optim      = mass_fractions_optimization(data_path, d_b, 10)
PR            = build_proteomics_data(data_path, model_name, d_p, dim(d_p)[1])
PR_optim      = proteomics_optimization(data_path, model_name, d_p, 10)
modeled_pfrac = calculate_modeled_proteome_fraction(data_path, model_name, d_p)
phi_obs       = calculate_phi_obs(data_path)

#---------------------------------#
# 4) Build the figures            #
#---------------------------------#
p1 = plot_predicted_growth_rate(d_state, d_obs)
p2 = plot_predicted_protein_fraction(d_c)
p3 = plot_predicted_ribosome_fraction(d_p, phi_obs)
p4 = plot_predicted_mass_fractions(MF, last(MF_optim$R2), last(MF_optim))
p5 = plot_predicted_proteomics(PR, last(PR_optim$R2), last(PR_optim$r2))

p_mf = plot_mass_fractions_optimization(MF_optim)
p_pr = plot_proteomics_optimization(PR_optim)
p7 = ggplot(d_f, aes(iter, L_LACt2r)) +
  geom_line() +
  xlab("Iteration") + ylab("Lactate excretion") +
  theme_classic()
p8 = ggplot(d_f, aes(iter, GLCpts)) +
  geom_line() +
  xlab("Iteration") + ylab("Glucose import") +
  theme_classic()
p10 = ggplot(d_p, aes(iter, ATPase)) +
  geom_line() +
  xlab("Iteration") + ylab("p(ATPase)") +
  theme_classic()
p12 = ggplot(d_c, aes(iter, ACP)) +
  geom_line() +
  geom_hline(yintercept=0.018*349*0.01, col="pink") +
  ggtitle(d_c[nrow(d_c),"ACP"]) +
  xlab("Iteration") + ylab("ACP") +
  theme_classic()
p13 = ggplot(d_c, aes(iter, dUTPase)) +
  geom_hline(yintercept=0.003*349*0.01, col="pink") +
  geom_line() +
  ggtitle(d_c[nrow(d_c),"dUTPase"]) +
  xlab("Iteration") + ylab("dUTPase") +
  theme_classic()
p11 = plot_predicted_housekeeping_fraction(d_p, 1-0.5)
#plot_grid(p1, p4)
plot_grid(p1, p2, p3, p4, p_mf[[3]], p5, p_pr[[3]], p11, ncol=2)

#MF[,c("id", "sim_fraction", "obs_fraction")]
#filter(MF, Category=="Macromolecules")

X1 = tapply(MF$sim_fraction, MF$Category, sum)
X2 = tapply(MF$obs_fraction, MF$Category, sum)
#X1
#X2
# par(mfrow=c(1,2))
# barplot(X1, ylim=c(0,1), las=2, main="Simulated")
# barplot(X2, ylim=c(0,1), las=2, main="Observed")
# sum(D$obs_fraction)
