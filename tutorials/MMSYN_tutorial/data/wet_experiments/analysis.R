#!/usr/bin/Rscript
# coding: utf-8

library("tidyverse")
library("cowplot")
library("rstudioapi")

directory = dirname(getActiveDocumentContext()$path)
setwd(directory)


##################
#      MAIN      #
##################

OD_SPAN      = 0.2
OD_DIFF_SPAN = 0.2
T_MIN        = 5
T_MAX        = 50
OD_FACTOR    = 0.65

### Load the OD data ###
d1              = read.table("long_data.tsv", h=T, sep="\t")
d1$CMRL         = rep("100", nrow(d1))
d1$glucose      = d1$glucose+0.4
d2              = read.table("long_data_2.tsv", h=T, sep="\t")
d2$Well         = paste0(d2$Well, "_2")
d               = rbind(d1, d2)
#d               = d2
d$OD[d$OD<=0.0] = 0.001
d$ln_OD         = log(d$OD)
d$glc_fact      = as.factor(d$glucose)

### Check time boundaries ###
# ggplot(d, aes(Time, ln_OD)) +
#   geom_vline(xintercept=T_MIN, col="red", lwd=2) +
#   geom_vline(xintercept=T_MAX, col="green", lwd=2) +
#   geom_line() +
#   facet_wrap(~Well) +
#   theme_classic()

d = filter(d, Time >= T_MIN & Time <= T_MAX)

### Calculate the smoothing for each well ###
for(Well in unique(d$Well)){
  d$OD_smooth[d$Well==Well] = loess(ln_OD ~ Time, d[d$Well==Well,], span=OD_SPAN, family="gaussian")$fitted
  d$OD_diff[d$Well==Well]   = diff((d$OD_smooth[d$Well==Well]))/diff(d$Time[d$Well==Well])
  d$OD_diff_smooth[d$Well==Well] = loess(OD_diff ~ Time, d[d$Well==Well,], span=OD_DIFF_SPAN, family="gaussian")$fitted
  d$OD_diff_rel[d$Well==Well] = d$OD_diff_smooth[d$Well==Well]/max(d$OD_diff_smooth[d$Well==Well])
}

well_vec = c()
glc_vec  = c()
mu_vec   = c()
OD_plots = list()
for(well in unique(d$Well))
{
  dl        = filter(d, Well==well)
  max_diff  = max(dl$OD_diff_smooth)
  max_index = which(dl$OD_diff_smooth==max(dl$OD_diff_smooth))
  dl2       = filter(dl, OD_diff_smooth>=max_diff*OD_FACTOR)
  min_time  = min(dl2$Time)
  max_time  = max(dl2$Time)
  #reg       = lm(dl2$OD_smooth~dl2$Time)
  reg       = lm(dl2$ln_OD~dl2$Time)
  well_vec  = c(well_vec, well)
  glc_vec   = c(glc_vec, dl2$glucose[1])
  mu_vec    = c(mu_vec, coef(reg)[[2]])
  p1 = ggplot(dl) +
    geom_vline(xintercept=min_time, col="cornflowerblue", lty=2) +
    geom_vline(xintercept=max_time, col="cornflowerblue", lty=2) +
    #geom_smooth(data=dl2, aes(Time, ln_OD), method="lm", se=F, col="cornflowerblue") +
    geom_line(aes(Time, ln_OD)) +
    geom_line(data=dl2, aes(Time, ln_OD), col="blue") +
    xlab("Time (h)") +
    ylab("OD") +
    ggtitle(paste0("Well ", well)) +
    theme_classic()
  p2 = ggplot(dl) +
    #geom_vline(xintercept=dl$Time[max_index], col="blue") +
    geom_hline(yintercept=max_diff*OD_FACTOR, col="blue") +
    geom_line(aes(Time, OD_diff)) +
    geom_line(aes(Time, OD_diff_smooth), col="red") +
    xlab("Time (h)") +
    ylab("OD") +
    theme_classic()
  OD_plots[[paste(well,"OD")]]      = p1
  OD_plots[[paste(well,"OD diff")]] = p2
}
D = data.frame(well=well_vec, glc=glc_vec, mu=mu_vec)

# ggplot(D, aes(as.factor(glc), mu)) +
#   geom_hline(yintercept=mean((mu_vec)), col="red") +
#   geom_boxplot() +
#   theme_classic()
# 
# ggplot(D, aes((glc), mu)) +
#   geom_hline(yintercept=mean((mu_vec)), col="red") +
#   geom_point() +
#   scale_x_log10() +
#   theme_classic()

write.table(D, "observed_mu.csv", sep=";", row.names=F, col.names=T, quote=F)

p1 = plot_grid(plotlist=OD_plots[1:20], ncol=4)
p2 = plot_grid(plotlist=OD_plots[21:40], ncol=4)
p3 = plot_grid(plotlist=OD_plots[41:60], ncol=4)
p4 = plot_grid(plotlist=OD_plots[61:80], ncol=4)
p5 = plot_grid(plotlist=OD_plots[81:100], ncol=4)
p6 = plot_grid(plotlist=OD_plots[101:120], ncol=4)
p7 = plot_grid(plotlist=OD_plots[121:132], ncol=4, nrow=5)

ggsave("OD_plots_1.pdf", p1, width=12, height=9)
ggsave("OD_plots_2.pdf", p2, width=12, height=9)
ggsave("OD_plots_3.pdf", p3, width=12, height=9)
ggsave("OD_plots_4.pdf", p4, width=12, height=9)
ggsave("OD_plots_5.pdf", p5, width=12, height=9)
ggsave("OD_plots_6.pdf", p6, width=12, height=9)
ggsave("OD_plots_7.pdf", p7, width=12, height=9)

print(paste0("> Nb wells=",length(OD_plots)/2))
print(paste0("> Average mu=", mean((mu_vec))))
