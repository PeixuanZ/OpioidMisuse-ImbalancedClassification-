setwd("C:/PSU/Courses/Fall2020/IE582/Project/Codes")
library(tidyverse)
library(gridExtra)
library(knitr)
library(dplyr)
library(ggplot2)
library(readxl)
dat1 <- read_xlsx("AUCvalues.xlsx",sheet = 1)
dat2 <- read_xlsx("AUCvalues.xlsx",sheet = 2)


# boxplot

ggplot(dat1, aes(x = Model, y = AUC, fill = Model))+geom_boxplot()
ggplot(dat2, aes(x = Model, y = AUC, fill = Model))+geom_boxplot()

dat11 = dat1 %>% group_by(Model) %>% summarise(mean = mean(AUC), sd = sd(AUC)) %>% mutate(type = 'Oversampling')
dat22 = dat2 %>% group_by(Model) %>% summarise(mean = mean(AUC), sd = sd(AUC)) %>% mutate(type = 'Standard')

dat = rbind(dat11,dat22)
# Error bars represent standard error of the mean
ggplot(dat, aes(x=reorder(Model,-mean), y=mean, fill=type)) + 
  geom_bar(position=position_dodge(), stat="identity") +
  geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd),
                width=.1,                    # Width of the error bars
                position=position_dodge(.9)) + labs(x = "ML Models", y = "AUC", title = "Model Performance (AUC values)") 
