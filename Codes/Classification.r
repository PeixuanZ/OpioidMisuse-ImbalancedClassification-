setwd("C:/PSU/Courses/Fall2020/IE582/Project/Codes")
# packages
library(tidyverse)
library(gridExtra)
library(knitr)
library(dplyr)
library(ggplot2)

NSDUH2019 <- read.csv("NSDUH_2019.CSV",header = T)
names(NSDUH2019) <- toupper(names(NSDUH2019))

# independent variables
fea <- c("AGE2", "SEXIDENT", "IRMARIT", "WRKSTATWK2", "INCOME", "COUTYP4","EDUSCHLGO", "NEWRACE2", "BOOKED",
         "HEALTH","HRTCONDEV","DIABETEVR", "COPDEVER","CIRROSEVR", "HEPBCEVER","KIDNYDSEV", "ASTHMAEVR",
         "HIVAIDSEV","CANCEREVR","HIGHBPEVR",
         "K6SCMON","ADDPREV","SUICPLAN", "SUICTHNK","SUICTRY", "TXEVRRCVD","AMDEYR","CIGEVER",
         "ALCEVER","MJEVER","COCEVER","HEREVER","MEDICARE","CAIDCHIP","PRVHLTIN","UDPYOPI",
         "ABODALC","ABODMRJ","ABODCOC","ABODHER")

dat <- NSDUH2019[,fea] %>% filter(AGE2>6)
target <- NSDUH2019%>% filter(AGE2>6) %>% select(OPINMYR)

# missing values

for (i in c(2:4,7,10:20,22:40)){
  dat[,i] = ifelse(dat[,i]<10, dat[,i],NA)
}

missing.values <- dat %>%
  gather(key = "key", value = "val") %>%
  mutate(isna = is.na(val)) %>%
  group_by(key) %>%
  mutate(total = n()) %>%
  group_by(key, total, isna) %>%
  summarise(num.isna = n()) %>%
  mutate(pct = num.isna / total * 100)


levels <-
  (missing.values  %>% filter(isna == T) %>% arrange(desc(pct)))$key

percentage.plot <- missing.values %>%
  ggplot() +
  geom_bar(aes(x = reorder(key, desc(pct)), 
               y = pct, fill=isna), 
           stat = 'identity', alpha=0.8) +
  scale_x_discrete(limits = levels) +
  scale_fill_manual(name = "", 
                    values = c('steelblue', 'tomato3'), labels = c("Present", "Missing")) +
  coord_flip() +
  labs(title = "Percentage of missing values", x =
         'Variable', y = "% of missing values")

percentage.plot

# transformation: AGE2, other variables(too many missing values)

dat$AGE2_new = rep(NA, nrow(dat))
for (j in 1:nrow(dat)){
  if (dat$AGE2[j]<=12&dat$AGE2[j]>6){
    dat$AGE2_new[j] = "18-25"
  }
  else if (dat$AGE2[j]<=14&dat$AGE2[j]>12){
    dat$AGE2_new[j] = "25-35"
  }
  else if (dat$AGE2[j]<=16&dat$AGE2[j]>14){
    dat$AGE2_new[j] = "35-65"
  }
  else{dat$AGE2_new[j] = ">65"}
}

# suicidal related
fea2 <-  c( "HRTCONDEV","DIABETEVR", "COPDEVER","CIRROSEVR", "HEPBCEVER","KIDNYDSEV", "ASTHMAEVR",
         "HIVAIDSEV","CANCEREVR","HIGHBPEVR","SUICPLAN", "SUICTHNK","SUICTRY", "TXEVRRCVD","AMDEYR","CIGEVER",
         "ALCEVER","MJEVER","COCEVER","HEREVER","MEDICARE","CAIDCHIP","PRVHLTIN","UDPYOPI",
         "ABODALC","ABODMRJ","ABODCOC","ABODHER", "ADDPREV", "EDUSCHLGO", "BOOKED")

for (k in fea2){
  dat[,k] <- ifelse(is.na(dat[,k]), 2, dat[,k])
  dat[,k] <- ifelse(dat[,k]==1,1,0)
}

dat_miss = na.omit(cbind(dat[,2:41], target))

for (i in c(1:19, 21:41)){
  dat_miss[,i] = as.factor(dat_miss[,i])
}


# zelig
library(Zelig)
library(caret)
# Folds are created on the basis of target variable
folds <- createFolds(factor(dat_miss$OPINMYR), k = 10, list = FALSE)

# cross validation
library(pROC)
for (i in 1:10){
  test = dat_miss[folds==i,]
  train = dat_miss[folds!=i,]
  m1 <- zelig(OPINMYR ~., model = "logit", data = train, cite = FALSE)
  pred <- predict(m1, test, type = "response")
  pred_test = ifelse(pred[[1]]>0.05,1,0)
  
  roc_obj <- roc(test$OPINMYR, pred_test)
  print(auc(roc_obj))
}


# EDA
library(tableone)
dat = read.csv("data2019.csv", stringsAsFactors = TRUE)[,2:42]
for (i in c(1:19, 21:41)){
  dat[,i] = as.factor(dat[,i])
}
vars <- c("AGE2_new", "SEXIDENT", "IRMARIT", "WRKSTATWK2", "INCOME", "COUTYP4","EDUSCHLGO", "NEWRACE2", "BOOKED",
          "HEALTH","HRTCONDEV","DIABETEVR", "COPDEVER","CIRROSEVR", "HEPBCEVER","KIDNYDSEV", "ASTHMAEVR",
          "HIVAIDSEV","CANCEREVR","HIGHBPEVR",
          "K6SCMON","ADDPREV","SUICPLAN", "SUICTHNK","SUICTRY", "TXEVRRCVD","AMDEYR","CIGEVER",
          "ALCEVER","MJEVER","COCEVER","HEREVER","MEDICARE","CAIDCHIP","PRVHLTIN","UDPYOPI",
          "ABODALC","ABODMRJ","ABODCOC","ABODHER")
Data_unmatched <- CreateTableOne(vars = vars, strata = "OPINMYR", data = dat, test = TRUE)
output = print(Data_unmatched, smd = T)
write.csv(output,"tab1.csv")
