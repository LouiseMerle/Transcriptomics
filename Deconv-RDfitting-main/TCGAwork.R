# LOAD NEEDED LIBRARIES
library(data.table)
library(jsonlite)
library(ggplot2)
library(ggpubr)
library(Biobase)
library(NMF)
library(SpNMF)

#TCGA.dff is a data frame with gene expression values from TCGA for GBM
#Measured genes (rows) are limited to the same genes that are measured in the organoids

#Firstly log-transform because of the distribution
TCGA.dff <- log2(TCGA.dff)
is.na(TCGA.dff)<-sapply(TCGA.dff, is.infinite)
TCGA.dff[is.na(TCGA.dff)]<-0

#Remove genes with low values
rows.Sum <- rowSums(TCGA.dff)
rows.0 <- which(rows.Sum == 0)
TCGA.dff$genes<-rownames(TCGA.dff)
TCGA.dff<-TCGA.dff[-rows.0, ]

if(requireNamespace("Biobase", quietly=TRUE)){
  # perform 40 runs for each value of r in range 2:8 for each method
  estim.log.nmf.brunet <- nmf(TCGA.dff, 2:8, nrun=50, method = 'brunet', seed=123456, .options='tp')
  estim.log.nmf.offset <- nmf(TCGA.dff, 2:8, nrun=50, method ='offset', seed=123456, .options='tp')
  estim.log.nmf.lee <- nmf(TCGA.dff, 2:8, nrun=50, method ='lee', seed=123456, .options='tp')
  estim.log.nmf.ns <- nmf(TCGA.dff, 2:8, nrun=50, method ='nsNMF', seed=123456, .options='tp')
  estim.log.nmf.als <- nmf(TCGA.dff, 2:8, nrun=50, method ='snmf', seed=123456, .options='tp')
}

plt.log.brun <- plot(estim.log.nmf.brun)
plt.log.lee <- plot(estim.log.nmf.lee)
plt.log.ns <- plot(estim.log.nmf.ns)
plt.log.off <- plot(estim.log.nmf.offset)
plt.log.als <- plot(estim.log.nmf.als)

############## Extract the Cell types from NMF results ####################
nmf.off.3 <- estim.log.nmf.offset$fit$`3`
nmf.off.3.H <- coefficients(nmf.off.3)
nmf.off.3.W <- basis(nmf.off.3)
rownames(nmf.off.3.W)<-rownames(TCGA.dff)

nmf.off.3.W[nmf.off.3.W > 0.01] <- 1
nmf.off.3.W[nmf.off.3.W <= 0.01] <- 0

colnames(nmf.off.3.W)<-c("V1", "V2", "V3")

nmf.off.3.ct1 <- dplyr::filter(as.data.frame(nmf.off.3.W), V1 ==1)
nmf.off.3.ct2 <- dplyr::filter(as.data.frame(nmf.off.3.W), V2 ==1)
nmf.off.3.ct3 <- dplyr::filter(as.data.frame(nmf.off.3.W), V3 ==1)

ct.off.1 <- rownames(nmf.off.3.ct1)
ct.off.2 <- rownames(nmf.off.3.ct2)
ct.off.3 <- rownames(nmf.off.3.ct3)

W.off <- basis(estim.log.nmf.offset$fit$`3`)
rownames(W.off)<-rownames(TCGA.dff)
colnames(W.off)<-c("V1","V2","V3")
H.off <- coef(estim.log.nmf.offset$fit$`3`)

# Order both matrices according to the gene names.
W.off <- W.off[order(rownames(W.off)), ]

# Approach taking only genes uniqely expressed in a given cell type

# X = W * H is to be solved.
# Where X is ORG.sorted, W is cell.types.als and H is unknown.
# In order to do it, generalized inverse of W will be calculated,
# and multiplied on both sides of the quation from left. 
library(MASS)

X <- subset(ORG.df, rownames(ORG.df) %in% rownames(W.als))
#X <- subset(ORG.df, rownames(ORG.df) %in% rownames(W.off))

# Order both matrices according to the gene names.
X <- as.matrix(X[order(rownames(X)), ])
inv.W.off <- ginv(W.off)
H <- abs(inv.W.off %*% X)

#Transform the H matrix such that each column sums up to 1.
for (j in 1:ncol(H)){
  col <- H[,j]
  scol <- sum(H[,j])
  col<-col/scol
  H[,j] <- col
}

H<-H[,order(colnames(H))]
props <- as.data.frame(t(H))
props$file <- rownames(props)
props.wide<-props

library(tidyr)
# Reshape the information to easily get a nice plot with changes in time.
props <- gather(props, key = "CellType", value = "Proportions", -file)
props$Patient <- cut.patient(props$file)
props$Time <- cut.time(props$file)
props$Treatment <- cut.treatment(props$file)
props$file<-NULL

props.wide$Patient <- cut.patient(props.wide$file)
props.wide$Time <- cut.time(props.wide$file)
props.wide$Treatment <- cut.treatment(props.wide$file)
props.wide$file<-NULL

# Replace time values, 0 means before any tratment
# 1 at time zero after applying the treatment
# 2 after 24 hours and 3 after 72 hours

props <- within(props, Time[Time == '72'] <- '3')
props <- within(props, Time[Time == '24'] <- '2')
props <- within(props, Time[Time == '0' & Treatment != '0'] <- '1')

# Delete data into two subsets depending on the applied treatment.
props.4 <- props[(props$Treatment == "0" | props$Treatment == "4") & props$Patient =="3565",]
props.10 <- props[(props$Treatment == "0" | props$Treatment == "10") & props$Patient =="3565",]

plt1 <- ggplot(props.4) + geom_point(aes(y = Proportions, x = Time, shape =Patient , color = CellType)) + labs(x = "Time points", y = "Proportions")
plt2 <- ggplot(props.10) + geom_point(aes(y = Proportions, x = Time, shape = Patient, color = CellType)) + labs(x = "Time points", y = "Proportions")
plot.data<-ggarrange(plt1, plt2, labels=c("Rx4","Rx10"), ncol=2,nrow=1, common.legend = TRUE, legend="bottom")
plot.data
