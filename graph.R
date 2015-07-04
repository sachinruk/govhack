setwd("Documents/thesis/govHack/")
cancerData=read.csv(url("http://data.gov.au/dataset/05696f6f-6ff5-42a2-904f-af5e4d1f56f8/resource/7fbac314-4bf9-4601-b812-0307316ef5a4/download/acimcombinedcounts.csv"))

cancerCombined=cancerData[cancerData$Sex=="Persons" & cancerData$Type=="Incidence",] #focus on just male and female combined
#get rid of missing value ones
cancerCombined=cancerCombined[!is.na(cancerCombined$Age_25_to_29),]

#visualise all cancers across time in 2004
cancer_type=unique(cancerCombined$Cancer_Type)
cancer_type2=unique(cancerData$Cancer_Type)
cancer2004=cancerCombined[cancerCombined$Year=="2004",]
matplot(t(cancer2004)[5:22,],type = "l")
plot(t(cancer2004)[5,],type = "l") #interesting case

#aggregate data across cancers by year
cancerCombined$aggregate=rowSums(as.matrix(cancerCombined[,5:dim(cancerCombined)[2]]))
covData=data.frame(unique(cancerCombined$Year))
names(covData)="Year"
for (i in 1:length(cancer_type)){
  a=cancerCombined[cancerCombined$Cancer_Type==cancer_type[i],c("Year","aggregate")]
  covData=merge(covData,a,by="Year",all=TRUE)
}
names(covData)[2:length(names(covData))]=cancer_type
rownames(covData)=covData[,1]
covData[,1]=NULL
covData=t(covData)

#machine learning part
library(huge)
library(matlab)
set.seed(1)
covData.npn=huge.npn(covData)
out.npn = huge(covData.npn,method = "glasso", lambda = linspace(0.99,0.97,100))

adj_mat=out.npn$path[[100]]
rownames(adj_mat)=cancer_type
colnames(adj_mat)=cancer_type
ig<-graph.adjacency(adj_mat,mode="undirected")
V(ig)$color<-"grey"
V(ig)[degree(ig)<=5]$color<-"yellow"
V(ig)[degree(ig)>5]$color<-"green"
V(ig)[degree(ig)>10]$color<-"red"
tkplot(ig)

for (i in 1:100){
  adj_mat=as.matrix(out.npn$path[[i]])
  rownames(adj_mat)=cancer_type
  colnames(adj_mat)=cancer_type
  write.csv(adj_mat,file=paste("adj_mat",i,".csv",sep = ""))
}