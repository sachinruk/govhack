#strip results csv into multiple files
setwd("Documents/thesis/govHack/")
#download.file("http://data.gov.au/dataset/05696f6f-6ff5-42a2-904f-af5e4d1f56f8/resource/7fbac314-4bf9-4601-b812-0307316ef5a4/download/acimcombinedcounts.csv",destfile = "data.csv")
cancerData=read.csv("data.csv")
predictions=read.csv("results.csv")
cancerData$aggregate=rowSums(cancerData[,5:23])
cancerData=cancerData[,-c(5:23)]

#pivot the tables where each row is per gender per type
