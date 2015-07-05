#strip results csv into multiple files
setwd("Documents/thesis/govHack/")
#download.file("http://data.gov.au/dataset/05696f6f-6ff5-42a2-904f-af5e4d1f56f8/resource/7fbac314-4bf9-4601-b812-0307316ef5a4/download/acimcombinedcounts.csv",destfile = "data.csv")
cancerData=read.csv("data.csv")
predictions=read.csv("results.csv")
cancerData$aggregate=rowSums(cancerData[,5:23])
cancerData=cancerData[,-c(5:23)]

#pivot the tables where each row is per gender per type
cancer_type=unique(cancerData$Cancer_Type)
gender=unique(cancerData$Sex)
status=unique(cancerData$Type)
res_combined=data.frame(unique(cancerData$Year))
names(res_combined)="Year"
for (k in 1:length(status))
  for (j in 1:length(gender))
    for (i in 1:length(cancer_type)){
      a=cancerData[cancerData$Cancer_Type==cancer_type[i] &
                     cancerData$Sex==gender[j] &
                     cancerData$Type==status[k],c("Year","aggregate")]
      res_combined=merge(res_combined,a,by="Year",all=TRUE)
    }

res_combined=res_combined[15:43,]
res_combined=t(res_combined)
colnames(res_combined)=as.character(res_combined[1,])
res_combined=res_combined[-1,]

#combine with thushans
res_combined=as.data.frame(res_combined)
extra_years=as.character(names(predictions[1,9:18]))
res_combined[,extra_years]=NA
#res_combined=as.matrix(res_combined)
#for (i in 1:63)
#  people=rowSums(as.numeric(predictions[(2*i-1):(2*i),]))

l=1
for (k in 1:length(status))
  for (j in 1:length(gender))
    for (i in 1:length(cancer_type)){
      
      preds=predictions[predictions$Cancer==cancer_type[i] & 
        predictions$Gender==as.character(gender[j]) &
        predictions$Status==status[k],9:18]
      if (dim(preds)[1]) #if there exists any dimensions
        res_combined[l,extra_years]=preds
      if (gender[j]=="Persons")
        
      l=l+1;
    }

res_combined=t(res_combined)

rownames(res_combined)=gsub("X","",rownames(res_combined))

#write tsv file
path="cancer_tsvs/"
l=1;
for (k in 1:length(status))
  for (j in 1:length(gender))
    for (i in 1:length(cancer_type)){
      cancer=gsub(" ","_",cancer_type[i]);
      gender_=substr(gender[j],1,1)
      status_=substr(status[k],1,1)
      filename=paste(path,cancer,"__",gender_,status_,".tsv",sep="")
      a=as.matrix(res_combined[,l])
      a <- cbind(Row.Names = rownames(a), a)
      colnames(a)=c("year","numdeaths")
      write.table(a, file=filename, quote=FALSE, sep='\t',row.names = FALSE)
      l=l+1
    }

