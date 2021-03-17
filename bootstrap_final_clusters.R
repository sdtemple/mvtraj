library(here)
here()

clust.6 <- read.csv(here("Project","march8","6-0.001","final-mb-clusters-6-0.001.csv"))

replace.zero.with.na <- function(x) ifelse(x==0,NA,x)
ids <- lapply(1:6,function(x) unique(subset(clust.6,cluster==x)$patientID))
ps <- sapply(ids,function(x) mean(sapply(x,function(y) !is.na(subset(clust.6,patientID==y)$agvhday[1])),na.rm=T))
ds <- sapply(ids,function(x) mean(sapply(x,function(y) subset(clust.6,patientID==y)$agvhday[1]),na.rm=T))
gs <- sapply(ids,function(x) mean(sapply(x,function(y) replace.zero.with.na(subset(clust.6,patientID==y)$agvhgrd[1])),na.rm=T))
ls <- sapply(ids,length)

barplot(ps)
barplot(ds)
barplot(gs)

ps
ds
gs
ls

patients <- lapply(unique(clust.6$patientID),function(x) subset(clust.6,patientID==x))
boot.diff <- function(funcs,lens) {
  ms <- array(dim=c(length(lens),length(funcs)))
  ids.remaining <- 1:length(unique(clust.6$patientID))
  for(i in 1:length(lens)) {
    len <- lens[i]
    which.ids <- sample(x=ids.remaining,size=len)
    ids.remaining <- ids.remaining[!(ids.remaining %in% which.ids)]
    for(j in 1:length(funcs)){
      func <- funcs[[j]]
      ms[i,j] <- mean(sapply(which.ids,function(x) func(patients[[x]])),na.rm=T)
    }
  }

  return(apply(ms,c(2),function(x) max(x) - min(x)))
}

diff.p <- max(ps) - min(ps)
diff.d <- max(ds) - min(ds)
diff.g <- max(gs) - min(gs)
diffs <- c(diff.p,diff.d,diff.g)
fs <- c(function(x) !is.na(x$agvhday[1]),function(x) x$agvhday[1],function(x) replace.zero.with.na(x$agvhgrd[1]))
nsim <- 10000
boots.pdg <- replicate(nsim,boot.diff(fs,ls))

mean(boots.pdg[1,] > diff.p) # 0.3
mean(boots.pdg[2,] > diff.d) # 0.17
mean(boots.pdg[3,] > diff.g) # 0.04








clust.10 <- read.csv(here("Project","march8","10-0.001","final-mb-clusters-10-0.001.csv"))

replace.zero.with.na <- function(x) ifelse(x==0,NA,x)
ids <- lapply(1:10,function(x) unique(subset(clust.10,cluster==x)$patientID))
ps <- sapply(ids,function(x) mean(sapply(x,function(y) !is.na(subset(clust.10,patientID==y)$agvhday[1])),na.rm=T))
ds <- sapply(ids,function(x) mean(sapply(x,function(y) subset(clust.10,patientID==y)$agvhday[1]),na.rm=T))
gs <- sapply(ids,function(x) mean(sapply(x,function(y) replace.zero.with.na(subset(clust.10,patientID==y)$agvhgrd[1])),na.rm=T))
ls <- sapply(ids,length)

barplot(ps)
barplot(ds)
barplot(gs)

ps
ds
gs
ls

patients <- lapply(unique(clust.10$patientID),function(x) subset(clust.10,patientID==x))
boot.diff <- function(funcs,lens) {
  ms <- array(dim=c(length(lens),length(funcs)))
  ids.remaining <- 1:length(unique(clust.10$patientID))
  for(i in 1:length(lens)) {
    len <- lens[i]
    which.ids <- sample(x=ids.remaining,size=len)
    ids.remaining <- ids.remaining[!(ids.remaining %in% which.ids)]
    for(j in 1:length(funcs)){
      func <- funcs[[j]]
      ms[i,j] <- mean(sapply(which.ids,function(x) func(patients[[x]])),na.rm=T)
    }
  }

  return(apply(ms,c(2),function(x) max(x) - min(x)))
}

diff.p <- max(ps) - min(ps)
diff.d <- max(ds) - min(ds)
diff.g <- max(gs) - min(gs)
diffs <- c(diff.p,diff.d,diff.g)
fs <- c(function(x) !is.na(x$agvhday[1]),function(x) x$agvhday[1],function(x) replace.zero.with.na(x$agvhgrd[1]))
nsim <- 100000
boots.pdg <- replicate(nsim,boot.diff(fs,ls))

mean(boots.pdg[1,] > diff.p,na.rm=T) # 0.02
mean(boots.pdg[2,] > diff.d,na.rm=T) # 0.87
mean(boots.pdg[3,] > diff.g,na.rm=T) # 0.421

