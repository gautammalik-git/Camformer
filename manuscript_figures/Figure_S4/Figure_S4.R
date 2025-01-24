
# Specify which metric to run this for (need to run twice to reproduce all plots)
#metric = "r"
metric = "rho"

get_mylist = function(subset,metric) {
  # subset - list of TRUE/FALSE
  # metric - "r" or "rho"
  mylist = c(); pos = 0; at = c()
  tests = c("Seq_Enc","LossFn","Optimizer","LRScheduler","Model")
  for (test in tests) {
    pos = pos+1 # Increase position counter
    column = which(colnames(dat) == test)
    groups = unique(dat[,column])
    groups = groups[order(sapply(1:length(groups),function(x) median(dat[subset & dat[,column]==groups[x],metric])),decreasing = FALSE)]
    for (g in groups) {
      mylist[[g]] = dat[subset & dat[,column] == g,metric]
      at = c(at,pos)
      pos = pos+1
    }
  }
  return(list(mylist, at))
}

get_stats = function(mylist, grpA, grpB) {
  p = wilcox.test(mylist[[grpA]],mylist[[grpB]], paired = TRUE)$p.value
  lengthA = length(mylist[[grpA]]); lengthB = length(mylist[[grpB]])
  meanA = mean(mylist[[grpA]]); sdA = sd(mylist[[grpA]]); medianA = median(mylist[[grpA]]); 
  meanB = mean(mylist[[grpB]]); sdB = sd(mylist[[grpB]]); medianB = median(mylist[[grpB]]); 
  ratio = meanA/meanB
  if (ratio < 1) { ratio = -1/ratio }
  ratioM = medianA/medianB
  if (ratioM < 1) { ratioM = -1/ratioM }
  cat("length =",lengthA,"\n")
  cat("length =",lengthB,"\n")
  cat("means =",meanA,"+/-",sdA,"\n")
  cat("means =",meanB,"+/-",sdB,"\n")
  cat("ratio (mean) =",ratio,"\n")
  cat("median =",medianA,"+/-",sdA,"\n")
  cat("median =",medianB,"+/-",sdB,"\n")
  cat("ratio (median) =",ratioM,"\n")
  cat("p =",p,"\n")
}

# Read all the grid search results
tmp1 = read.csv("data/score_board_LargeModel.csv")
tmp1["Model"] = "Original"

tmp2 = read.csv("data/score_board_SmallModel.csv")
tmp2["Model"] = "Small"

tmp3 = read.csv("data/score_board_MiniModel.csv")
tmp3["Model"] = "Mini"

dat = rbind(tmp1,tmp2,tmp3)
dim(dat)

# Use to print best configuration for each architecture
dat[which.max(dat$r),]
dat[dat$Model=="Original",][which.max(dat[dat$Model=="Original",]$r),]
dat[dat$Model=="Small",][which.max(dat[dat$Model=="Small",]$r),]
dat[dat$Model=="Mini",][which.max(dat[dat$Model=="Mini",]$r),]

# Include all runs
subset = rep(TRUE,nrow(dat))
tmp = get_mylist(subset,metric); mylist = tmp[[1]]; at = tmp[[2]]

# Plot performance across groups
pdf(paste("Performance_all_",metric,".pdf",sep=""),width=6,height=5,pointsize=10)
par(oma = c(0,0,0,0) + 0.5)
par(mar = c(2,8,1,10)+0.5, mgp=c(1.5,0.6,0))
par(mfrow=c(1,1))
bp = boxplot(mylist,outline = FALSE, horizontal = TRUE,yaxt="n",at=at, frame=FALSE,col="#CCCCCC",border="#666666", xlab=metric,main="All") #,ylim=c(0,1))
pars = par("usr")
for (i in 1:length(mylist)) {
  scatter = runif(length(mylist[[i]]))*.3-.15
  points(mylist[[i]],rep(at[i],length(mylist[[i]]))+scatter,pch=20,cex=.3,col="#000000CC")
  for (diff in 1:4) {
    if (i+diff<=length(mylist) & at[min(length(mylist),i+diff)]-at[i] == diff) {
      if (length(mylist[[i]])==0 | length(mylist[[i+diff]])==0) next; # Avoid empty categories
      pval = wilcox.test(mylist[[i]],mylist[[i+diff]], paired = TRUE)$p.value
      text(max(mylist[[i]])+0.2*(par("usr")[2]-par("usr")[1])*(diff-1),at[i],paste("p=",signif(pval,1),sep=""),xpd=NA,pos=4,cex=.8,col="#CCCCCC")
    }
  }
}
text(par("usr")[1]+.03*(par("usr")[2]-par("usr")[1]),at,bp$names,xpd=NA,srt=0,pos=2)
dev.off()

# Stats for paper
get_stats(mylist,"Lion","AdamW")

# Only Lion optimiser
subset = dat$Optimizer == "Lion"
tmp = get_mylist(subset,metric); mylist = tmp[[1]]; at = tmp[[2]]

pdf(paste("Performance_Lion_",metric,".pdf",sep=""),width=6,height=5,pointsize=10)
par(oma = c(0,0,0,0) + 0.5)
par(mar = c(2,8,1,10)+0.5, mgp=c(1.5,0.6,0))
par(mfrow=c(1,1))
bp = boxplot(mylist,outline = FALSE, horizontal = TRUE,yaxt="n",at=at, frame=FALSE,col="#CCCCCC",border="#666666", xlab=metric,main="Only Lion") #,ylim=c(0,1))
for (i in 1:length(mylist)) {
  scatter = runif(length(mylist[[i]]))*.3-.15
  points(mylist[[i]],rep(at[i],length(mylist[[i]]))+scatter,pch=20,cex=.3,col="#000000CC")
  for (diff in 1:4) {
    if (i+diff<=length(mylist) & at[min(length(mylist),i+diff)]-at[i] == diff) {
      if (length(mylist[[i]])==0 | length(mylist[[i+diff]])==0) next; # Avoid empty categories
      pval = wilcox.test(mylist[[i]],mylist[[i+diff]], paired = TRUE)$p.value
      text(max(mylist[[i]])+0.2*(par("usr")[2]-par("usr")[1])*(diff-1),at[i],paste("p=",signif(pval,1),sep=""),xpd=NA,pos=4,cex=.8,col="#CCCCCC")
    }
  }
}
text(par("usr")[1]+.03*(par("usr")[2]-par("usr")[1]),at,bp$names,xpd=NA,srt=0,pos=2)
dev.off()

# Statistical tests for paper
get_stats(mylist,"onehot","onehotWithInt")
get_stats(mylist,"onehot","onehotWithBoth")
get_stats(mylist,"onehotWithP","onehotWithInt")
get_stats(mylist,"onehotWithP","onehotWithBoth")
get_stats(mylist,"onehotWithN","onehotWithInt")
get_stats(mylist,"onehotWithN","onehotWithBoth")


# Exclude Lion optimiser
subset = dat$Optimizer != "Lion"
tmp = get_mylist(subset,metric); mylist = tmp[[1]]; at = tmp[[2]]

pdf(paste("Performance_AdamW_",metric,".pdf",sep=""),width=6,height=5,pointsize=10)
par(oma = c(0,0,0,0) + 0.5)
par(mar = c(2,8,1,10)+0.5, mgp=c(1.5,0.6,0))
par(mfrow=c(1,1))
bp = boxplot(mylist,outline = FALSE, horizontal = TRUE,yaxt="n",at=at, frame=FALSE,col="#CCCCCC",border="#666666", xlab=metric,main="Only AdamW") #,ylim=c(0,1))
for (i in 1:length(mylist)) {
  scatter = runif(length(mylist[[i]]))*.3-.15
  points(mylist[[i]],rep(at[i],length(mylist[[i]]))+scatter,pch=20,cex=.3,col="#000000CC")
  for (diff in 1:4) {
    if (i+diff<=length(mylist) & at[min(length(mylist),i+diff)]-at[i] == diff) {
      if (length(mylist[[i]])==0 | length(mylist[[i+diff]])==0) next; # Avoid empty categories
      pval = wilcox.test(mylist[[i]],mylist[[i+diff]], paired = TRUE)$p.value
      text(max(mylist[[i]])+0.2*(par("usr")[2]-par("usr")[1])*(diff-1),at[i],paste("p=",signif(pval,1),sep=""),xpd=NA,pos=4,cex=.8,col="#CCCCCC")
    }
  }
}
text(par("usr")[1]+.03*(par("usr")[2]-par("usr")[1]),at,bp$names,xpd=NA,srt=0,pos=2)
dev.off()

# Statistical tests for paper
get_stats(mylist,"Original","Mini")
get_stats(mylist,"Small","Mini")
get_stats(mylist,"onehot","onehotWithInt")
get_stats(mylist,"onehot","onehotWithBoth")
get_stats(mylist,"onehotWithP","onehotWithInt")
get_stats(mylist,"onehotWithP","onehotWithBoth")
get_stats(mylist,"onehotWithN","onehotWithInt")
get_stats(mylist,"onehotWithN","onehotWithBoth")

