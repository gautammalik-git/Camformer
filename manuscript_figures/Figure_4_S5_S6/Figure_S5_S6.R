
library(extrafont)

# Read results for top models on private test set
dat.pvt = read.csv("data/grid_res_all_models_pvt.csv",sep=",", stringsAsFactors = FALSE)

# Make colnames consistent
for (c in c("spearman","pearson")) {
  colnames(dat.pvt)[colnames(dat.pvt) == c] = paste("all_",c,sep="")
  colnames(dat.pvt)[colnames(dat.pvt) == paste(c,"s_score",sep="")] = paste("score_",c,sep="")
}

# Extract pearson and spearman column numbers
colp = which(grepl("_pearson",colnames(dat.pvt)) & !grepl("_r2",colnames(dat.pvt)))
cols = which(grepl("_spearman",colnames(dat.pvt)))

# Read baseline from file
# This data is from the official challenge paper
# https://doi.org/10.1038/s41587-024-02414-w
baseline = read.csv("data/sota.csv",sep="\t",stringsAsFactors = FALSE)
bl_models = unique(baseline$X.1)
bl_colors = paste(c("#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#000000"),"66",sep="")

##### Make boxplots for Figure S5 and S6 #####
# Specify what models and groups to plot
models = c("Leg","Orig","Mini","Small","Large")
groups = gsub("_pearson","",colnames(dat.pvt)[colp])

pdf("Individual_boxplots.pdf",height=10,width=6,pointsize=10)
par(oma = c(0,0,0,0) + 0.5)
par(mar = c(2,2,2,6)+0.5, mgp=c(1.5,0.6,0))
par(mfrow=c(5,2))
for (g in groups) {
  for (c in c("spearman","pearson")) {
    mylist = c()
    for (m in models) {
      if (c == "spearman" | g == "score") { # Avoid square if handling either Spearman correlation or scores
        mylist[[m]] = dat.pvt[dat.pvt$Model == m,paste(g,"_",c,sep="")]
      } else {
        mylist[[m]] = dat.pvt[dat.pvt$Model == m,paste(g,"_",c,sep="")]**2
      }
    }
    
    # Find global limits for plot and set formatting options
    lims = c(min(baseline[baseline$X==c,g],unlist(mylist)),
             max(baseline[baseline$X==c,g],unlist(mylist)))
    par(font.main=1, cex.main=1)
    
    # Make boxplot
    bp = boxplot(mylist,outline = FALSE, horizontal = FALSE,frame=FALSE,col=paste(c("#e41a1c","#984ea3","#666666","#666666","#666666"),"66",sep=""),border="#666666", xlab=paste(gsub("^pea","Pea",gsub("spe","Spe",c))," correlation",sep=""),main=g, font.main=1,ylim=lims)
    
    # Add points for replicates
    for (i in 1:length(mylist)) {
      scatter = runif(length(mylist[[i]]))*.3-.15
      points(rep(i,length(mylist[[i]]))+scatter,mylist[[i]],pch=20,cex=1,col="#000000CC",xpd=NA)
    }
    
    # Plot performance of baseline models
    bl = baseline[baseline$X==c,which(g == groups)+2]
    segments(.5,bl,5.5,bl,lty=2,col=bl_colors)
    text(5.5,bl,bl_models,xpd=NA,pos=4,col=bl_colors)
    
    # Retrieve baseline performance (Vaishnav, de Boer et al.)
    vaishnav = bl[length(bl)]
    
    # Calculate and display statistics compared to baseline
    all_FCs = c() # Relative improvement
    for (i in 1:length(mylist)) {
      p = t.test(x = mylist[[i]], mu = vaishnav)$p.value
      fc = mean(mylist[[i]]/vaishnav)
      if (fc < 1) { fc = -1/fc }
      
      # How much is the error reduced?
      neg_fc = 1+(1-mean(1-mylist[[i]])/(1-vaishnav))
      all_FCs = c(all_FCs,neg_fc*100-100)
      
      if (neg_fc < 1) { neg_fc = -1/neg_fc }
      
      text(i,lims[2],paste("p=",signif(p,1),"\n",signif(fc,4),"x","\n",signif(neg_fc,4),"x\n",round(mean(mylist[[i]]),4),"+/-",round(sd(mylist[[i]]),4),sep=""),xpd=NA,cex=.5,pos=3)
    }
    # Print relative improvements
    text(1:length(mylist),bp$stats[1,],paste(ifelse(all_FCs<0,"","+"),signif(all_FCs,2),"%",sep=""),xpd=NA,pos=1,col=c("#e41a1c","#984ea3","#666666","#666666","#666666"))
  }
}
dev.off()

##### Make boxplots for Figure 4 #####
# Specify what models and groups to plot
models = c("Leg","Orig")
groups = gsub("_pearson","",colnames(dat.pvt)[colp])

pdf("Individual_boxplots_mainFig.pdf",height=10,width=3.5,pointsize=10, fonts = "Arial")
par(oma = c(0,0,0,0) + 0.5)
par(mar = c(2,2,2,8)+0.5, mgp=c(1.5,0.6,0))
par(mfrow=c(5,2))
for (g in groups) {
  for (c in c("spearman","pearson")) {
    mylist = c()
    for (m in models) {
      if (c == "spearman" | g == "score") {
        mylist[[m]] = dat.pvt[dat.pvt$Model == m,paste(g,"_",c,sep="")]
      } else {
        mylist[[m]] = dat.pvt[dat.pvt$Model == m,paste(g,"_",c,sep="")]**2
      }
    }
    lims = c(min(baseline[baseline$X==c,g],unlist(mylist)),
             max(baseline[baseline$X==c,g],unlist(mylist)))
    bp = boxplot(mylist,outline = FALSE, horizontal = FALSE,frame=FALSE,col=c(bl_colors[1],bl_colors[4]),border="#666666", xlab=paste(gsub("^pea","Pea",gsub("spe","Spe",c))," correlation",sep=""),main=g,ylim=lims)
    for (i in 1:length(mylist)) {
      scatter = runif(length(mylist[[i]]))*.3-.15
      points(rep(i,length(mylist[[i]]))+scatter,mylist[[i]],pch=20,cex=1,col="#000000CC",xpd=NA)
    }
    bl = baseline[baseline$X==c,which(g == groups)+2]
    segments(.5,bl,2.5,bl,lty=2,col=bl_colors)
    text(2.5,bl,bl_models,xpd=NA,pos=4,col=bl_colors)
    vaishnav = bl[length(bl)]
    for (i in 1:length(mylist)) {
      p = t.test(x = mylist[[i]], mu = vaishnav)$p.value
      fc = mean(mylist[[i]]/vaishnav)
      if (fc < 1) { fc = -1/fc }
      
      # How much is the error reduced?
      neg_fc = 1+(1-mean(1-mylist[[i]])/(1-vaishnav))
      if (neg_fc < 1) { neg_fc = -1/neg_fc }
      
      text(i,lims[2],paste("p=",signif(p,1),"\n",signif(fc,4),"x","\n",signif(neg_fc,4),"x\n",round(mean(mylist[[i]]),4),"+/-",round(sd(mylist[[i]]),4),sep=""),xpd=NA,cex=.5,pos=3)
    }
  }
}
dev.off()
