# Figure 5
# Scatterplots for Camformer applied to external data

library(extrafont)

# Plot each file in the data directory
files = system("ls data/*.csv.gz",intern=TRUE)

for (file in files) {
  dat = read.csv(file)
  dim(dat)
  name = gsub("data/","",gsub("_scatter.csv.gz","",file))
  
  pdf(paste(name,".pdf",sep=""),width=2.22,height=2.2,pointsize=10, fonts = "Arial")
  
  par(cex=1, oma = c(2.5,2.5,0,0) + 0.5)
  par(mar = c(.1,.1,0,0) + 0)
  par(mgp = c(1.6, 0.5, 0))
  
  plot(dat$y_true,dat$y_pred,pch=16,cex=.3, col="#00000044",xpd=NA, xlab="True expression",ylab="Predicted expression")
  
  n = dim(dat)[1]
  r = cor(dat$y_true,dat$y_pred)
  rho = cor(dat$y_true,dat$y_pred,method = "spearman")
  
  abline(lm(dat$y_pred ~ dat$y_true),col="#FF3333",lty=2)
  
  # Add some annotation to figure
  x = par("usr")[1]+0.2*(par("usr")[2]-par("usr")[1])
  text(x, par("usr")[4]-0.05*(par("usr")[4]-par("usr")[3]),paste("n=",n,sep=""))#,pos=4)
  text(x, par("usr")[4]-0.14*(par("usr")[4]-par("usr")[3]),paste("r=",round(r,3),sep=""))#,pos=4)
  text(x, par("usr")[4]-0.23*(par("usr")[4]-par("usr")[3]),paste("h=",round(rho,3),sep=""))#,pos=4)
  
  dev.off()
}

