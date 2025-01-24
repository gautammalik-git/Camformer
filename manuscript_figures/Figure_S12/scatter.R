# Figure S12
# Produces a scatterplot between GC content and gene expression

library(extrafont)

# Read data
dat = read.csv("data/GC_content_nt_effect_snv_test10k.csv")
dim(dat)

pdf("Scatter.pdf",width=2.22,height=2.2,pointsize=10, fonts = "Arial")
par(cex=1, oma = c(2.5,2.5,0,0) + 0.5)
par(mar = c(.1,.1,0,0) + 0)
par(mgp = c(1.6, 0.5, 0))

plot(dat$GC.Content....,dat$Expr,pch=16,cex=.5, col="#00000016",xpd=NA, xlab="GC content (%)",ylab="Predicted expression")

n = dim(dat)[1]
r = cor(dat$GC.Content....,dat$Expr)
rho = cor(dat$GC.Content....,dat$Expr,method = "spearman")

abline(lm(dat$Expr ~ dat$GC.Content....),col="#FF3333",lty=2)

# Add some annotation
x = par("usr")[1]+0.2*(par("usr")[2]-par("usr")[1])
text(x, par("usr")[4]-0.05*(par("usr")[4]-par("usr")[3]),paste("n=",n,sep=""))#,pos=4)
text(x, par("usr")[4]-0.14*(par("usr")[4]-par("usr")[3]),paste("r=",round(r,3),sep=""))#,pos=4)
text(x, par("usr")[4]-0.23*(par("usr")[4]-par("usr")[3]),paste("h=",round(rho,3),sep=""))#,pos=4)

dev.off()

