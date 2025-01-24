# Figure 3a and Figure S3a
# Creates scatterplots

library(extrafont)

dat = read.csv("data/data_scatter_O_Mi.csv.gz")

pdf("scatter_original.pdf",width=3,height=3,pointsize=10, fonts = "Arial")

# Specify layout as
#     [,1] [,2] [,3]
#[1,]    3    3    2
#[2,]    1    1    4
#[3,]    1    1    4
layout(matrix(c(3,1,1,3,1,1,2,4,4),3,3),widths=c(2.5,2.5,1.1),heights=c(1.1,2.5,2.5))
par(cex=1, oma = c(2.5,2.5,0,0) + 0.5); par(mar = c(.1,.1,0,0) + 0); par(mgp = c(1.6, 0.5, 0))

plot(dat$true_expr,dat$orig_model_expr,pch=20,col="#00000022",type="n",xpd=NA, xlab="True expression",ylab="Predicted expression") # Use type="n" to produce an empty plot with correct axes

# Calculate with and height of inner plot area
coords = par("usr")
gx <- grconvertX(coords[1:2], "user", "inches")
gy <- grconvertY(coords[3:4], "user", "inches")
width <- max(gx) - min(gx)
height <- max(gy) - min(gy)

# Produce a rasterised plot with data points in inner plot area (tmp.png)
system("rm tmp.png")
png("tmp.png", width = width, height = height, units = "in", res = 600, bg = "transparent")
par(oma = c(0,0,0,0) + 0)
par(mar = c(0,0,0,0), mgp=c(2.2, 1, 0), las=0)
plot.new()
plot.window(coords[1:2], coords[3:4], mar = c(0,0,0,0), xaxs = "i", yaxs = "i")
plot(dat$true_expr,dat$orig_model_expr,pch=20,col="#00000022",cex=.2)
dev.off()

# Add raster image to existing plot
panel <- png::readPNG("tmp.png")
rasterImage(panel, coords[1], coords[3], coords[2], coords[4])

n = dim(dat)[1]
r = cor(dat$true_expr,dat$orig_model_expr)
rho = cor(dat$true_expr,dat$orig_model_expr,method = "spearman")

abline(lm(dat$orig_model_expr ~ dat$true_expr),col="#FF3333",lty=2)

#x = par("usr")[1]+0*(par("usr")[2]-par("usr")[1])
x = par("usr")[1]+0.2*(par("usr")[2]-par("usr")[1])
text(x, par("usr")[4]-0.05*(par("usr")[4]-par("usr")[3]),paste("n=",n,sep=""))#,pos=4)
text(x, par("usr")[4]-0.11*(par("usr")[4]-par("usr")[3]),paste("r=",round(r,3),sep=""))#,pos=4)
text(x, par("usr")[4]-0.17*(par("usr")[4]-par("usr")[3]),paste("h=",round(rho,3),sep=""))#,pos=4)

# Empty plot
plot.new()

# Barplot
xhist = hist(dat$true_expr,nclass=100,plot=FALSE)
barplot(xhist$counts, axes = TRUE, space = 0, horiz=FALSE, ylab="Counts", border=NA,col="#CCCCCC",xpd=NA)

# Barplot
yhist = hist(dat$orig_model_expr,nclass=100,plot=FALSE)
barplot(yhist$counts, axes = TRUE, space = 0, horiz=TRUE, xlab= "Counts", border=NA,col="#CCCCCC",xpd=NA)

dev.off()


####

pdf("scatter_mini.pdf",width=3,height=3,pointsize=10, fonts = "Arial")

# Specify layout as
#     [,1] [,2] [,3]
#[1,]    3    3    2
#[2,]    1    1    4
#[3,]    1    1    4
layout(matrix(c(3,1,1,3,1,1,2,4,4),3,3),widths=c(2.5,2.5,1.1),heights=c(1.1,2.5,2.5))
par(cex=1, oma = c(2.5,2.5,0,0) + 0.5); par(mar = c(.1,.1,0,0) + 0); par(mgp = c(1.6, 0.5, 0))

plot(dat$true_expr,dat$mini_model_expr,pch=20,col="#00000022",type="n",xpd=NA, xlab="True expression",ylab="Predicted expression") # Use type="n" to produce an empty plot with correct axes

# Calculate with and height of inner plot area
coords = par("usr")
gx <- grconvertX(coords[1:2], "user", "inches")
gy <- grconvertY(coords[3:4], "user", "inches")
width <- max(gx) - min(gx)
height <- max(gy) - min(gy)

# Produce a rasterised plot with data points in inner plot area (tmp.png)
system("rm tmp.png")
png("tmp.png", width = width, height = height, units = "in", res = 600, bg = "transparent")
par(oma = c(0,0,0,0) + 0)
par(mar = c(0,0,0,0), mgp=c(2.2, 1, 0), las=0)
plot.new()
plot.window(coords[1:2], coords[3:4], mar = c(0,0,0,0), xaxs = "i", yaxs = "i")
plot(dat$true_expr,dat$mini_model_expr,pch=20,col="#00000022",cex=.2)
dev.off()

# Add raster image to existing plot
panel <- png::readPNG("tmp.png")
rasterImage(panel, coords[1], coords[3], coords[2], coords[4])

n = dim(dat)[1]
r = cor(dat$true_expr,dat$mini_model_expr)
rho = cor(dat$true_expr,dat$mini_model_expr,method = "spearman")

abline(lm(dat$mini_model_expr ~ dat$true_expr),col="#FF3333",lty=2)

#x = par("usr")[1]+0*(par("usr")[2]-par("usr")[1])
x = par("usr")[1]+0.2*(par("usr")[2]-par("usr")[1])
text(x, par("usr")[4]-0.05*(par("usr")[4]-par("usr")[3]),paste("n=",n,sep=""))#,pos=4)
text(x, par("usr")[4]-0.11*(par("usr")[4]-par("usr")[3]),paste("r=",round(r,3),sep=""))#,pos=4)
text(x, par("usr")[4]-0.17*(par("usr")[4]-par("usr")[3]),paste("h=",round(rho,3),sep=""))#,pos=4)

# Empty plot
plot.new()

# Barplot
xhist = hist(dat$true_expr,nclass=100,plot=FALSE)
barplot(xhist$counts, axes = TRUE, space = 0, horiz=FALSE, ylab="Counts", border=NA,col="#CCCCCC",xpd=NA)

# Barplot
yhist = hist(dat$mini_model_expr,nclass=100,plot=FALSE)
barplot(yhist$counts, axes = TRUE, space = 0, horiz=TRUE, xlab= "Counts", border=NA,col="#CCCCCC",xpd=NA)

dev.off()

