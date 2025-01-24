# Figure 8b, 8c
# Show mean expression (across ten replicate models) for individual motifs and motif co-occurrences

library(pheatmap)
library(viridis)
library(extrafont)

# Read data
dat = read.csv("data/combinations.train.fc.txt",stringsAsFactors = FALSE,sep="\t")

# Extract unique motifs and count them
motifs = unique(dat$Motif1)
N = length(motifs)

# Create an expression matrix and populate this; we are doing this per row for speed
real = matrix(ncol=N,nrow=N)
colnames(real) = motifs
rownames(real) = motifs
pred = pred.N = N_mat = real

for (i in 1:length(motifs)) {
  dat.sub = dat[dat$Motif1==motifs[i],]
  real[motifs[i],dat.sub$Motif2] = dat.sub$Mean_real
  pred[motifs[i],dat.sub$Motif2] = dat.sub$Mean_pred
  N_mat[motifs[i],dat.sub$Motif2] = dat.sub$N
}
N_mat[is.na(N_mat)] = ""

number_of_colors = 40

# Calculate expression order for real and predicted values
order_real = order(sapply(1:ncol(real),function (x) real[x,x]), decreasing = TRUE)
order_pred = order(sapply(1:ncol(pred),function (x) pred[x,x]), decreasing = TRUE)

width = 1.5 + 2.5/19*nrow(pred)
height = 1 + 2.5/19*nrow(pred)

# Plot expression values with 20 colours
pheatmap(real[order_real,order_real],cluster_rows=FALSE,cluster_cols=FALSE,color = inferno(number_of_colors),border=NA, na_col=NA, display_numbers = N_mat[order_real,order_real], fontsize_number=3, fontsize = 7, fontfamily = "Arial", width = width, height = height, filename = "Heatmap_expression_real.pdf")
pheatmap(real[order_real,order_real],cluster_rows=FALSE,cluster_cols=FALSE,color = inferno(number_of_colors),border=NA, na_col=NA, fontsize = 7, fontfamily = "Arial", width = width/1.3, height = height/1.3, filename = "Heatmap_expression_real_no_numbers.pdf")
pheatmap(pred[order_pred,order_pred],cluster_rows=FALSE,cluster_cols=FALSE,color = inferno(number_of_colors),border=NA, na_col=NA, display_numbers = N_mat[order_pred,order_pred], fontsize_number=3, fontsize = 7, fontfamily = "Arial", width = width, height = height, filename = "Heatmap_expression_pred.pdf")
pheatmap(pred[order_pred,order_pred],cluster_rows=FALSE,cluster_cols=FALSE,color = inferno(number_of_colors),border=NA, na_col=NA, fontsize = 7, fontfamily = "Arial", width = width/1.3, height = height/1.3, filename = "Heatmap_expression_pred_no_numbers.pdf")

paletteLength <- 40
myColor <- colorRampPalette(c("darkblue", "white", "darkorange"))(paletteLength)

# Plot differences
diagonal = sapply(1:dim(pred)[1],function(x) pred[order_pred,order_pred][x,x])
test = pred[order_pred,order_pred]-diagonal
top = max(max(test),-min(test))
breaksList <- c(seq(-top, 0, length.out=ceiling(paletteLength/2) + 1), seq(top/paletteLength, top, length.out=floor(paletteLength/2)))
pred_diff = pred[order_pred,order_pred]-diagonal
pheatmap(pred_diff,cluster_rows=FALSE,cluster_cols=FALSE,breaks=breaksList, color = myColor,border=NA, na_col=NA, display_numbers = N_mat[order_pred,order_pred], fontsize_number=3, fontsize = 7, fontfamily = "Arial", width = width, height = height, filename = "Heatmap_expression_pred_difference.pdf")
pheatmap(pred_diff,cluster_rows=FALSE,cluster_cols=FALSE,breaks=breaksList, color = myColor,border=NA, na_col=NA, fontsize = 7, fontfamily = "Arial", width = width/1.3, height = height/1.3, filename = "Heatmap_expression_pred_difference_no_numbers.pdf")




