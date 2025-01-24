# Figure 1b
# Histogram over expression values in training data

# Use the following command to download the training data as "train_sequences.txt":
# mkdir -p data
# wget https://zenodo.org/records/7395397/files/train_sequences.txt?download=1 -O data/train_sequences.txt
dat = read.csv("data/train_sequences.txt",sep="\t",header=FALSE)

pdf("histogram.pdf",width=2.3,height=1.5,pointsize=10)
par(oma = c(0,0,0,0) + 0.5)
par(mar = c(2.5,2.5,0,0)+0, mgp=c(1.5,0.6,0))

tmp = hist(dat$V2,nclass=100,border=NA,col="#666666",xaxt="n",xlab="Expression",ylab="Count",main="") # Make histogram

shift = tmp$mids[1] # Correct positions for midpoint of bars
axis(1,seq(0,max(dat$V2),1),at=seq(-shift,max(dat$V2)-shift,1)) # Draw axis manually to add all ticks

dev.off()
