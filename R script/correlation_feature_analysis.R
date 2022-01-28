# load basic packages
source('R script/basicpackage.R')

library("circlize", lib.loc="C:/R-4.0.0/library")
library("gtools", lib.loc="C:/R-4.0.0/library")

mise()




##=============Load data
result_conv1 <- read_csv("Data set/set1/result_conv1.csv", # low-level feature conv1
                         col_names = FALSE)

result_conv2 <- read_csv("Data set/set1/result_conv2.csv", # low-level featuer conv2
                         col_names = FALSE)


result_embedding <- read_csv("Data set/set1/result_embedding.csv", # high-level feature embedding
                             col_names = FALSE)

X_hand <- read_csv("Data set/set1/X_hand.csv") # physical features 
X_hand$pos_slope <- X_hand$pos_slope %>% replace_na(0)

result_Conv1_embedding1 <- read_csv("R script/result_Conv1_embedding1.csv", 
                                    col_names = FALSE)


# index <- sample(1:5000, 5000, replace=FALSE) # permutation test
# result_conv1_per <- result_conv1[index,]


mat1 <- cor(result_conv1, X_hand) # correlation coefficient
mat2 <- cor(result_conv2, X_hand) # correlation coefficient
mat3 <- cor(result_embedding, X_hand) # correlation coefficient

mat4 <- cor(result_Conv1_embedding1, X_hand) 


#p2 <- pheatmap(coeff_conv1, color = colorRampPalette(rev(brewer.pal(n = 7, name =
#                                                                       "PuOr")))(100),
#               cutree_rows = 2)





##======Plot conv1 features
mat <- mat4
col_mat = rand_color(length(mat), transparency = 0.5)
col_mat[mat < 0.39 & mat>-0.39] = "#00000000"
col_mat[mat > 0.39
        
        
        ] = "#fee6ce"
col_mat[mat > 0.59] = "#fdae6b"
col_mat[mat > 0.79] = "#e6550d"
#col_mat[mat > 0.8] = "#e6550d"
#col_mat[mat > 0.9] = "#a63603"

col_mat[mat < -0.39] = "#deebf7"
col_mat[mat < -0.59] = "#9ecae1"
col_mat[mat < -0.79] = "#3182bd"
#col_mat[mat < -0.8] = "#3182bd"
#col_mat[mat < -0.9] = "#08519c"

dim(col_mat) = dim(mat)  # to make sure it is a matrix

chordDiagram(mat, col = col_mat)
circos.clear()




# scatter plots
index <- which(mat1 == max(mat1), arr.ind = TRUE)
pair_corr <- data.frame(result_conv1[,index[1]], X_hand[,index[2]] )

range01 <- function(x){(x-min(x))/(max(x)-min(x))}
pair_corr[,1] <- range01(pair_corr[,1])
pair_corr[,2] <- range01(pair_corr[,2])


p1 <- ggplot(pair_corr, aes(x = spectralSlope, y=X32)) + theme_pubclean() +
  
  
  geom_point(fill = "#3182bd",
             colour = "#3182bd",
             alpha = 0.1,
             size=0.3) +
  
  stat_density_2d(geom = "polygon", 
                  aes(fill = "HD",alpha = ..level..),
                  bins = 20,linetype = 2) +
  
  scale_fill_manual(values = c("#e6550d")) +
  
  # geom_smooth(method=lm, se=FALSE, fullrange=TRUE,color ="grey") + # linear regression
  # geom_quantile(quantiles = 0.5,color = "red") + # quantitle regression
  

  
  labs(x= "Spectral slope",
       y = "Deep acoustic feature")


p1
  
  


index <- which(mat2 == max(mat2), arr.ind = TRUE)
pair_corr <- data.frame(result_conv2[,index[1]], X_hand[,index[2]] )

range01 <- function(x){(x-min(x))/(max(x)-min(x))}
pair_corr[,1] <- range01(pair_corr[,1])
pair_corr[,2] <- range01(pair_corr[,2])


p2 <- ggplot(pair_corr, aes(x = spectralSlope, y=X15)) + theme_pubclean() +
  
  stat_density_2d(geom = "polygon", 
                  aes(fill = "HD",alpha = ..level..),
                  bins = 20,linetype = 2) +
  
  scale_fill_manual(values = c("#e6550d")) +
  
  # geom_smooth(method=lm, se=FALSE, fullrange=TRUE,color ="grey") + # linear regression
  # geom_quantile(quantiles = 0.5,color = "red") + # quantitle regression
  
  geom_point(fill = "#3182bd",
             colour = "#3182bd",
             alpha = 0.1,
             size=0.3) +
  
  labs(x= "Spectral slope",
       y = "Deep acoustic feature")
  
p2



index <- which(mat3 == max(mat3), arr.ind = TRUE)
pair_corr <- data.frame(result_embedding[,index[1]], X_hand[,index[2]] )

range01 <- function(x){(x-min(x))/(max(x)-min(x))}
pair_corr[,1] <- range01(pair_corr[,1])
pair_corr[,2] <- range01(pair_corr[,2])


p3 <- ggplot(pair_corr, aes(x = ratioLGLA, y=X70)) + theme_pubclean() +
  
  geom_point(fill = "#3182bd",
             colour = "#3182bd",
             alpha = 0.1,
             size=0.3) +
  
  stat_density_2d(geom = "polygon", 
                  aes(fill = "HD",alpha = ..level..),
                  bins = 10,linetype = 2) +
  
  scale_fill_manual(values = c("#e6550d")) +
  
  # geom_smooth(method=lm, se=FALSE, fullrange=TRUE,color ="grey") + # linear regression
  # geom_quantile(quantiles = 0.5,color = "red") + # quantitle regression
  

  
  labs(x= "Ratio LG/LA",
       y = "Deep acoustic feature")
p3


##Correlation coefficient with LA, LC, LG

noise_indicator <- X_hand[ c("LA", "ratioLGLA", "ratioLCLA") ]
noise_indicator$ratioLGLA <- noise_indicator$ratioLGLA * noise_indicator$LA
noise_indicator$ratioLCLA <- noise_indicator$ratioLCLA * noise_indicator$LA



cor_conv1 <- cor(result_conv1, noise_indicator) %>%
  as.data.frame()


p4 <- ggplot(cor_conv1) + theme_pubclean() + 
  
  geom_point( aes( x = 1:32, y=sort(LA) ),
              colour = "#3182bd",
              size = 0.5) + 
  
  geom_point( aes( x = 1:32, y=sort(ratioLCLA) ),
              colour =  "#e6550d",
              size = 0.5) +
  
  geom_point( aes( x = 1:32, y=sort(ratioLGLA) ),
              colour = "#756bb1",
              size = 0.5) + 
  
  scale_y_continuous(limits = c(-1, 1)) +
  
  scale_x_continuous(limits = c(1, 32),
                     breaks = c(1, 16, 32)) +
  
  labs(x = "Featur id",
       y = "Pearson correlation coefficient")



p4



cor_conv2 <- cor(result_conv2, noise_indicator) %>%
  as.data.frame()


p5 <- ggplot(cor_conv2) + theme_pubclean() + 
  
  geom_point( aes( x = 1:16, y=sort(LA) ),
              colour= "#3182bd",
              size = 0.5) + 

  
  geom_point( aes( x = 1:16, y=sort(ratioLCLA) ),
              colour= "#e6550d",
              size = 0.5) +
  
  geom_point( aes( x = 1:16, y=sort(ratioLGLA) ),
              colour= "#756bb1" ,
              size = 0.5) + 
  
  scale_y_continuous(limits = c(-1, 1)) +
  scale_x_continuous(limits = c(1, 16),
                     breaks = c(1, 8, 16)) +
  
  labs(x = "Featur id",
       y = "Pearson correlation coefficient")



p5





cor_embedding <- cor(result_embedding, noise_indicator) %>%
  as.data.frame()


p6 <- ggplot(cor_embedding) + theme_pubclean() + 
  
  geom_point( aes( x = 1:128, y=sort(LA) ),
              colour= "#3182bd",
              size = 0.5) + 
  
  
  geom_point( aes( x = 1:128, y=sort(ratioLCLA) ),
              colour=  "#e6550d",
              size = 0.5) +
  
  geom_point( aes( x = 1:128, y=sort(ratioLGLA) ),
              colour= "#756bb1",
              size = 0.5) +
  
  scale_y_continuous(limits = c(-1, 1)) +
  scale_x_continuous(limits = c(1, 128),
                     breaks = c(1,  64,  128)) +
  
  labs(x = "Featur id",
       y = "Pearson correlation coefficient")
  


p6



spp1 <- ggarrange(p4,p5,p6,
                  labels = c("", ""),
                  align = c("v"),
                  ncol = 3,nrow = 1,
                  widths = c(1, 1),
                  heights = c(1, 1))
spp1


