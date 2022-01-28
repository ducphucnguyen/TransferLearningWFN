source('R script/basicpackage.R')


mise()

UMAP_spatial <- read_csv("R script/UMAP_spatial.csv", 
                         col_names = FALSE)


colnames(UMAP_spatial) <- c("UMAP1", "UMAP2")


locs <- c(1*replicate(5000, 1), 
          2*replicate(3000, 1), 
          3*replicate(1000, 1), 
          4*replicate(1000, 1))

UMAP_spatial$locs <- as.factor( locs )





# Noise characteristics at day and night are revealed
p1 <- ggplot(UMAP_spatial, aes(x=UMAP1, y=UMAP2, colour=locs)) + theme_void() +
  
  stat_ellipse(geom = "polygon",
               aes(fill = locs), 
               alpha = 0.1,
               level = 0.68) +
  
  geom_point(size=0.1,
             alpha=0.7
  ) 
  
  
  #scale_colour_manual(values=c("#fdc086", "#ffff99", "#7fc97f", "#beaed4")) +
  
  #theme(legend.position = "none") 

p1

#ggsave("UMAP_visual_spatial.png", width = 9, height = 9, units = "cm", dpi = 600)





