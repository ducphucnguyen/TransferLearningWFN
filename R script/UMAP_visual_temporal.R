source('R script/basicpackage.R')

require("ggrepel")
library(scales)
mise()

UMAP_temporal <- read_csv("R script/UMAP_temporal.csv", 
                                          col_names = FALSE)


colnames(UMAP_temporal) <- c("UMAP1", "UMAP2")



HL_info <- read_csv("R:/CMPH-Windfarm Field Study/Duc Phuc Nguyen/4. Machine Listening/Deepfeature/Matlab_script/HL_info.csv") # Hallett info

#HL_noise_metric <- read_csv("Matlab_script/HL_noise_metric.csv") # read noise metric
#colnames(HL_noise_metric) <- c("datetime", "L","LA","LC","LG")

# add to big dataframe
UMAP_temporal$hour = as.factor(HL_info$hour)
UMAP_temporal$month = as.factor(HL_info$month)

# day-night definition
UMAP_temporal$daynight <- cut(as.integer(as.character(UMAP_temporal$hour)), 
                                      breaks = c(0, 6, 18, 23),
                                      include.lowest = TRUE,
                                      labels = c("night", "day", "night")
)

# season definition
UMAP_temporal$season <- cut(as.integer(as.character(UMAP_temporal$month)), 
                                    breaks = c(1, 2, 5, 8, 11, 13),
                                    include.lowest = TRUE,
                                    labels = c("Summer", "Autumn", "Winter", "Spring", "Summer")
)



df_hour <- UMAP_temporal %>% # calculate centroid of each hourly data points
  group_by(hour) %>% 
  summarize(X_centroid = mean(UMAP1),
            Y_centroid = mean(UMAP2))


# daynight characteristics
p1 <- ggplot(UMAP_temporal, aes(x=UMAP1, y=UMAP2, colour=daynight)) + theme_void() +
  
  stat_ellipse(geom = "polygon",
               aes(fill = daynight),
               size = 0.5,
               alpha = 0.1,
               level = 0.68) +
  
  geom_point(size=0.1,
             alpha=0.1
  ) +
  
  scale_colour_manual(values=c("#1c9099", "#e6550d")) +
  scale_fill_manual(values=c("#1c9099", "#e6550d")) #+
  
  #theme(legend.position = "none")

p1

#ggsave("UMAP_visual_daynight.png", width = 9, height = 9, units = "cm", dpi = 600)







p2 <- ggplot(UMAP_temporal, aes(x=UMAP1, y=UMAP2, 
                                colour=as.numeric(as.character(hour)) ) ) +  theme_void() +
  
  
  geom_point( shape = 19,
              size = 0.1,
              alpha = 0.15 ) +
  
  
  geom_point(data=df_hour, aes(x=X_centroid, y=Y_centroid, 
                               fill=as.numeric( as.character(hour) ) ),
             shape=21,
             size=2,
             colour = "white",
             alpha=1.0) +
  
  geom_text_repel(data=df_hour, aes(x=X_centroid, y=Y_centroid, label=hour),
                  max.overlaps=30, 
                  size=2.5,
                  colour="#636363") +

  scale_colour_gradientn(colours = c("#1c9099", "#bdc9e1", "#fd8d3c", "#bdc9e1", "#1c9099"),
                       values = c(0, 0.2, 0.5, 0.8, 1)) +
  
  scale_fill_gradientn(colours = c("#02818a", "#bdc9e1", "#d94701", "#bdc9e1", "#02818a"),
                       values = c(0, 0.2, 0.5, 0.8, 1)) +
  
  ylim(-6, 10) +
  xlim(-6, 10)# +
  
  #theme(legend.position = "none") 
  

p2

#ggsave("UMAP_visual_daynight.png", width = 9, height = 9, units = "cm", dpi = 600)



p2_1 <- ggplot(UMAP_temporal, aes(x=UMAP1, y=UMAP2) ) + theme_void() +
  
  stat_ellipse(
             aes(fill = daynight, colour = daynight),
             geom = "polygon",
             size = 0.5,
             alpha = 0.1,
             level = 0.68) +
  
  scale_colour_manual(values=c("#1c9099", "#fd8d3c")) +
  scale_fill_manual(values=c("#1c9099", "#fd8d3c")) +
  
  ylim(-6, 10) +
  xlim(-6, 10) +
  
  theme(legend.position = "none")

p2_1

#ggsave("UMAP_visual_daynight_elip.pdf", width = 9, height = 9, units = "cm")




# seasonal pattern
p3 <- ggplot(UMAP_temporal, aes(x=UMAP1, y=UMAP2, colour=season, fill=season)) + theme_void() +
  
  stat_ellipse(
    geom = "polygon",
    size = 0.5,
    alpha = 0.1,
    level = 0.68) +
  
  geom_point( shape = 19,
              size = 0.1,
              alpha = 0.2 ) +
  
  ylim(-6, 10) +
  xlim(-6, 10) +
  
  scale_colour_manual(values=c("#fdc086", "#ffff99", "#7fc97f", "#beaed4")) #+
  
  #theme(legend.position = "none")

p3


ggsave("UMAP_visual_season.png", width = 9, height = 9, units = "cm", dpi = 600)

