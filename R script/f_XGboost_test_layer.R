source('R script/basicpackage.R')

mise()


df <- read_csv("R script/f_XGboost_test_layer.csv")

df_long <- gather(df, feature, AUC , conv1:conv1_embedding, factor_key=TRUE) 



ggplot(df_long, aes(x=feature, y=AUC, colour=feature)) + theme_pubclean() +
  
  geom_jitter( position=position_jitter(0.1),
               shape=19,
                size=1,
               alpha=0.5) +
  
  
  stat_summary(fun=mean, geom="point", shape=21,
               size=3, color="#cccccc", fill="#525252") +
  
  
  scale_y_continuous( limits = c(0.73, 0.87),
                      breaks = c(0.75, 0.8, 0.85)) +
  
  scale_color_brewer(palette="Dark2") +
  
  
  theme(legend.position = "none")
  
  
