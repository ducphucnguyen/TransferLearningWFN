# load basic packages
source('R script/basicpackage.R')

mise()


Yclass <- read_csv("Data set/set1/Yclass.csv", 
                   col_names = FALSE, col_types = cols(X1 = col_double()))

yamnet_class_map <- read_csv("YAMnet/yamnet/yamnet_class_map.csv")

X1_info <- read_csv("C:/Users/nguy0936/.julia/dev/DeepAMdetection/data/set1/X1_info.csv")

Yall <- cbind(Yclass, X1_info)


# Top 10 noise types at H1
count_type <- as.data.frame(table(unlist(Yall$X1)))
count_type <- count_type[order(-count_type$Freq),] 








## Top 10 noise types at H1 (Hallett)

yamnet_class_map$display_name[500]

ggplot(Yall, aes(y=X1, x = Hour)) +
  
  geom_point()

