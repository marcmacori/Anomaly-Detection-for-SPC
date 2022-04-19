if (!require("ggplot2")) {
  install.packages("ggplot2")
  library("ggplot2")
}

if (!require("egg")) {
  install.packages("egg")
  library("egg")
}

theme_linedraw(
  base_size = 11,
  base_family = "",
  base_line_size = base_size/22,
  base_rect_size = base_size/22
)

#Dataset1 Visualization
TimeSeries1 <- read.csv(file = 'TimeSeries1.csv',
                        header = TRUE, row.names = 1)
TimeSeries1_Classification <- read.csv(file = 'TimeSeries1_Classification.csv',
                        header = TRUE, row.names = 1)

TS1Vis <- t(TimeSeries1)
TS1ClassVis <- t(TimeSeries1_Classification)


function (Data, V)
plot1 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F1))
plot1 <- plot1 + geom_line (alpha = 0.3) +
  geom_point(aes(x= time, y = LabelF1, color = as.factor(LabelF1))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()



#Dataset2 Visualization

TimeSeries2 <- read.csv(file = 'TimeSeries2.csv',
                        header = TRUE, row.names = 1)

time = 1:2800

plot1 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F1))
plot1 <- plot1 + geom_line (alpha = 0.3) +
  geom_point(aes(x= time, y = LabelF1, color = as.factor(LabelF1))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot2 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F2))
plot2 <- plot2 + geom_line (alpha = 0.3) +
  geom_point(aes(x= time, y = LabelF2, color = as.factor(LabelF2))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot3 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F3))
plot3 <- plot3 + geom_line (alpha = 0.3) +
  geom_point(aes(x= time, y = LabelF3, color = as.factor(LabelF3))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot4 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F4))
plot4 <- plot4 + geom_line (alpha = 0.3) +
  geom_point(aes(x= time, y = LabelF4, color = as.factor(LabelF4))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot5 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F5))
plot5 <- plot5 + geom_line (alpha = 0.3) +
  geom_point(aes(x= time, y = LabelF5, color = as.factor(LabelF5))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()
  
  
ggarrange(plot1, plot2,plot3, plot4, plot5, ncol = 3, nrow = 2)

  