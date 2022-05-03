if (!require("ggplot2")) {
  install.packages("ggplot2")
  library("ggplot2")
}

if (!require("egg")) {
  install.packages("egg")
  library("egg")
}

if (!require("patchwork")) {
  install.packages("patchwork")
  library("patchwork")
}

#Dataset2 visualization

TimeSeries2 <- read.csv(file = "TimeSeries2.csv",
                        header = TRUE, row.names = 1)

time <- 1:90000

plot1 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F1))
plot1 <- plot1 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF1, color = as.factor(LabelF1))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot2 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F2))
plot2 <- plot2 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF2, color = as.factor(LabelF2))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot3 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F3))
plot3 <- plot3 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF3, color = as.factor(LabelF3))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot4 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F4))
plot4 <- plot4 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF4, color = as.factor(LabelF4))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot5 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F5))
plot5 <- plot5 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF5, color = as.factor(LabelF5))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

layout <- "
A#B#C
#D#E#
"
Data2 <- wrap_plots(plot1, plot2, plot3, plot4, plot5, ncol = 3, nrow = 2,
                    design = layout) +
  plot_layout(guides = "collect")

#Dataset2 with WE anomalies visualization

WE_TS2_Class <- read.csv(file = "TimeSeries2_WE_Classification.csv",
                         header = TRUE, row.names = 1)
time <- 1:90000

plot12 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F1))
plot12 <- plot12 + geom_line(alpha = 0.3) +
  geom_point(data = WE_TS2_Class, mapping = aes(x = time, y = WE_TS2_Class[,1],
                                                color = as.factor(WE_TS2_Class[,1]))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot22 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F2))
plot22 <- plot22 + geom_line(alpha = 0.3) +
  geom_point(data = WE_TS2_Class, mapping = aes(x = time, y = WE_TS2_Class[,2],
                                                color = as.factor(WE_TS2_Class[,2]))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot32 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F3))
plot32 <- plot32 + geom_line(alpha = 0.3) +
  geom_point(data = WE_TS2_Class, mapping = aes(x = time, y = WE_TS2_Class[,3],
                                                color = as.factor(WE_TS2_Class[,3]))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot42 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F4))
plot42 <- plot12 + geom_line(alpha = 0.3) +
  geom_point(data = WE_TS2_Class, mapping = aes(x = time, y = WE_TS2_Class[,4],
                                                color = as.factor(WE_TS2_Class[,4]))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot52 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F5))
plot52 <- plot12 + geom_line(alpha = 0.3) +
  geom_point(data = WE_TS2_Class, mapping = aes(x = time, y = WE_TS2_Class[,5],
                                                color = as.factor(WE_TS2_Class[,5]))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

Data22 <- wrap_plots(plot12, plot22, plot32, plot42, plot52, ncol = 3, nrow = 2,
                     design = layout) +
  plot_layout(guides = "collect")